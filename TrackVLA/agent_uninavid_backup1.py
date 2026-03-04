# agent_uninavid.py
import os
import os.path as osp
import json
import re
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import cv2
import imageio

import habitat
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.utils.visualizations import maps
from habitat_sim.gfx import LightInfo, LightPositionModel

from tqdm import trange

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria


# ---------------------------
#  UniNaVid-style dataset logger
# ---------------------------
class TrackDatasetLogger:
    """
    Write UniNaVid-style training samples, aligned with open_uninavid_sampled_500.json:
      {
        "id": "...",
        "video": "nav_videos/xxx.mp4",
        "conversations": [{"from":"human","value":...},{"from":"gpt","value":"left forward ..."}],
        "extra": {...}   # optional
      }

    Output layout under dataset_root:
      dataset_root/
        track_dataset.jsonl
        nav_videos/
          TRACK_ID_xxx.mp4
    """

    def __init__(
        self,
        dataset_root: str,
        jsonl_name: str = "track_dataset.jsonl",
        fps: int = 1,
        max_history_frames: int = 32,
    ):
        self.dataset_root = dataset_root
        self.jsonl_path = osp.join(dataset_root, jsonl_name)
        self.fps = int(fps)
        self.max_history_frames = int(max_history_frames) if max_history_frames is not None else None

        os.makedirs(self.dataset_root, exist_ok=True)
        os.makedirs(osp.join(self.dataset_root, "nav_videos"), exist_ok=True)

    @staticmethod
    def _ensure_uint8_rgb(frame: np.ndarray) -> np.ndarray:
        # Habitat rgb observations are typically uint8 already; keep robust.
        x = np.asarray(frame)
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        if x.ndim == 3 and x.shape[2] > 3:
            x = x[:, :, :3]
        return x

    def log_inference(
        self,
        frames: List[np.ndarray],
        prompt: str,
        actions_str: str,
        episode: NavigationEpisode,
        infer_idx: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if frames is None or len(frames) == 0:
            print(
                f"[WARN] Empty frames, skip logging. episode_id={getattr(episode,'episode_id',None)} infer_idx={infer_idx}"
            )
            return

        if self.max_history_frames is not None and len(frames) > self.max_history_frames:
            frames = frames[-self.max_history_frames :]

        frames = [self._ensure_uint8_rgb(f) for f in frames]

        scene_id = getattr(episode, "scene_id", "unknown_scene")
        ep_id = getattr(episode, "episode_id", "unknown_episode")
        scene_key = osp.splitext(osp.basename(scene_id))[0].split(".")[0]

        sample_id = f"TRACK_ID_{scene_key}_{ep_id}_{infer_idx:04d}"
        video_rel = osp.join("nav_videos", f"{sample_id}.mp4").replace("\\", "/")
        video_abs = osp.join(self.dataset_root, video_rel)

        # Save clip
        imageio.mimsave(video_abs, frames, fps=self.fps)

        # Save jsonl record
        rec = {
            "id": sample_id,
            "video": video_rel,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": actions_str},
            ],
        }
        if extra:
            rec["extra"] = extra

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------
#  Scene existence utilities (skip MP3D if missing)
# ---------------------------
def _resolve_scenes_dir(config) -> str:
    # Allow override
    env_dir = os.environ.get("HABITAT_SCENES_DIR", "").strip()
    if env_dir:
        return env_dir

    try:
        ds = getattr(getattr(config, "habitat", None), "dataset", None)
        scenes_dir = getattr(ds, "scenes_dir", None) if ds is not None else None
        if scenes_dir:
            return scenes_dir
    except Exception:
        pass

    # TrackVLA default
    return "data/scene_datasets"


def _scene_exists(scenes_dir: str, scene_id: str) -> bool:
    if not scene_id:
        return False
    if osp.isabs(scene_id):
        return osp.exists(scene_id)
    return osp.exists(osp.join(scenes_dir, scene_id))


# ---------------------------
#  Evaluation / data collection entry
# ---------------------------
def evaluate_agent(config, model_path, dataset_split, save_path) -> None:
    """
    Called by render_train_split.py:
      evaluate_agent(config, model_path, dataset_split, save_path)

    Outputs:
      save_path/<scene_key>/<episode_id>.json           (episode summary)
      save_path/<scene_key>/<episode_id>_info.json      (per-step distances/facing)
      save_path/<scene_key>/<episode_id>.mp4            (debug video w/ topdown map text overlay) [if enabled]

      save_path/track_dataset.jsonl
      save_path/nav_videos/*.mp4                        (UniNaVid-style clips)
    """

    os.makedirs(save_path, exist_ok=True)

    # Robust: filter missing scenes here too (in case caller forgot)
    scenes_dir = _resolve_scenes_dir(config)
    if hasattr(dataset_split, "episodes") and dataset_split.episodes is not None:
        kept, dropped = [], 0
        for ep in dataset_split.episodes:
            if _scene_exists(scenes_dir, ep.scene_id):
                kept.append(ep)
            else:
                dropped += 1
        if dropped > 0:
            print(f"[INFO] Filtered episodes in evaluate_agent: kept={len(kept)} dropped={dropped}")
        dataset_split.episodes = kept

    agent = UniNaVid_Agent(model_path=model_path, save_root=save_path, exp_save="video")

    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        sim = env.sim
        agent.reset()

        num_episodes = len(env.episodes)
        for _ in trange(num_episodes):
            try:
                env.reset()
            except Exception as e:
                print(f"[WARN] env.reset failed; skip episode. err={repr(e)}")
                continue

            agent.start_episode(env.current_episode)

            # IMPORTANT: update instruction every episode (your old first_init was wrong)
            instruction = ""
            if hasattr(env.current_episode, "info") and isinstance(env.current_episode.info, dict):
                instruction = env.current_episode.info.get("instruction", "")
            if not instruction:
                instruction = "Pursue the first individual in your path."

            # Keep your light setup
            light_setup = [
                LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
            ]
            sim.set_light_setup(light_setup)

            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            iter_step = 0
            followed_step = 0
            too_far_count = 0
            status = "Normal"
            finished = False
            record_infos: List[Dict[str, Any]] = []
            result: Dict[str, Any] = {}

            info = env.get_metrics()

            while not env.episode_over:
                obs = sim.get_sensor_observations()

                # Your old code used env.task._get_observations(...) which is unnecessary and may crash;
                # remove it entirely.
                action_base_vel = agent.act(obs, info, instruction, env.current_episode.episode_id)

                action_dict = {
                    "action": (
                        "agent_0_humanoid_navigate_action",
                        "agent_1_base_velocity",
                        "agent_2_oracle_nav_randcoord_action_obstacle",
                        "agent_3_oracle_nav_randcoord_action_obstacle",
                        "agent_4_oracle_nav_randcoord_action_obstacle",
                        "agent_5_oracle_nav_randcoord_action_obstacle",
                    ),
                    "action_args": {"agent_1_base_vel": action_base_vel},
                }

                iter_step += 1
                env.step(action_dict)
                info = env.get_metrics()

                # Metrics & termination heuristics
                if float(info.get("human_following", 0.0)) == 1.0:
                    followed_step += 1
                    too_far_count = 0
                else:
                    # not facing/ following
                    pass

                dist_to_human = float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos))
                if dist_to_human > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        status = "Lost"
                        finished = False
                        break

                record_infos.append(
                    {
                        "step": iter_step,
                        "dis_to_human": dist_to_human,
                        "facing": float(info.get("human_following", 0.0)),
                    }
                )

                if float(info.get("human_collision", 0.0)) == 1.0:
                    status = "Collision"
                    finished = False
                    break

            info = env.get_metrics()
            agent.reset(env.current_episode)

            if env.episode_over:
                finished = True

            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split(".")[0]
            save_dir = osp.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)

            with open(osp.join(save_dir, f"{env.current_episode.episode_id}_info.json"), "w", encoding="utf-8") as f:
                json.dump(record_infos, f, indent=2, ensure_ascii=False)

            result["finish"] = finished
            result["status"] = status

            if iter_step > 0:
                if iter_step < 300:
                    result["success"] = bool(info.get("human_following_success", 0.0) and info.get("human_following", 0.0))
                else:
                    result["success"] = bool(info.get("human_following", 0.0))
                result["following_rate"] = float(followed_step / iter_step)
            else:
                result["success"] = False
                result["following_rate"] = 0.0

            result["following_step"] = followed_step
            result["total_step"] = iter_step
            result["collision"] = float(info.get("human_collision", 0.0))

            with open(osp.join(save_dir, f"{env.current_episode.episode_id}.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)


# ---------------------------
#  UniNaVid agent
# ---------------------------
class UniNaVid_Agent(Agent):
    """
    Core requirements you asked for:
      1) Run UniNaVid as the controller to collect trajectories under TrackVLA train config
      2) One inference -> 4 discrete tokens -> execute 1st, cache next 3 in pending_action_list
      3) Save UniNaVid-style dataset (jsonl + mp4 clips) for later distillation training
      4) Work even when MP3D missing (handled in evaluate_agent / render_train_split filtering)
    """

    _ALLOWED = ("forward", "left", "right", "back", "stop")

    def __init__(self, model_path: str, save_root: str, exp_save: str = "video"):
        print("Initialize UniNaVid Agent")

        self.save_root = save_root
        self.require_map = True if "video" in exp_save else False
        self.require_data = True if "video" in exp_save else False

        os.makedirs(self.save_root, exist_ok=True)

        self.conv_mode = "vicuna_v1"
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, get_model_name_from_path(model_path)
        )

        self.promt_template = (
            "Imagine you are a robot programmed for navigation tasks. "
            "You have been given a video of historical observations and an image of the current observation <image>. "
            "Your assigned task is: '{}'. "
            "Analyze this series of images to determine your next four actions. "
            "The predicted action should be one of the following: forward, left, right, back, or stop."
        )

        # History used as video input to UniNaVid (will be cleared inside predict_inference)
        self.rgb_list: List[np.ndarray] = []
        # Debug overlay video
        self.topdown_map_list: List[np.ndarray] = []

        self.current_episode: Optional[NavigationEpisode] = None
        self.step_idx = 0
        self.infer_idx = 0
        self.pending_action_list: List[str] = []

        # Write dataset directly under save_root (aligned with build_uninavid_track_dataset.py)
        self.logger = TrackDatasetLogger(dataset_root=self.save_root, fps=1, max_history_frames=32)

        self.reset()

    def start_episode(self, episode: NavigationEpisode):
        self.current_episode = episode
        self.step_idx = 0
        self.infer_idx = 0
        self.pending_action_list = []
        self.rgb_list = []
        self.topdown_map_list = []

    def process_images(self, rgb_list: List[np.ndarray]):
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors="pt")["pixel_values"].half().cuda()
        return [video]

    def predict_inference(self, prompt: str) -> str:
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"

        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs.replace("<image>", "")
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "")

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]

        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx : idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1 :]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        # IMPORTANT: clears history after feeding into model
        self.rgb_list = []

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff > 0:
            print(f"[Warning] {n_diff} output_ids differ from input_ids prefix")

        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()
        return outputs

    @staticmethod
    def _normalize_four_tokens(raw: str) -> List[str]:
        tokens = re.findall(r"(forward|left|right|back|stop)", (raw or "").lower())
        if len(tokens) == 0:
            tokens = ["stop"]
        tokens = tokens[:4]
        while len(tokens) < 4:
            tokens.append("stop")
        return tokens

    @staticmethod
    def _token_to_base_vel(tok: str) -> List[float]:
        tok = (tok or "").lower()
        if tok == "stop":
            return [0.0, 0.0, 0.0]
        if tok == "forward":
            return [0.5, 0.0, 0.0]
        if tok == "left":
            return [0.0, 0.0, 1.0]
        if tok == "right":
            return [0.0, 0.0, -1.0]
        if tok == "back":
            return [-0.67, 0.0, 0.0]
        return [0.0, 0.0, 0.0]

    def addtext(self, image: np.ndarray, instruction: str, navigation: str) -> np.ndarray:
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instruction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2
        y_line = textY

        words = instruction.split(" ")
        x = 10
        line = ""
        for word in words:
            test_line = (line + " " + word) if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)
            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1] + 5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        y_line = y_line + textsize[1] + 10
        cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)
        return new_image

    def reset(self, episode: Optional[NavigationEpisode] = None):
        # Save per-episode debug video (topdown overlay)
        if episode is not None and len(self.topdown_map_list) != 0:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split(".")[0]
            save_dir = osp.join(self.save_root, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = osp.join(save_dir, f"{episode.episode_id}.mp4")
            imageio.mimsave(output_video_path, self.topdown_map_list, fps=10)
            print(f"[INFO] Saved debug video: {output_video_path}")

        self.rgb_list = []
        self.topdown_map_list = []
        self.pending_action_list = []
        self.step_idx = 0
        self.infer_idx = 0

        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def act(self, observations, info, instruction, episode_id):
        """
        Returns: base velocity [vx, vy, wz] for TrackEnv action_args["agent_1_base_vel"].
        """

        self.step_idx += 1

        rgb = observations["agent_1_articulated_agent_jaw_rgb"][:, :, :3]
        self.rgb_list.append(rgb)

        output_im = None
        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_following"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        # 1) If we have pending tokens: execute without inference
        if len(self.pending_action_list) != 0:
            tok = self.pending_action_list.pop(0)
            base_vel = self._token_to_base_vel(tok)

            if self.require_map and output_im is not None:
                img = self.addtext(output_im, instruction, f"Pending action: {tok}")
                self.topdown_map_list.append(img)

            return base_vel

        # 2) Need inference: copy frames BEFORE predict_inference clears rgb_list
        prompt = self.promt_template.format(instruction)
        frames_for_log = list(self.rgb_list)

        raw_actions = self.predict_inference(prompt)
        tokens4 = self._normalize_four_tokens(raw_actions)
        actions_str = " ".join(tokens4)

        # Cache next 3 tokens
        self.pending_action_list = tokens4[1:]

        if self.require_map and output_im is not None:
            img = self.addtext(output_im, instruction, actions_str)
            self.topdown_map_list.append(img)

        # 3) Log one UniNaVid-style sample per inference
        if self.require_data and self.logger is not None and self.current_episode is not None:
            self.infer_idx += 1
            self.logger.log_inference(
                frames=frames_for_log,
                prompt=prompt,
                actions_str=actions_str,
                episode=self.current_episode,
                infer_idx=self.infer_idx,
                extra={
                    "episode_id": getattr(self.current_episode, "episode_id", None),
                    "scene_id": getattr(self.current_episode, "scene_id", None),
                    "step_idx": self.step_idx,
                    "human_following": float(info.get("human_following", -1.0)),
                    "human_collision": float(info.get("human_collision", -1.0)),
                },
            )

        # Execute first token now
        return self._token_to_base_vel(tokens4[0])
