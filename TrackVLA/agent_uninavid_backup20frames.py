import os
import os.path as osp
import json
import gc
from collections import deque
from typing import Optional, Union, Dict, Any, List

import numpy as np
import torch
import cv2
import imageio
from tqdm import trange

import habitat
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.utils.visualizations import maps
from habitat_sim.gfx import LightInfo, LightPositionModel

from uninavid.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from uninavid.conversation import conv_templates, SeparatorStyle


# =========================
# Track 数据采集 Logger
# =========================
class TrackDatasetLogger:
    """
    将 Uni-NaVid 在 Track 任务中的推理调用整理成 sampled_500 的 JSONL 形式：
      - id:  TRACK_<scene>_<episode>_<infer_idx>
      - video: nav_videos/<id>.mp4
      - conversations:
          [ {"from": "human", "value": prompt},
            {"from": "gpt",   "value": actions_str} ]
    """

    def __init__(
        self,
        root_dir: str,
        jsonl_name: str = "track_dataset.jsonl",
        fps: int = 1,
        max_history_frames: int = 32,
        log_every_n_infer: int = 1,
    ) -> None:
        self.root_dir = root_dir
        self.jsonl_path = osp.join(root_dir, jsonl_name)
        self.fps = int(fps)
        self.max_history_frames = int(max_history_frames) if max_history_frames is not None else None
        self.log_every_n_infer = int(log_every_n_infer)

        os.makedirs(self.root_dir, exist_ok=True)
        self.sample_count = 0

    def log_infer(
        self,
        frames: List[np.ndarray],
        prompt: str,
        actions_str: str,
        episode: NavigationEpisode,
        infer_idx: int,
    ) -> None:
        if self.log_every_n_infer > 1 and (infer_idx % self.log_every_n_infer != 0):
            return
        if not frames:
            return

        if self.max_history_frames is not None and len(frames) > self.max_history_frames:
            frames = frames[-self.max_history_frames :]

        scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split(".")[0]
        sample_id = f"TRACK_{scene_key}_{episode.episode_id}_{infer_idx:04d}"

        video_rel = osp.join("nav_videos", f"{sample_id}.mp4").replace("\\", "/")
        video_abs = osp.join(self.root_dir, video_rel)
        os.makedirs(osp.dirname(video_abs), exist_ok=True)

        # frames: list of HxWx3 uint8
        imageio.mimsave(video_abs, frames, fps=self.fps)

        record = {
            "id": sample_id,
            "video": video_rel,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": actions_str},
            ],
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.sample_count += 1


# =========================
# 外部入口：evaluate_agent
# =========================
def evaluate_agent(
    config,
    model_path: str,
    dataset_split,
    save_path: str,
    cuda_device: Optional[int] = None,
    device_map: Optional[Union[str, Dict[str, Any]]] = None,
    load_4bit: Optional[bool] = None,
    load_8bit: Optional[bool] = None,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    empty_cache_every: int = 10,
) -> None:
    """
    评测 / 数据采集入口。关键点：
    - save_path 是输出根目录（会在其下创建 scene 子目录及 uninavid_track_dataset）
    - cuda_device 是“可见 GPU 列表”里的 index（建议配合 CUDA_VISIBLE_DEVICES 使用更可控）
    """
    agent = UniNaVid_Agent(
        model_path=model_path,
        save_root=save_path,         # 统一用 save_root；同时兼容 save_path/result_path
        exp_save="video",
        cuda_device=cuda_device,
        device_map=device_map,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        empty_cache_every=empty_cache_every,
    )

    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        sim = env.sim
        agent.reset()

        num_episodes = len(env.episodes)
        for _ in trange(num_episodes):
            env.reset()

            # 每个 episode 都更新 instruction
            instruction = env.current_episode.info.get("instruction", "")

            # 告诉 agent 当前 episode（用于 sample id）
            agent.start_episode(env.current_episode)

            # 设置光照
            light_setup = [
                LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
            ]
            sim.set_light_setup(light_setup)

            record_infos = []
            finished = False
            status = "Normal"

            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            iter_step = 0
            followed_step = 0
            too_far_count = 0

            info = env.get_metrics()

            while not env.episode_over:
                obs = sim.get_sensor_observations()
                action = agent.act(obs, info, instruction, env.current_episode.episode_id)

                action_dict = {
                    "action": (
                        "agent_0_humanoid_navigate_action",
                        "agent_1_base_velocity",
                        "agent_2_oracle_nav_randcoord_action_obstacle",
                        "agent_3_oracle_nav_randcoord_action_obstacle",
                        "agent_4_oracle_nav_randcoord_action_obstacle",
                        "agent_5_oracle_nav_randcoord_action_obstacle",
                    ),
                    "action_args": {"agent_1_base_vel": action},
                }

                iter_step += 1
                env.step(action_dict)
                info = env.get_metrics()

                # metrics
                if info.get("human_following", 0.0) == 1.0:
                    followed_step += 1
                    too_far_count = 0
                else:
                    pass

                # too far check
                if np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        status = "Lost"
                        finished = False
                        break

                record_infos.append(
                    {
                        "step": iter_step,
                        "dis_to_human": float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos)),
                        "facing": float(info.get("human_following", 0.0)),
                    }
                )

                # collision
                if info.get("human_collision", 0.0) == 1.0:
                    status = "Collision"
                    finished = False
                    break

            # episode end
            info = env.get_metrics()
            agent.reset(env.current_episode)  # 保存 episode video（如果启用）

            if env.episode_over:
                finished = True

            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split(".")[0]
            save_dir = osp.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)

            with open(osp.join(save_dir, f"{env.current_episode.episode_id}_info.json"), "w") as f:
                json.dump(record_infos, f, indent=2)

            result = {
                "finish": finished,
                "status": status,
                "success": bool(info.get("human_following", 0.0)),
                "following_rate": (followed_step / iter_step) if iter_step > 0 else 0.0,
                "following_step": followed_step,
                "total_step": iter_step,
                "collision": float(info.get("human_collision", 0.0)),
            }
            with open(osp.join(save_dir, f"{env.current_episode.episode_id}.json"), "w") as f:
                json.dump(result, f, indent=2)


# =========================
# UniNaVid Agent
# =========================
class UniNaVid_Agent(Agent):
    def __init__(
        self,
        model_path: str,
        # --- 兼容参数名（你之前的错误就在这里）：---
        save_root: Optional[str] = None,
        save_path: Optional[str] = None,
        result_path: Optional[str] = None,
        # ---------------------------------------
        exp_save: str = "video",
        cuda_device: Optional[int] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        load_4bit: Optional[bool] = None,
        load_8bit: Optional[bool] = None,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        empty_cache_every: int = 10,
    ) -> None:
        """
        关键点：
        - save_root/save_path/result_path 三者等价，任选其一传入
        - cuda_device：torch 的 device index（建议配合 CUDA_VISIBLE_DEVICES）
        - max_new_tokens：强烈建议小（<=64），防止 generate KV cache 爆显存
        """
        print("Initialize UniNaVid_Agent")

        # ---- 统一输出目录 ----
        if save_root is None:
            save_root = save_path
        if save_root is None:
            save_root = result_path
        if save_root is None:
            raise ValueError("You must provide one of: save_root / save_path / result_path")

        self.result_path = save_root
        self.require_map = ("video" in exp_save)
        self.require_data = ("video" in exp_save)

        os.makedirs(self.result_path, exist_ok=True)

        # ---- 设备选择 ----
        self.cuda_device = cuda_device
        self.device = self._resolve_device(cuda_device)
        self.empty_cache_every = int(empty_cache_every) if empty_cache_every is not None else 0

        # ---- 生成参数 ----
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)

        self.conv_mode = "vicuna_v1"

        # ---- 加载模型 ----
        self.model_name = get_model_name_from_path(model_path)

        # 尝试把量化/设备参数透传给 builder（若 builder 不支持则自动回退）
        self.tokenizer, self.model, self.image_processor, self.context_len = self._safe_load_model(
            model_path=model_path,
            model_name=self.model_name,
            device_map=device_map,
            load_4bit=load_4bit,
            load_8bit=load_8bit,
        )

        # 若没有 device_map 分配，则尽量把模型放到 self.device
        # 注：量化模型/accelerate device_map 下不一定支持 .to()
        try:
            if device_map is None and hasattr(self.model, "to"):
                self.model.to(self.device)
        except Exception as e:
            print(f"[WARN] model.to(device) skipped: {e}")

        print(f"[INFO] UniNaVid device = {self.device}, cuda_device = {self.cuda_device}")

        # Prompt 模板
        self.promt_template = (
            "Imagine you are a robot programmed for navigation tasks. "
            "You have been given a video of historical observations and an image of the current observation <image>. "
            "Your assigned task is: '{}'. "
            "Analyze this series of images to determine your next four actions. "
            "The predicted action should be one of the following: forward, left, right, back, or stop."
        )

        # episode 状态
        self.current_episode: Optional[NavigationEpisode] = None
        self.step_idx = 0
        self.infer_idx = 0

        # 缓存（全部在 CPU，避免无意占 GPU）
        self.history_len = 20  # sliding history window (frames)
        self.rgb_history = deque(maxlen=self.history_len)  # stores latest RGB frames on CPU
        self.topdown_map_list: List[np.ndarray] = []

        # 数据集 logger（采样格式）

        self.enable_logging = True
        if self.enable_logging:
            dataset_root = osp.join(self.result_path, "uninavid_track_dataset")
            self.logger = TrackDatasetLogger(
                root_dir=dataset_root,
                fps=1,                  # 想和官方风格更接近就用 1
                max_history_frames=self.history_len,  # 历史窗口（帧）
                log_every_n_infer=1,
            )
        else:
            self.logger = None

        self.reset()

    @staticmethod
    def _resolve_device(cuda_device: Optional[int]) -> torch.device:
        if torch.cuda.is_available():
            if cuda_device is not None:
                try:
                    torch.cuda.set_device(int(cuda_device))
                except Exception as e:
                    print(f"[WARN] torch.cuda.set_device({cuda_device}) failed: {e}")
                return torch.device(f"cuda:{int(cuda_device)}")
            return torch.device("cuda:0")
        return torch.device("cpu")

    @staticmethod
    def _safe_load_model(
        model_path: str,
        model_name: str,
        device_map: Optional[Union[str, Dict[str, Any]]],
        load_4bit: Optional[bool],
        load_8bit: Optional[bool],
    ):
        # 兼容 builder 是否支持这些 kwargs
        kwargs = {}
        if device_map is not None:
            kwargs["device_map"] = device_map
        if load_4bit is not None:
            kwargs["load_4bit"] = load_4bit
        if load_8bit is not None:
            kwargs["load_8bit"] = load_8bit

        try:
            return load_pretrained_model(model_path, None, model_name, **kwargs)
        except TypeError:
            # builder 老版本不接受量化/设备参数
            return load_pretrained_model(model_path, None, model_name)

    def start_episode(self, episode: NavigationEpisode) -> None:
        self.current_episode = episode
        self.step_idx = 0
        self.infer_idx = 0
        self.rgb_history = deque(maxlen=self.history_len)
        self.topdown_map_list = []

    def reset(self, episode: NavigationEpisode = None):
        # 保存上一 episode 的 debug video（如果有）
        if episode is not None and len(self.topdown_map_list) != 0:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split(".")[0]
            save_dir = osp.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = osp.join(save_dir, f"{episode.episode_id}.mp4")
            try:
                imageio.mimsave(output_video_path, self.topdown_map_list)
                print(f"[INFO] Saved episode video: {output_video_path}")
            except Exception as e:
                print(f"[WARN] Failed to save episode video: {e}")

        # 清理 CPU 缓存
        self.rgb_history = deque(maxlen=self.history_len)
        self.topdown_map_list = []
        self.step_idx = 0
        self.infer_idx = 0

        # 清理模型在线缓存（若实现支持）
        try:
            self.model.config.run_type = "eval"
        except Exception:
            pass

        try:
            if hasattr(self.model, "get_model"):
                gm = self.model.get_model()
                if hasattr(gm, "initialize_online_inference_nav_feat_cache"):
                    gm.initialize_online_inference_nav_feat_cache()
                if hasattr(gm, "new_frames"):
                    gm.new_frames = 0
        except Exception as e:
            print(f"[WARN] online cache init skipped: {e}")

        # 尽量释放显存碎片
        self._maybe_empty_cache(force=True)

    def _maybe_empty_cache(self, force: bool = False):
        if not torch.cuda.is_available():
            return
        if force or (self.empty_cache_every > 0 and (self.infer_idx % self.empty_cache_every == 0)):
            torch.cuda.empty_cache()

    def _process_images_to_video_tensor(self, rgb_list: List[np.ndarray]) -> List[torch.Tensor]:
        """
        将 CPU 的 rgb_list 转成模型需要的 video tensor。
        注意：这里只在推理瞬间把 tensor 上 GPU，推理完立刻释放。
        """
        batch_image = np.asarray(rgb_list, dtype=np.uint8)  # (T,H,W,3)
        # 让模型知道新帧数（若支持在线缓存）
        try:
            if hasattr(self.model, "get_model") and hasattr(self.model.get_model(), "new_frames"):
                self.model.get_model().new_frames = int(len(rgb_list))
        except Exception:
            pass

        video = self.image_processor.preprocess(batch_image, return_tensors="pt")["pixel_values"]
        # half + move to device
        video = video.to(dtype=torch.float16, device=self.device, non_blocking=True)
        return [video]

    def _predict_actions(self, instruction: str) -> str:
        """
        返回动作字符串：例如 "right right forward forward"
        """
        navigation_qs = self.promt_template.format(instruction)
        question = navigation_qs.replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
        qs = navigation_qs

        # 特殊 token
        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IMAGE_SEPARATOR = "<image_sep>"

        def _tok(s: str) -> torch.Tensor:
            return self.tokenizer(s, return_tensors="pt").input_ids[0][1:].to(self.device)

        image_start_special_token = _tok(IMAGE_START_TOKEN)
        image_end_special_token = _tok(IMAGE_END_TOKEN)
        video_start_special_token = _tok(VIDEO_START_SPECIAL_TOKEN)
        video_end_special_token = _tok(VIDEO_END_SPECIAL_TOKEN)
        navigation_special_token = _tok(NAVIGATION_SPECIAL_TOKEN)
        image_seperator = _tok(IMAGE_SEPARATOR)

        # 拼 prompt
        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs.replace("<image>", "")
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "")

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(self.device)
        indices_to_replace = torch.where(token_prompt == -200)[0]

        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        # stop 条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        # 取历史帧（CPU）→ video tensor（GPU）
        history_frames = list(self.rgb_history)
        imgs = self._process_images_to_video_tensor(history_frames)

        # 触发推理计数
        self.infer_idx += 1

        # 可选：让模型更新 prompt（若支持）
        try:
            if hasattr(self.model, "update_prompt"):
                self.model.update_prompt([[question]])
        except Exception:
            pass

        # generate（控制 max_new_tokens 防 OOM）
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=self.do_sample,
                temperature=0.2,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        # decode
        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()

        # 释放显存占用（重要）
        try:
            del imgs, input_ids, output_ids, token_prompt
        except Exception:
            pass
        gc.collect()
        self._maybe_empty_cache(force=False)

        return navigation_qs, outputs

    @staticmethod
    def _action_to_velocity(first_action: str) -> List[float]:
        if first_action == "stop":
            return [0.0, 0.0, 0.0]
        if first_action == "forward":
            return [0.5, 0.0, 0.0]
        if first_action == "left":
            return [0.0, 0.0, 1.0]
        if first_action == "right":
            return [0.0, 0.0, -1.0]
        if first_action == "back":
            return [-0.67, 0.0, 0.0]
        raise ValueError(f"Unknown action token: {first_action}")

    def _add_text(self, image: np.ndarray, instruction: str, navigation: str) -> np.ndarray:
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instruction, font, 0.5, 2)[0]
        y_line = h + (50 + textsize[1]) // 2

        words = instruction.split(" ")
        x = 10
        line = ""

        for word in words:
            test_line = line + " " + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)
            if test_line_size[0] > (w - x):
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

    def act(self, observations, info, instruction: str, episode_id):
        # step 计数（episode 内）
        self.step_idx += 1

        # 取 RGB（CPU）
        rgb = observations["agent_1_articulated_agent_jaw_rgb"][:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        # 记录历史帧（CPU）
        self.rgb_history.append(rgb)

        # 生成 debug 拼图（CPU）
        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_following"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)
        else:
            output_im = rgb

        # 当需要推理时：调用模型输出动作字符串
        res = self._predict_actions(instruction)
        if isinstance(res, (tuple, list)) and len(res) == 2:
            navigation_qs, navigation = res
        elif isinstance(res, (tuple, list)) and len(res) >= 2:
            navigation_qs, navigation = res[0], res[1]
        else:
            navigation_qs, navigation = '', res

        # 始终使用“滑动历史缓冲区”的最近 H 帧作为视频片段
        history_frames = list(self.rgb_history)
        print("[UniNaVid] Output actions:", navigation)

        if self.require_map:
            img = self._add_text(output_im, instruction, navigation)
            self.topdown_map_list.append(img)

        # 记录样本（以“每次推理”为单位，而不是每 step）
        if self.logger is not None and self.current_episode is not None:
            # 注意：这里用的是“推理前的历史帧”，但我们在 _predict_actions 内部已清空 rgb_history，
            # 所以 logger 应该使用推理前的历史。为此我们在推理前已经把 rgb_history 转成 tensor，
            # 这里直接用 topdown_map_list 或者把推理前历史在 _predict_actions 里缓存一份都可以。
            # 简洁起见：用 debug 的 output_im 序列会变“拼图风格”，不适合训练；
            # 因此这里改为：在推理前保存一份 rgb_history 的浅拷贝。
            # ——为了不引入额外复杂性，这里取 logger 的输入改为最近 1 帧也能跑通，
            # 但你想要 sampled_500 风格应保留历史帧。建议把下面 frames 改成你更偏好的缓存策略。
            pass

        # 为了保证训练采集数据有历史窗口：在推理前记录 shallow copy（不占 GPU）
        # 注意：此处我们只记录“刚刚这一步之前”的历史（即推理窗口），更符合 sampled_500 的逻辑。
        if self.logger is not None and self.current_episode is not None:
            # 由于 _predict_actions 清空了 history，这里用 topdown_map_list 不合适；
            # 退一步：用最近若干帧的 RGB（我们刚 append 了 rgb，且在 _predict_actions 清空前已转 tensor）。
            # 为了稳定，我们在清空前无法再取，所以这里采用策略：每次推理只保存当前帧作为视频（最小可用）。
            # 如果你希望严格“历史窗口视频”，我建议把 _predict_actions 改成返回 history_frames_copy。
            self.logger.log_infer(
                frames=history_frames,
                prompt=navigation_qs,
                actions_str=navigation,
                episode=self.current_episode,
                infer_idx=self.infer_idx,
            )

        # 动作解析：只执行第一个 token 的动作（保持你原来的控制逻辑）
        first = navigation.split(" ")[0].strip()
        return self._action_to_velocity(first)