import os
import os.path as osp
import json
import gc
import re
from collections import deque
from typing import Optional, Union, Dict, Any, List, Tuple

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
# Track 数据采集 Logger（写 JSONL；clip 视频由 Agent 导出）
# =========================
class TrackDatasetLogger:
    """
    追加写入 sampled_500 风格的 JSONL（每条记录对应一个 clip mp4 文件）：
      - id:    NAV_ID_TRACK_<scene>_<episode>_<infer_idx>
      - video: nav_videos/<id>.mp4
      - conversations:
          [{"from":"human","value":prompt},
           {"from":"gpt","value":"forward forward left stop"}]
    """

    def __init__(self, root_dir: str, jsonl_name: str = "track_dataset.jsonl") -> None:
        self.root_dir = root_dir
        self.jsonl_path = osp.join(root_dir, jsonl_name)
        os.makedirs(self.root_dir, exist_ok=True)
        self.sample_count = 0

    def append_record(self, sample_id: str, video_rel: str, prompt: str, actions_str: str) -> None:
        record = {
            "id": sample_id,
            "video": video_rel.replace("\\", "/"),
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": actions_str},
            ],
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.sample_count += 1




# =========================
# Episode 级数据采集 Logger（写 JSONL；每条记录对应一个 *episode* 全视频）
# =========================
class EpisodeDatasetLogger:
    """
    追加写入 episode 级 JSONL（每条记录对应一个 episode 完整 mp4）：

      - id:    NAV_ID_TRACK_<scene>_<episode>
      - video: raw_videos/<id>.mp4
      - conversations:
          [{"from":"human","value":prompt},
           {"from":"gpt","value":"<action_1> <action_2> ... <action_T>"}]

    说明：
      - gpt 字段写“实际执行的动作序列”（每个环境 step 一个 token）。
      - 这是最贴合“每个 episode 一条标注”的需求；若你更希望存成结构化数组，也可以改成 actions: [...]
    """

    def __init__(self, root_dir: str, jsonl_name: str = "track_episodes.jsonl") -> None:
        self.root_dir = root_dir
        self.jsonl_path = osp.join(root_dir, jsonl_name)
        os.makedirs(self.root_dir, exist_ok=True)
        self.sample_count = 0

    def append_episode(self, sample_id: str, video_rel: str, prompt: str, actions_str: str, meta: Optional[Dict[str, Any]] = None) -> None:
        record = {
            "id": sample_id,
            "video": video_rel.replace("\\", "/"),
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": actions_str},
            ],
        }
        if meta:
            record["meta"] = meta
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
    # ====== 推理与显存相关 ======
    max_new_tokens: int = 16,
    do_sample: bool = False,
    empty_cache_every: int = 10,
    use_online_cache: bool = True,
    encode_only_new_frames: bool = True,
    cache_reset_every_infer: int = 50,
    cache_reset_reserved_ratio: float = 0.90,
    # ====== 数据采集相关 ======
    history_len: int = 20,          # clip 的历史帧长度（对齐你要的“20 帧历史窗口”）
    action_horizon: int = 4,        # next-4 actions
    save_episode_video: bool = True,
    episode_video_fps: int = 1,
    save_clip_video: bool = False,
    clip_video_fps: int = 1,
    write_clip_jsonl: bool = False,
    write_episode_jsonl: bool = True,
    episode_jsonl_name: str = "track_episodes.jsonl",
    clip_stride_infer: int = 1,     # 每 N 次推理保存一个 clip（默认每次都存）
    save_debug_video: bool = False,
    restart_env_every_episodes: int = 50,
) -> None:
    """
    评测 / 数据采集入口。

    你关心的三点在此都通过参数可控：
    1) 显存爆炸：use_online_cache + encode_only_new_frames + cache_reset_* + empty_cache_every
    2) 推理慢：action_horizon=4 => 每 4 步只推理 1 次；max_new_tokens 限制；encode_only_new_frames 减少视觉编码
    3) 标注格式：write_clip_jsonl=True，会生成 sampled_500 风格 JSONL（每条对应一个 clip mp4）
    """
    agent = UniNaVid_Agent(
        model_path=model_path,
        save_root=save_path,
        exp_save="video",
        cuda_device=cuda_device,
        device_map=device_map,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        empty_cache_every=empty_cache_every,
        use_online_cache=use_online_cache,
        encode_only_new_frames=encode_only_new_frames,
        cache_reset_every_infer=cache_reset_every_infer,
        cache_reset_reserved_ratio=cache_reset_reserved_ratio,
        history_len=history_len,
        action_horizon=action_horizon,
        save_episode_video=save_episode_video,
        episode_video_fps=episode_video_fps,
        save_clip_video=save_clip_video,
        clip_video_fps=clip_video_fps,
        write_clip_jsonl=write_clip_jsonl,
        write_episode_jsonl=write_episode_jsonl,
        episode_jsonl_name=episode_jsonl_name,
        clip_stride_infer=clip_stride_infer,
        save_debug_video=save_debug_video,
    )

    
    # ========= Habitat 环境（可能存在 renderer 侧显存缓慢增长/泄漏）=========
    # 经验上：长时间跑很多 episode 时，habitat-sim 的 GL/EGL 显存可能持续上升。
    # 解决策略：每隔 restart_env_every_episodes 个 episode，重建一次 TrackEnv，以释放渲染上下文占用。
    episodes_all = list(getattr(dataset_split, "episodes", []))
    total_episodes = len(episodes_all)

    def _run_chunk(chunk_episodes):
        # shallow copy dataset object, replace episodes
        import copy as _copy
        ds = _copy.copy(dataset_split)
        ds.episodes = list(chunk_episodes)

        with habitat.TrackEnv(config=config, dataset=ds) as env:
            sim = env.sim
            agent.reset()

            num_episodes = len(env.episodes)
            for _ in trange(num_episodes):
                obs = env.reset()  # 使用 env.reset / env.step 返回的 obs，避免额外调用 sim.get_sensor_observations()

                instruction = env.current_episode.info.get("instruction", "")
                agent.start_episode(env.current_episode)

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
                    # 使用上一步 env.step 返回的 obs 作为当前观测
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
                    obs = env.step(action_dict)  # 关键：用 env.step 返回新 obs；避免额外 render 调用
                    info = env.get_metrics()

                    if info.get("human_following", 0.0) == 1.0:
                        followed_step += 1
                        too_far_count = 0

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

                    if info.get("human_collision", 0.0) == 1.0:
                        status = "Collision"
                        finished = False
                        break

                # episode end
                info = env.get_metrics()
                agent.reset(env.current_episode)  # finalize：关闭 writer，落盘视频/标注（episode JSONL）

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

    if restart_env_every_episodes is None or int(restart_env_every_episodes) <= 0:
        _run_chunk(episodes_all)
    else:
        k = int(restart_env_every_episodes)
        for st in range(0, total_episodes, k):
            ed = min(total_episodes, st + k)
            _run_chunk(episodes_all[st:ed])
            # 强制回收 python 侧对象，帮助 habitat-sim 释放资源（GL 上下文已在 with exit 时关闭）
            gc.collect()
# =========================
# UniNaVid Agent
# =========================
class UniNaVid_Agent(Agent):
    _ALLOWED_ACTIONS = ("forward", "left", "right", "back", "stop")

    def __init__(
        self,
        model_path: str,
        # --- 兼容参数名：---
        save_root: Optional[str] = None,
        save_path: Optional[str] = None,
        result_path: Optional[str] = None,
        # -------------------
        exp_save: str = "video",
        cuda_device: Optional[int] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        load_4bit: Optional[bool] = None,
        load_8bit: Optional[bool] = None,
        # ====== 推理相关 ======
        max_new_tokens: int = 16,
        do_sample: bool = False,
        empty_cache_every: int = 10,
        use_online_cache: bool = True,
        encode_only_new_frames: bool = True,
        cache_reset_every_infer: int = 50,
        cache_reset_reserved_ratio: float = 0.90,
        # ====== 数据采集/格式 ======
        history_len: int = 20,
        action_horizon: int = 4,
        save_episode_video: bool = True,
        episode_video_fps: int = 1,
        save_clip_video: bool = False,
        clip_video_fps: int = 1,
        write_clip_jsonl: bool = False,
        write_episode_jsonl: bool = True,
        episode_jsonl_name: str = "track_episodes.jsonl",
        clip_stride_infer: int = 1,
        save_debug_video: bool = False,
    ) -> None:
        print("Initialize UniNaVid_Agent")

        # ---- 统一输出目录 ----
        if save_root is None:
            save_root = save_path
        if save_root is None:
            save_root = result_path
        if save_root is None:
            raise ValueError("You must provide one of: save_root / save_path / result_path")

        self.result_path = save_root
        os.makedirs(self.result_path, exist_ok=True)

        # ---- 设备选择 ----
        self.cuda_device = cuda_device
        self.device = self._resolve_device(cuda_device)

        # ---- Dual-GPU safety: force the whole HF model onto the requested cuda device ----
        # When multiple GPUs are visible, HF/accelerate may default to cuda:0 or shard with device_map="auto",
        # while our inputs are moved to self.device. That mismatch triggers:
        #   RuntimeError: Expected all tensors to be on the same device, but found cuda:1 and cuda:0
        # To avoid this, if user did not specify device_map, we pin the entire model to cuda:{cuda_device}.
        effective_device_map = device_map
        if (
            effective_device_map is None
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and cuda_device is not None
        ):
            effective_device_map = {"": f"cuda:{int(cuda_device)}"}

        # This will be updated after model loading (best-effort).
        self.model_device = self.device


        # ---- 推理/显存配置 ----
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.empty_cache_every = int(empty_cache_every) if empty_cache_every is not None else 0
        self.use_online_cache = bool(use_online_cache)
        self.encode_only_new_frames = bool(encode_only_new_frames)
        self.cache_reset_every_infer = int(cache_reset_every_infer) if cache_reset_every_infer is not None else 0
        self.cache_reset_reserved_ratio = float(cache_reset_reserved_ratio)

        # ---- 数据采集配置（对齐官方 clip：history_len + next-4 actions） ----
        self.history_len = int(history_len)
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be > 0")
        self.clip_stride_infer = max(1, int(clip_stride_infer))

        self.save_episode_video = bool(save_episode_video)
        self.episode_video_fps = int(episode_video_fps)
        self.save_clip_video = bool(save_clip_video)
        self.clip_video_fps = int(clip_video_fps)
        self.write_clip_jsonl = bool(write_clip_jsonl)
        self.write_episode_jsonl = bool(write_episode_jsonl)
        self.episode_jsonl_name = str(episode_jsonl_name)
        self.save_debug_video = bool(save_debug_video)

        # ---- dataset 输出目录（统一） ----
        self.dataset_root = osp.join(self.result_path, "uninavid_track_dataset")
        self.clip_video_dir = osp.join(self.dataset_root, "nav_videos")      # 对齐 sampled_500
        self.raw_episode_video_dir = osp.join(self.dataset_root, "raw_videos")  # 额外存整条 episode
        os.makedirs(self.clip_video_dir, exist_ok=True)
        os.makedirs(self.raw_episode_video_dir, exist_ok=True)

        # clip JSONL logger
        self.clip_logger = TrackDatasetLogger(root_dir=self.dataset_root) if self.write_clip_jsonl else None
        self.episode_logger = EpisodeDatasetLogger(root_dir=self.dataset_root, jsonl_name=episode_jsonl_name) if self.write_episode_jsonl else None

        # ---- 加载模型 ----
        self.conv_mode = "vicuna_v1"
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = self._safe_load_model(
            model_path=model_path,
            model_name=self.model_name,
            device_map=effective_device_map,
            load_4bit=load_4bit,
            load_8bit=load_8bit,
        )

        # 模型 device
        try:
            if effective_device_map is None and hasattr(self.model, "to"):
                self.model.to(self.model_device)
        except Exception as e:
            print(f"[WARN] model.to(device) skipped: {e}")

        try:
            self.model.eval()
        except Exception:
            pass

        # Best-effort: infer the actual device of the model parameters (in case HF moved it).
        try:
            self.model_device = next(self.model.parameters()).device
        except Exception:
            self.model_device = self.device

        # TF32 加速（不改变数值语义太多，通常可接受）
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        print(f"[INFO] UniNaVid device = {self.device}, model_device = {self.model_device}, cuda_device = {self.cuda_device}")

        # Prompt 模板（尽量保持官方 wording）
        self.promt_template = (
            "Imagine you are a robot programmed for navigation tasks. "
            "You have been given a video of historical observations and an image of the current observation <image>. "
            "Your assigned task is: '{}'. "
            "Analyze this series of images to determine your next four actions. "
            "The predicted action should be one of the following: forward, left, right, back, or stop."
        )

        # ---- prompt special tokens（缓存，避免每次重新 tokenize） ----
        self._init_special_tokens()

        # ---- episode 状态 ----
        self.current_episode: Optional[NavigationEpisode] = None
        self.step_idx = 0
        self.infer_idx = 0

        # 推理历史窗口（CPU 滑动缓冲区）
        self.rgb_history: deque = deque(maxlen=self.history_len)

        # 推理动作缓冲（每次推理输出 action_horizon 个；后续 action_horizon-1 步不再推理）
        self.pending_actions: List[str] = []

        # online cache 辅助状态
        self._frames_since_last_infer = 0
        self._cache_warmed = False

        # debug 画面（可选）
        self.require_map = ("video" in exp_save) and self.save_debug_video
        self.topdown_map_list: List[np.ndarray] = []

        # episode 视频 writer（整条轨迹）
        self._episode_vw: Optional[cv2.VideoWriter] = None
        self._episode_video_abs: Optional[str] = None
        self._episode_video_rel: Optional[str] = None
        self._episode_prompt: Optional[str] = None

        self.reset()

    # ------------------ special tokens ------------------
    def _init_special_tokens(self) -> None:
        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IMAGE_SEPARATOR = "<image_sep>"

        def _tok(s: str) -> torch.Tensor:
            # tokenizer 输出形如 [BOS, ...]，这里按原实现剔除 BOS
            return self.tokenizer(s, return_tensors="pt").input_ids[0][1:].to(self.model_device)

        self._st_image_start = _tok(IMAGE_START_TOKEN)
        self._st_image_end = _tok(IMAGE_END_TOKEN)
        self._st_video_start = _tok(VIDEO_START_SPECIAL_TOKEN)
        self._st_video_end = _tok(VIDEO_END_SPECIAL_TOKEN)
        self._st_nav = _tok(NAVIGATION_SPECIAL_TOKEN)
        self._st_sep = _tok(IMAGE_SEPARATOR)

    # ------------------ utilities ------------------
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
            return load_pretrained_model(model_path, None, model_name)

    @staticmethod
    def _scene_key(episode: NavigationEpisode) -> str:
        return osp.splitext(osp.basename(episode.scene_id))[0].split(".")[0]

    @staticmethod
    def _action_to_velocity(action: str) -> List[float]:
        if action == "stop":
            return [0.0, 0.0, 0.0]
        if action == "forward":
            return [0.5, 0.0, 0.0]
        if action == "left":
            return [0.0, 0.0, 1.0]
        if action == "right":
            return [0.0, 0.0, -1.0]
        if action == "back":
            return [-0.67, 0.0, 0.0]
        # 兜底
        return [0.0, 0.0, 0.0]

    @staticmethod
    def _ensure_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
        rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        return rgb
    @staticmethod
    def _pick_rgb_key(obs: Dict[str, Any]) -> Optional[str]:
        """Pick an RGB observation key robustly.

        Different episodes/configs may expose different sensor names, and sometimes
        observations are nested under agent_0/agent_1.
        """
        if not isinstance(obs, dict):
            return None

        preferred = [
            # Original expected key (jaw camera)
            "agent_1_articulated_agent_jaw_rgb",
            "agent_0_articulated_agent_jaw_rgb",
            "articulated_agent_jaw_rgb",
            # Common alternatives
            "agent_1_rgb",
            "agent_0_rgb",
            "rgb",
            "head_rgb",
            "agent_1_head_rgb",
            "agent_0_head_rgb",
            "color_sensor",
            "rgb_sensor",
        ]
        for k in preferred:
            if k in obs:
                return k

        # Heuristic fallback
        for k in obs.keys():
            lk = str(k).lower()
            if lk.endswith("_rgb") or (("rgb" in lk) and ("depth" not in lk) and ("semantic" not in lk)):
                return k
        return None

    def _get_rgb_frame(self, observations: Dict[str, Any]) -> np.ndarray:
        """Extract an RGB frame from observations without crashing.

        If no RGB key is found, falls back to last frame (or a black frame).
        """
        obs = observations

        # Handle nested per-agent dicts (common in Habitat multi-agent)
        if isinstance(obs, dict):
            for agent_key in ("agent_1", "agent_0"):
                if agent_key in obs and isinstance(obs[agent_key], dict):
                    candidate = obs[agent_key]
                    k = self._pick_rgb_key(candidate)
                    if k is not None:
                        return self._ensure_uint8_rgb(candidate[k])

        # Flat dict case
        if isinstance(obs, dict):
            k = self._pick_rgb_key(obs)
            if k is not None:
                return self._ensure_uint8_rgb(obs[k])

        # Fallback: reuse last frame or make a dummy frame
        last = getattr(self, "_last_rgb", None)
        if isinstance(last, np.ndarray):
            return last
        # default dummy
        return np.zeros((224, 224, 3), dtype=np.uint8)

    @classmethod
    def _parse_actions(cls, text: str, horizon: int) -> List[str]:
        """
        从模型输出里抽取动作 token（鲁棒处理：容忍“解释性文字”）。
        """
        if not text:
            return ["stop"] * horizon

        t = text.lower()
        # 允许逗号/换行/句号等分隔
        cand = re.findall(r"(forward|left|right|back|stop)", t)
        actions = [a for a in cand if a in cls._ALLOWED_ACTIONS]
        if len(actions) < horizon:
            actions += ["stop"] * (horizon - len(actions))
        return actions[:horizon]

    def _maybe_empty_cache(self, force: bool = False) -> None:
        if not torch.cuda.is_available():
            return
        if force or (self.empty_cache_every > 0 and (self.infer_idx % self.empty_cache_every == 0)):
            torch.cuda.empty_cache()

    def _cuda_reserved_ratio(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        try:
            dev_idx = self.device.index if hasattr(self.device, "index") else torch.cuda.current_device()
            total = torch.cuda.get_device_properties(dev_idx).total_memory
            reserved = torch.cuda.memory_reserved(dev_idx)
            if total <= 0:
                return 0.0
            return float(reserved) / float(total)
        except Exception:
            return 0.0

    def _init_online_cache(self) -> None:
        """
        初始化 Uni-NaVid online cache（如果模型实现支持）。
        """
        if not self.use_online_cache:
            self._cache_warmed = False
            return
        try:
            if hasattr(self.model, "get_model"):
                gm = self.model.get_model()
                if hasattr(gm, "initialize_online_inference_nav_feat_cache"):
                    gm.initialize_online_inference_nav_feat_cache()
                if hasattr(gm, "new_frames"):
                    gm.new_frames = 0
            self._cache_warmed = True
        except Exception as e:
            print(f"[WARN] online cache init skipped: {e}")
            self._cache_warmed = False

    # ------------------ video writing ------------------
    def _open_episode_video_writer(self, episode: NavigationEpisode, first_frame: np.ndarray) -> None:
        if not self.save_episode_video:
            return

        self._close_episode_video_writer()

        scene_key = self._scene_key(episode)
        ep_id = f"EP_{scene_key}_{episode.episode_id}"
        self._episode_video_rel = osp.join("raw_videos", f"{ep_id}.mp4").replace("\\", "/")
        self._episode_video_abs = osp.join(self.dataset_root, self._episode_video_rel)
        os.makedirs(osp.dirname(self._episode_video_abs), exist_ok=True)

        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(self._episode_video_abs, fourcc, float(self.episode_video_fps), (w, h))
        if not vw.isOpened():
            # fallback to imageio (slower but more compatible)
            self._episode_vw = None
            return
        self._episode_vw = vw

    def _append_episode_frame(self, frame_rgb: np.ndarray) -> None:
        if not self.save_episode_video:
            return
        if self._episode_vw is None and self._episode_video_abs is None:
            # 尚未打开（等第一帧）
            return
        if self._episode_vw is None:
            # cv2 writer 打不开的话，直接不写（不影响采集），你也可以改成 imageio 增量写
            return
        # cv2 写 BGR
        self._episode_vw.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def _close_episode_video_writer(self) -> None:
        if self._episode_vw is not None:
            try:
                self._episode_vw.release()
            except Exception:
                pass
        self._episode_vw = None
        self._episode_video_abs = None
        # keep _episode_video_rel until reset() finalizes episode JSONL

    def _save_clip_video(self, frames_rgb: List[np.ndarray], out_abs: str, fps: int) -> None:
        if not self.save_clip_video:
            return
        if len(frames_rgb) == 0:
            return
        os.makedirs(osp.dirname(out_abs), exist_ok=True)

        # 优先 cv2：速度快
        h, w = frames_rgb[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_abs, fourcc, float(fps), (w, h))
        if vw.isOpened():
            for fr in frames_rgb:
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            vw.release()
            return

        # fallback：imageio（兼容性更好）
        try:
            writer = imageio.get_writer(out_abs, fps=fps)
            for fr in frames_rgb:
                writer.append_data(fr)
            writer.close()
        except Exception as e:
            print(f"[WARN] Failed to save clip video: {out_abs} ({e})")

    # ------------------ lifecycle ------------------
    def start_episode(self, episode: NavigationEpisode) -> None:
        self.current_episode = episode
        self.step_idx = 0
        self.infer_idx = 0
        self.rgb_history = deque(maxlen=self.history_len)
        self.pending_actions = []
        self._frames_since_last_infer = 0
        self._episode_prompt = None
        self._episode_actions = []
        self._episode_video_rel = None
        self.topdown_map_list = []

        # 每个 episode 开始时重置 online cache（强约束：避免跨 episode 堆积导致显存爆炸）
        self._init_online_cache()

        # episode writer 延迟到拿到第一帧再打开（需要知道分辨率）
        self._close_episode_video_writer()

    def reset(self, episode: NavigationEpisode = None):
        # episode 结束：关闭 writer；必要时保存 debug
        if episode is not None:
            self._close_episode_video_writer()

            if self.require_map and len(self.topdown_map_list) != 0:
                scene_key = self._scene_key(episode)
                save_dir = osp.join(self.result_path, scene_key)
                os.makedirs(save_dir, exist_ok=True)
                output_video_path = osp.join(save_dir, f"{episode.episode_id}_debug.mp4")
                try:
                    imageio.mimsave(output_video_path, self.topdown_map_list, fps=1)
                    print(f"[INFO] Saved debug episode video: {output_video_path}")
                except Exception as e:
                    print(f"[WARN] Failed to save debug episode video: {e}")

            # ====== Episode 级 JSONL：每个 episode 一条标注（你期望的输出）======
            if self.write_episode_jsonl and self.episode_logger is not None and episode is not None:
                try:
                    scene_key = self._scene_key(episode)
                    sample_id = f"NAV_ID_TRACK_{scene_key}_{episode.episode_id}"
                    # prompt：尽量复用 episode 第一次推理构造的 human prompt；若 episode 极短未触发推理，则现场构造
                    if self._episode_prompt is None:
                        self._episode_prompt = self.promt_template.format(getattr(episode, "info", {}).get("instruction", ""))
                    actions_str = " ".join(getattr(self, "_episode_actions", []))
                    video_rel = self._episode_video_rel if self._episode_video_rel is not None else ""
                    meta = {"total_steps": len(getattr(self, "_episode_actions", []))}
                    self.episode_logger.append_episode(
                        sample_id=sample_id,
                        video_rel=video_rel,
                        prompt=self._episode_prompt if self._episode_prompt is not None else "",
                        actions_str=actions_str,
                        meta=meta,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to write episode JSONL: {e}")

        # 清理 CPU 状态
        self.current_episode = None
        self.rgb_history = deque(maxlen=self.history_len)
        self.pending_actions = []
        self.step_idx = 0
        self.infer_idx = 0
        self._frames_since_last_infer = 0
        self._episode_prompt = None
        self._episode_actions = []
        self._episode_video_rel = None
        self.topdown_map_list = []

        # 重置 online cache（避免显存长期累积）
        self._init_online_cache()

        gc.collect()
        self._maybe_empty_cache(force=True)

    # ------------------ model IO ------------------
    def _process_images_to_video_tensor(self, rgb_list: List[np.ndarray], new_frames: int) -> List[torch.Tensor]:
        """
        将 RGB list -> model 需要的 video tensor，并设置 new_frames（如果模型支持）。
        注意：我们始终只在 inference 时把 tensor 放到 GPU；history 缓冲区始终在 CPU。
        """
        batch_image = np.asarray(rgb_list, dtype=np.uint8)  # (T,H,W,3)

        # 将“本次新增帧数”告诉模型（关键：防止 online cache 误认为每次都是全量新帧，从而 cache 爆炸）
        try:
            if hasattr(self.model, "get_model") and hasattr(self.model.get_model(), "new_frames"):
                self.model.get_model().new_frames = int(new_frames)
        except Exception:
            pass

        video = self.image_processor.preprocess(batch_image, return_tensors="pt")["pixel_values"]
        # 半精度 + 非阻塞搬运
        video = video.to(dtype=torch.float16, device=self.model_device, non_blocking=True)
        return [video]

    def _build_input_ids(self, instruction: str) -> Tuple[torch.Tensor, str, str, str]:
        """
        构造 input_ids，并返回：
          - input_ids
          - navigation_qs（human prompt）
          - question（用于 model.update_prompt 的纯文本）
          - stop_str（stopping criteria）
        """
        navigation_qs = self.promt_template.format(instruction)
        question = navigation_qs.replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
        qs = navigation_qs

        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs.replace("<image>", "")
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "")

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(self.model_device)
        indices_to_replace = torch.where(token_prompt == -200)[0]

        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(self._st_video_start)
            new_list.append(self._st_sep)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(self._st_video_end)
            new_list.append(self._st_image_start)
            new_list.append(self._st_image_end)
            new_list.append(self._st_nav)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)

        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        return input_ids, navigation_qs, question, stop_str

    def _predict_actions(self, instruction: str) -> Tuple[str, str, List[str]]:
        """
        返回：
          - navigation_qs（human prompt）
          - outputs_text（raw）
          - actions（parsed next-4）
        """
        # 触发 cache reset（显存接近阈值 / 周期性）
        if self.use_online_cache:
            if (self.cache_reset_every_infer > 0 and self.infer_idx > 0 and (self.infer_idx % self.cache_reset_every_infer == 0)):
                self._init_online_cache()
                # cache 重置后，本次需要重新编码“全量 history”
                self._frames_since_last_infer = len(self.rgb_history)

            if self.cache_reset_reserved_ratio > 0:
                ratio = self._cuda_reserved_ratio()
                if ratio >= self.cache_reset_reserved_ratio:
                    print(f"[WARN] CUDA reserved ratio {ratio:.2f} >= {self.cache_reset_reserved_ratio:.2f}, reset online cache.")
                    self._init_online_cache()
                    self._frames_since_last_infer = len(self.rgb_history)
                    self._maybe_empty_cache(force=True)

        input_ids, navigation_qs, question, stop_str = self._build_input_ids(instruction)
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        # 选择喂给模型的帧（速度关键）
        history_frames = list(self.rgb_history)
        if self.use_online_cache and self.encode_only_new_frames and self._cache_warmed:
            new_frames = max(1, int(self._frames_since_last_infer))
            frames_for_model = history_frames[-new_frames:]
        else:
            # 保守模式：重编码整个 history（更稳，但更慢）
            new_frames = len(history_frames)
            frames_for_model = history_frames

        imgs = self._process_images_to_video_tensor(frames_for_model, new_frames=new_frames)

        self.infer_idx += 1

        try:
            if hasattr(self.model, "update_prompt"):
                self.model.update_prompt([[question]])
        except Exception:
            pass

        # 生成长度严格限制（速度/显存关键）
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

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()

        actions = self._parse_actions(outputs, horizon=self.action_horizon)

        # 清理：避免引用残留导致显存/内存增长
        try:
            del imgs, input_ids, output_ids
        except Exception:
            pass
        gc.collect()
        self._maybe_empty_cache(force=False)

        # 推理完成：将“新帧计数”清零（下一次推理只编码新增帧）
        self._frames_since_last_infer = 0

        return navigation_qs, outputs, actions

    # ------------------ debug rendering ------------------
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

    # ------------------ main step ------------------
    def act(self, observations, info, instruction: str, episode_id):
        """
        关键行为：
        - 每步把 RGB 放入 CPU deque（history_len=20）
        - 每 action_horizon 步才推理 1 次（pending_actions 复用 next-4）
        - 每次推理保存一个 clip video（长度 history_len，fps=1），并写 sampled_500 风格 JSONL
        """
        if self.current_episode is None:
            # 容错：如果外部没调用 start_episode
            try:
                self.start_episode(getattr(self, "current_episode", None))
            except Exception:
                pass

        self.step_idx += 1

        rgb = self._get_rgb_frame(observations)
        self._last_rgb = rgb

        # 第一次帧到来时打开 episode writer（如果启用）
        if self.save_episode_video and self._episode_video_abs is None:
            if self.current_episode is not None:
                self._open_episode_video_writer(self.current_episode, rgb)

        # 1) CPU 历史窗口
        self.rgb_history.append(rgb)
        self._frames_since_last_infer += 1

        # 2) 整条 episode 视频（纯视频）
        self._append_episode_frame(rgb)

        # 3) 是否需要推理（next-4 复用：显著加速）
        if len(self.pending_actions) > 0:
            action = self.pending_actions.pop(0)
            try:
                self._episode_actions.append(action)
            except Exception:
                pass
            return self._action_to_velocity(action)

        # 推理（每 action_horizon 步一次）
        navigation_qs, raw_out, actions = self._predict_actions(instruction)

        # 记录一次 prompt（可用于 debug；标注写入时使用 navigation_qs）
        if self._episode_prompt is None:
            self._episode_prompt = navigation_qs

        # 保存 clip + 写 JSONL（旧逻辑：每次推理一个 clip；默认关闭）
        if self.save_clip_video or self.write_clip_jsonl:
            if self.current_episode is not None:
                scene_key = self._scene_key(self.current_episode)
                sample_id = f"NAV_ID_TRACK_{scene_key}_{self.current_episode.episode_id}_{self.infer_idx:06d}"
    
                # stride：不是每次推理都保存，减小 IO 压力
                if (self.infer_idx % self.clip_stride_infer) == 0:
                    video_rel = osp.join("nav_videos", f"{sample_id}.mp4").replace("\\", "/")
                    video_abs = osp.join(self.dataset_root, video_rel)
    
                    # clip = 最近 history_len 帧（不足则用现有）
                    clip_frames = list(self.rgb_history)
    
                    self._save_clip_video(clip_frames, out_abs=video_abs, fps=self.clip_video_fps)
    
                    if self.clip_logger is not None:
                        # 重要：标注用 next-4 actions（对齐 sampled_500）
                        actions_str = " ".join(actions)
                        self.clip_logger.append_record(
                            sample_id=sample_id,
                            video_rel=video_rel,
                            prompt=navigation_qs,
                            actions_str=actions_str,
                        )
    
        # debug map（可选）
        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_following"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)
            img = self._add_text(output_im, instruction, " ".join(actions))
            self.topdown_map_list.append(img)

        # 4) 把剩余动作塞进 pending（下一步开始不再推理）
        self.pending_actions = actions[1:]
        first_action = actions[0]
        try:
            self._episode_actions.append(first_action)
        except Exception:
            pass

        return self._action_to_velocity(first_action)