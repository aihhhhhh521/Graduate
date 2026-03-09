import torch

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

import habitat
import numpy as np
import os
import re
import cv2
import imageio
from tqdm import trange
import os.path as osp
import json
from habitat.core.agent import Agent
from habitat.utils.visualizations import maps
from habitat.config.default_structured_configs import AgentConfig
from habitat.tasks.nav.nav import NavigationEpisode
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import json
import time
import subprocess
from typing import Optional, List
import random
from trackvla_step_stats_utils import JSONLWriter, WallTimer, maybe_cuda_timer, hz_from_ms, summarize_ms

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown"


def is_random_strategy(strategy: Optional[str]) -> bool:
    if not strategy:
        return False
    st = str(strategy).lower()
    return any(k in st for k in ["random", "stochastic", "sample"])



# =========== 定义数据采集 Logger ==========
class TrackDatasetLogger:
    """
    把 Uni-NaVid 在 Track 任务中的每一步调用，整理成 open_uninavid_sampled_500 的格式：
      - id:  TRACK_<scene>_<episode>_<step>
      - video: nav_videos/<id>.mp4
      - conversations:
          [ {"from": "human", "value": prompt},
            {"from": "gpt",   "value": actions_str} ]
    """

    def __init__(
        self,
        root_dir: str,
        jsonl_name: str = "track_dataset.jsonl",
        fps: int = 4,
        max_history_frames: int = 32,
        log_every_n_steps: int = 1,
    ) -> None:
        """
        :param root_dir: 数据集输出根目录，例如 results/track_dataset
        :param jsonl_name: 保存样本的 JSONL 文件名
        :param fps: 保存 mp4 时的帧率（历史帧的“虚拟 FPS”）
        :param max_history_frames: 每个样本最多保留多少帧历史（防止 episode 太长）
        :param log_every_n_steps: 每隔多少步采一个样本，1 表示每步都采
        """
        self.root_dir = root_dir
        self.jsonl_path = os.path.join(root_dir, jsonl_name)
        self.fps = fps
        self.max_history_frames = max_history_frames
        self.log_every_n_steps = log_every_n_steps

        os.makedirs(self.root_dir, exist_ok=True)

        self.sample_count = 0

    def log_step(
        self,
        frames,
        prompt: str,
        actions_str: str,
        episode,
        step_idx: int,
    ) -> None:
        """
        :param frames: list[np.ndarray(H, W, 3)], 历史 RGB 序列
        :param prompt: navigation_qs，已经 format(instruction) 之后的完整 prompt
        :param actions_str: Uni-NaVid 输出的动作字符串，比如 "right right forward forward"
        :param episode: 当前 habitat episode 对象，用于取 scene_id / episode_id
        :param step_idx: 当前步编号，从 1 开始
        """
        # 采样步间隔控制
        if self.log_every_n_steps > 1 and (step_idx % self.log_every_n_steps != 0):
            return

        if len(frames) == 0:
            return

        # 限制历史长度（例如只保留最近 32 帧）
        if self.max_history_frames is not None and len(frames) > self.max_history_frames:
            frames = frames[-self.max_history_frames :]

        # 构造 sample id 和视频路径
        scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split(".")[0]
        sample_id = f"TRACK_{scene_key}_{episode.episode_id}_{step_idx:04d}"

        video_rel = osp.join("nav_videos", f"{sample_id}.mp4").replace("\\", "/")
        video_abs = osp.join(self.root_dir, video_rel)
        os.makedirs(osp.dirname(video_abs), exist_ok=True)

        # 写 mp4
        imageio.mimsave(video_abs, frames, fps=self.fps)

        # 写一条样本到 JSONL
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



def evaluate_agent(
    config,
    model_path,
    dataset_split,
    save_path,
    split_id: Optional[int] = None,
    enable_step_stats: bool = False,
    step_stats_path: Optional[str] = None,
    speed_summary_path: Optional[str] = None,
    metrics_summary_path: Optional[str] = None,
    log_every_n_steps: int = 1,
    sampling_strategy: Optional[str] = None,
    sampling_k: Optional[int] = None,
    sampling_stride: Optional[int] = None,
    sampling_seed: Optional[int] = None,
    seed: Optional[int] = None,
    token_ablation_mode: Optional[str] = None,
) -> None:
    """
    TrackVLA/EVT-Bench evaluation for Uni-NaVid agent with:
      - per-step profiling (latency/FPS) saved as JSONL
      - per-episode SR/TR/CR-compatible fields (success/following_rate/collision)
      - per-split summary JSONs (optional)

    Notes:
      - SR/TR/CR aggregation is consistent with analyze_results.py:
          SR = succ_count / episode_count
          TR = sum(following_step) / sum(total_step)   (optionally revised by track_episode_step if exists)
          CR = sum(collision) / episode_count
    """
    agent = UniNaVid_Agent(model_path, save_path)
    
    # Keep backward-compatible defaults by only overriding values when CLI provided.
    sampling_config = {
        "sampling_strategy": (
            sampling_strategy
            if sampling_strategy is not None
            else getattr(agent.model.config, "sampling_strategy", None)
        ),
        "sampling_k": (
            sampling_k if sampling_k is not None else getattr(agent.model.config, "sampling_k", None)
        ),
        "sampling_stride": (
            sampling_stride
            if sampling_stride is not None
            else getattr(agent.model.config, "sampling_stride", None)
        ),
        "sampling_seed": (
            sampling_seed
            if sampling_seed is not None
            else getattr(agent.model.config, "sampling_seed", None)
        ),
        "token_ablation_mode": (
            token_ablation_mode
            if token_ablation_mode is not None
            else getattr(agent.model.config, "token_ablation_mode", None)
        ),
    }
    # In ablation mode, explicitly disable legacy sampling switches.
    if sampling_config.get("token_ablation_mode"):
        sampling_config["sampling_strategy"] = None
        sampling_config["sampling_k"] = None
        sampling_config["sampling_stride"] = None
        sampling_config["sampling_seed"] = None

    for key, value in sampling_config.items():
        setattr(agent.model.config, key, value)
        
    effective_seed = int(seed) if seed is not None else sampling_config.get("sampling_seed")
    if effective_seed is not None:
        set_global_seed(int(effective_seed))
    if is_random_strategy(sampling_config.get("sampling_strategy")) and seed is None and sampling_seed is None:
        print("[WARNING] Detected stochastic sampling strategy but no explicit seed was provided.")

    run_id = f"split{split_id if split_id is not None else -1}_{int(time.time())}"
    strategy_tag = sampling_config.get("sampling_strategy")

    # Default output paths
    if split_id is None:
        split_id = -1
    if enable_step_stats and step_stats_path is None:
        step_stats_path = os.path.join(save_path, f"step_stats_split{split_id}.jsonl")
    if enable_step_stats and speed_summary_path is None:
        speed_summary_path = os.path.join(save_path, f"speed_summary_split{split_id}.json")
    if enable_step_stats and metrics_summary_path is None:
        metrics_summary_path = os.path.join(save_path, f"metrics_summary_split{split_id}.json")
        
    run_meta = {
        "git_commit": get_git_commit_hash(),
        "model_path": model_path,
        "sampling": dict(sampling_config),
        "seed": effective_seed,
        "split_config": {
            "split_id": split_id,
            "episode_count": len(getattr(dataset_split, "episodes", [])) if hasattr(dataset_split, "episodes") else None,
        },
        "run_id": run_id,
    }
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    writer = JSONLWriter(step_stats_path) if enable_step_stats else None
    cuda_start, cuda_end, cuda_sync_elapsed_ms, cuda_enabled = maybe_cuda_timer()

    def _write(obj: dict) -> None:
        if writer is not None:
            writer.write(obj)

    # Accumulators for split-level summaries
    episode_count = 0
    succ_count = 0
    collision_sum = 0.0
    total_following_steps = 0
    total_steps_weight = 0  # may be revised by track_episode_step

    # Speed accumulators (wall-clock breakdown)
    step_total_ms_all: List[float] = []
    step_obs_ms_all: List[float] = []
    step_taskobs_ms_all: List[float] = []
    step_act_wall_ms_all: List[float] = []
    step_act_cuda_ms_all: List[float] = []
    step_env_ms_all: List[float] = []
    step_getm_ms_all: List[float] = []
    step_vis_tokens_all: List[float] = []
    step_vis_nav_tokens_all: List[float] = []
    step_vis_total_visual_tokens_all: List[float] = []

    def _get_latest_visual_token_stats() -> dict:
        """Read latest visual-token runtime stats from model (if available)."""
        runtime_holder = None
        if hasattr(agent, "model"):
            runtime_holder = agent.model
            if not hasattr(runtime_holder, "get_runtime_stats") and hasattr(runtime_holder, "get_model"):
                try:
                    runtime_holder = runtime_holder.get_model()
                except Exception:
                    runtime_holder = None

        if runtime_holder is None or not hasattr(runtime_holder, "get_runtime_stats"):
            return {}

        try:
            stats = runtime_holder.get_runtime_stats() or {}
        except Exception:
            return {}

        steps = stats.get("vis_tokens_llm_steps", [])
        structs = stats.get("vis_tokens_llm_structure", [])
        step_latest = steps[-1] if steps else None
        struct_latest = structs[-1] if structs else {}
        if not isinstance(struct_latest, dict):
            struct_latest = {}

        return {
            "vis_tokens_step": int(step_latest) if step_latest is not None else None,
            "vis_tokens_total": stats.get("vis_tokens_llm_total", None),
            "vis_structure_step": {
                "history_blocks": struct_latest.get("history_blocks", []),
                "nav_tokens": struct_latest.get("nav_tokens", None),
                "total_visual_tokens": struct_latest.get("total_visual_tokens", None),
            },
        }

    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        sim = env.sim
        agent.reset()

        num_episodes = len(env.episodes)
        for _ in trange(num_episodes):
            _ = env.reset()

            # tell agent which episode it is (for optional dataset logging)
            agent.start_episode(env.current_episode)

            # per-episode instruction (do NOT reuse a single instruction across episodes)
            instruction = env.current_episode.info.get("instruction", "")

            # lighting setup (kept same as original)
            light_setup = [
                LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
            ]
            sim.set_light_setup(light_setup)

            result: dict = {}
            record_infos: List[dict] = []

            finished = False
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            iter_step = 0
            followed_step = 0
            too_far_count = 0
            status = "Normal"

            info = env.get_metrics()

            # per-episode speed stats
            ep_step_total_ms: List[float] = []
            ep_step_act_wall_ms: List[float] = []
            ep_step_act_cuda_ms: List[float] = []

            while not env.episode_over:
                step_wall = WallTimer()

                # (1) sensor obs
                t_obs = WallTimer()
                obs = sim.get_sensor_observations()
                obs_ms = t_obs.ms()

                # (2) task obs (kept to preserve original behavior; might be a no-op depending on task)
                t_task = WallTimer()
                _ = env.task._get_observations(env.current_episode)
                task_ms = t_task.ms()

                # (3) model/agent act
                t_act = WallTimer()
                act_cuda_ms = None
                if cuda_enabled and cuda_start is not None and cuda_end is not None and cuda_sync_elapsed_ms is not None:
                    cuda_start()
                action = agent.act(obs, info, instruction, env.current_episode.episode_id)
                vis_token_stats = _get_latest_visual_token_stats() if enable_step_stats else {}
                if cuda_enabled and cuda_start is not None and cuda_end is not None and cuda_sync_elapsed_ms is not None:
                    cuda_end()
                    try:
                        act_cuda_ms = cuda_sync_elapsed_ms()
                    except Exception:
                        act_cuda_ms = None
                act_wall_ms = t_act.ms()

                # build action dict (same as original)
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

                # (4) env.step
                t_env = WallTimer()
                iter_step += 1
                env.step(action_dict)
                env_ms = t_env.ms()

                # (5) metrics
                t_m = WallTimer()
                info = env.get_metrics()
                getm_ms = t_m.ms()

                total_ms = step_wall.ms()
                hz = hz_from_ms(total_ms)

                # episode bookkeeping for TR / early stop reasons
                if info.get("human_following", 0.0) == 1.0:
                    followed_step += 1
                    too_far_count = 0
                else:
                    # do not reset too_far_count here (matches original)
                    pass

                dis_to_human = float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos))

                if dis_to_human > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        status = "Lost"
                        finished = False
                        # log final step before break
                        pass

                # collision early break
                if info.get("human_collision", 0.0) == 1.0:
                    status = "Collision"
                    finished = False

                # record per-step info for json (original behavior)
                record_infos.append(
                    {
                        "step": iter_step,
                        "dis_to_human": dis_to_human,
                        "facing": info.get("human_following", 0.0),
                    }
                )

                # step stats logging
                if enable_step_stats and (log_every_n_steps <= 1 or (iter_step % log_every_n_steps) == 0):
                    scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split(".")[0]
                    _write(
                        {
                            "type": "step",
                            "split_id": split_id,
                            "scene_key": scene_key,
                            "episode_id": str(env.current_episode.episode_id),
                            "step": iter_step,
                            "action": action,
                            "timing_ms": {
                                "obs": obs_ms,
                                "task_obs": task_ms,
                                "act_wall": act_wall_ms,
                                "act_cuda": act_cuda_ms,
                                "env_step": env_ms,
                                "get_metrics": getm_ms,
                                "total": total_ms,
                            },
                            "hz": hz,
                            "env_metrics": {
                                "human_following": info.get("human_following", None),
                                "human_following_success": info.get("human_following_success", None),
                                "human_collision": info.get("human_collision", None),
                                "dis_to_human": dis_to_human,
                                "too_far_count": too_far_count,
                            },
                            "vis_tokens_step": vis_token_stats.get("vis_tokens_step", None),
                            "vis_structure_step": vis_token_stats.get(
                                "vis_structure_step",
                                {
                                    "history_blocks": [],
                                    "nav_tokens": None,
                                    "total_visual_tokens": None,
                                },
                            ),
                            "vis_tokens_total": vis_token_stats.get("vis_tokens_total", None),
                            "sampling_config": dict(sampling_config),
                            "run_id": run_id,
                            "seed": effective_seed,
                            "strategy": strategy_tag,
                        }
                    )

                # accumulate speed stats
                step_total_ms_all.append(total_ms)
                step_obs_ms_all.append(obs_ms)
                step_taskobs_ms_all.append(task_ms)
                step_act_wall_ms_all.append(act_wall_ms)
                if act_cuda_ms is not None:
                    step_act_cuda_ms_all.append(act_cuda_ms)
                step_env_ms_all.append(env_ms)
                step_getm_ms_all.append(getm_ms)
                if enable_step_stats:
                    vis_step = vis_token_stats.get("vis_tokens_step", None)
                    if vis_step is not None:
                        step_vis_tokens_all.append(float(vis_step))

                    vis_struct = vis_token_stats.get("vis_structure_step", {})
                    if isinstance(vis_struct, dict):
                        nav_tokens = vis_struct.get("nav_tokens", None)
                        total_visual_tokens = vis_struct.get("total_visual_tokens", None)
                        if nav_tokens is not None:
                            step_vis_nav_tokens_all.append(float(nav_tokens))
                        if total_visual_tokens is not None:
                            step_vis_total_visual_tokens_all.append(float(total_visual_tokens))

                ep_step_total_ms.append(total_ms)
                ep_step_act_wall_ms.append(act_wall_ms)
                if act_cuda_ms is not None:
                    ep_step_act_cuda_ms.append(act_cuda_ms)

                # break conditions exactly like original
                if status in ("Collision", "Lost"):
                    break

            # episode end
            info = env.get_metrics()
            agent.reset(env.current_episode)

            if env.episode_over:
                finished = True

            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split(".")[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)

            # write per-step distance/facing debug info (original)
            with open(os.path.join(save_dir, f"{env.current_episode.episode_id}_info.json"), "w") as f:
                json.dump(record_infos, f, indent=2)

            # SR/TR/CR-compatible per-episode result
            result["finish"] = finished
            result["status"] = status
            if iter_step < 300:
                result["success"] = bool(info.get("human_following_success", False) and info.get("human_following", 0.0))
            else:
                result["success"] = bool(info.get("human_following", 0.0))
            result["following_rate"] = float(followed_step / iter_step) if iter_step > 0 else 0.0
            result["following_step"] = int(followed_step)
            result["total_step"] = int(iter_step)
            result["collision"] = float(info.get("human_collision", 0.0))

            with open(os.path.join(save_dir, f"{env.current_episode.episode_id}.json"), "w") as f:
                json.dump(result, f, indent=2)

            # episode summary record
            if enable_step_stats:
                _write(
                    {
                        "type": "episode_summary",
                        "split_id": split_id,
                        "scene_key": scene_key,
                        "episode_id": str(env.current_episode.episode_id),
                        "result": result,
                        "timing_ms_summary": {
                            "step_total": summarize_ms(ep_step_total_ms),
                            "act_wall": summarize_ms(ep_step_act_wall_ms),
                            "act_cuda": summarize_ms(ep_step_act_cuda_ms),
                        },
                        "sampling_config": dict(sampling_config),
                        "run_id": run_id,
                        "seed": effective_seed,
                        "strategy": strategy_tag,
                    }
                )

            # split-level aggregation (consistent with analyze_results.py)
            episode_count += 1
            if result["success"]:
                succ_count += 1
            collision_sum += float(result["collision"])
            total_following_steps += int(result["following_step"])

            # "revised total_step" behavior (if track_episode_step exists)
            revised_total_step = int(result["total_step"])
            try:
                revised_path = os.path.join("track_episode_step", scene_key, f"{env.current_episode.episode_id}.json")
                if os.path.exists(revised_path):
                    with open(revised_path, "r") as rf:
                        revised_data = json.load(rf)
                    if isinstance(revised_data, dict) and "total_step" in revised_data:
                        revised_total_step = max(int(revised_data["total_step"]), revised_total_step)
            except Exception:
                pass
            total_steps_weight += revised_total_step

    # write split summaries
    if enable_step_stats:
        metrics_summary = {
            "episode_count": episode_count,
            "success_rate": (succ_count / episode_count) if episode_count else None,
            "following_rate": (total_following_steps / total_steps_weight) if total_steps_weight else None,
            "collision_rate": (collision_sum / episode_count) if episode_count else None,
            "sampling_config": dict(sampling_config),
            "run_id": run_id,
            "seed": effective_seed,
            "strategy": strategy_tag,
        }
        speed_summary = {
            "run_id": run_id,
            "seed": effective_seed,
            "strategy": strategy_tag,
            "step_total_ms": summarize_ms(step_total_ms_all),
            "obs_ms": summarize_ms(step_obs_ms_all),
            "task_obs_ms": summarize_ms(step_taskobs_ms_all),
            "act_wall_ms": summarize_ms(step_act_wall_ms_all),
            "act_cuda_ms": summarize_ms(step_act_cuda_ms_all),
            "env_step_ms": summarize_ms(step_env_ms_all),
            "get_metrics_ms": summarize_ms(step_getm_ms_all),
            "vis_tokens_step": summarize_ms(step_vis_tokens_all),
            "vis_structure_step_nav_tokens": summarize_ms(step_vis_nav_tokens_all),
            "vis_structure_step_total_visual_tokens": summarize_ms(step_vis_total_visual_tokens_all),
            "approx_hz_mean": hz_from_ms(speed_summary_path and summarize_ms(step_total_ms_all).get("mean") or 0.0),
        }

        # fix approx_hz_mean calculation
        total_mean = speed_summary["step_total_ms"].get("mean")
        speed_summary["approx_hz_mean"] = hz_from_ms(total_mean) if total_mean else None

        if metrics_summary_path:
            with open(metrics_summary_path, "w", encoding="utf-8") as f:
                json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
        if speed_summary_path:
            with open(speed_summary_path, "w", encoding="utf-8") as f:
                json.dump(speed_summary, f, indent=2, ensure_ascii=False)

    if writer is not None:
        writer.close()


class UniNaVid_Agent(Agent):
    def __init__(self, model_path, result_path, exp_save='video'):
        print("Initialize UniNaVid")

        self.result_path = result_path
        self.require_map = True if "video" in exp_save else False
        self.require_data = True if "video" in exp_save else False

        self.conv_mode = "vicuna_v1"
        
        if self.require_map or self.require_data:
            os.makedirs(self.result_path, exist_ok=True)

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))
        # runtime device (respects CUDA_VISIBLE_DEVICES)
        try:
            self.device = next(self.model.parameters()).device
        except Exception:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        print("Initialization Complete")

        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, back, or stop."
        # If 1: use the model's 'next four actions' as a queue (open-loop for 4 steps)
        self.use_action_queue = bool(int(os.environ.get('TRACKVLA_USE_ACTION_QUEUE', '0')))


        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []

        self.count_id = 0

        # ====== 用于样本采集的状态 ======
        self.current_episode = None  # 当前 episode 对象
        self.step_idx = 0  # 当前 episode 内步数

        # 是否开启数据采集（如果只想评测，设成 False）
        # dataset logging is expensive; default OFF for benchmarking
        self.enable_logging = bool(int(os.environ.get('TRACKVLA_ENABLE_LOGGING', '0')))

        if self.enable_logging:
            dataset_root = os.path.join(self.result_path, "uninavid_track_dataset")
            # fps = 4，最多保留 32 帧历史，你可以按需调整
            self.logger = TrackDatasetLogger(
                root_dir=dataset_root,
                fps=4,
                max_history_frames=32,
                log_every_n_steps=1,
            )
        else:
            self.logger = None
        # ====================================

        self.reset()


    def start_episode(self, episode: NavigationEpisode) -> None:
        """
        在每个 episode 开始时调用，只是记录当前 episode 和重置步计数。
        不做图像缓存的 reset（缓存由 reset 负责）。
        """
        self.current_episode = episode
        self.step_idx = 0


    def process_images(self, rgb_list):
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().to(self.device)

        return [video]


    def predict_inference(self, prompt):
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].to(self.device)
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].to(self.device)
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].to(self.device)
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].to(self.device)
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].to(self.device)
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].to(self.device)

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(self.device)
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

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []

        cur_prompt = question
        with torch.inference_mode():
            # Some Uni-NaVid checkpoints expose update_prompt for online caching, others don't.
            if hasattr(self.model, 'update_prompt'):
                try:
                    self.model.update_prompt([[cur_prompt]])
                except Exception:
                    pass
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs


    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:
            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image


    def reset(self, episode: NavigationEpisode = None):
        if len(self.topdown_map_list) != 0:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.topdown_map_list)

            print(f"Successfully save the episode video with episode id {episode.episode_id}")

        self.history_rgb_tensor = None
        self.transformation_list = []
        self.rgb_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0
        self.first_forward = False

    def _action_to_vel(self, a: str):
        a = (a or "").strip().lower()
        if a == "stop":
            return [0.0, 0.0, 0.0]
        if a == "forward":
            return [0.5, 0.0, 0.0]
        if a == "left":
            return [0.0, 0.0, 1.0]
        if a == "right":
            return [0.0, 0.0, -1.0]
        if a == "back":
            return [-0.67, 0.0, 0.0]
        # unknown -> safe stop
        return [0.0, 0.0, 0.0]

    def _parse_actions(self, navigation: str) -> List[str]:
        # Robustly extract {forward,left,right,back,stop} from free-form text.
        nav = (navigation or "").lower()
        actions = re.findall(r"(forward|left|right|back|stop)", nav)
        return actions

    def act(self, observations, info, instruction, episode_id):
        self.episode_id = episode_id
        rgb = observations["agent_1_articulated_agent_jaw_rgb"][:,:,:3]
        self.rgb_list.append(rgb)

        # 新增：步计数（用于 sample id）
        self.step_idx += 1

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_following"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            if self.require_map:
                img = self.addtext(output_im, instruction, "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            
            return self._action_to_vel(temp_action)

        navigation_qs = self.promt_template.format(instruction)
        navigation = self.predict_inference(navigation_qs)
        print("Output actions: ", navigation)
        if self.require_map:
            img = self.addtext(output_im, instruction, navigation)
            self.topdown_map_list.append(img)

            # ====== 记录一条样本 ======
            if self.logger is not None and self.current_episode is not None:
                # 使用当前 episode 从起始到当前 step 的所有 RGB 作为“历史视频”
                # logger 内部会自动裁成最多 max_history_frames 帧
                self.logger.log_step(
                    frames=list(self.rgb_list),  # 浅拷贝一份列表，避免后续修改
                    prompt=navigation_qs,
                    actions_str=navigation,
                    episode=self.current_episode,
                    step_idx=self.step_idx,
                )
            # ==============================

        actions = self._parse_actions(navigation)
        if not actions:
            # avoid crashing evaluation due to formatting noise
            actions = ["stop"]

        # Optional: consume the model's "next four actions" as an open-loop queue
        if self.use_action_queue and len(actions) > 1:
            self.pending_action_list.extend(actions[1:4])

        action = self._action_to_vel(actions[0])
        return action


        
