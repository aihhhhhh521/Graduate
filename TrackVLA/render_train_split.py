"""
render_train_split.py — TrackVLA -> Uni-NaVid finetune data collection.

Teacher    : oracle shortest-path (Habitat pathfinder + ground-truth main-humanoid pose).
Vocab      : {forward, left, right, stop}     (no 'back')
Granularity: one record per 4-action chunk    (matches Uni-NaVid stage-2 sample format)
Prompt     : verbatim Uni-NaVid open_uninavid_sampled_500.json template
Layout     : <save-path>/<video-subdir>/EP_<scene>_<ep>_chunkNNN.mp4
             <save-path>/<configs>/track_train.json   (data_path passed to stage-2)
             stage-2 should set --video_folder <save-path>

The 'video' field in each record is "<video-subdir>/EP_<scene>_<ep>_chunkNNN.mp4",
so train.py at  os.path.join(video_folder, rec["video"])  resolves correctly.
"""

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Action space (no 'back') and velocity mapping for agent_1_base_velocity.
# Mirrors agent_uninavid_origin.act() except 'back' has been removed.
# -----------------------------------------------------------------------------
ACTION_VOCAB = ("forward", "left", "right", "stop")
ACTION_TO_VEL = {
    "forward": [0.5, 0.0, 0.0],
    "left":    [0.0, 0.0, 1.0],
    "right":   [0.0, 0.0, -1.0],
    "stop":    [0.0, 0.0, 0.0],
}

# Uni-NaVid stage-2 sample prompt — taken verbatim from
# Uni-NaVid/data/Nav-Finetune/open_uninavid_sampled_500.json (action set excludes 'back').
PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of "
    "the current observation <image>. "
    "Your assigned task is: '{instruction}'. "
    "Analyze this series of images to determine your next four actions. "
    "The predicted action should be one of the following: forward, left, right, or stop."
)

# Uni-NaVid emits 4 action tokens per decision step.
CHUNK = 4

# Action-tuple template covering every agent in the STT yaml (agent_0..agent_8).
# agent_1 is the robot we're controlling; agents 0/2..8 take their oracle actions.
DEFAULT_ACTION_TUPLE = (
    "agent_0_humanoid_navigate_action",
    "agent_1_base_velocity",
    "agent_2_oracle_nav_randcoord_action_obstacle",
    "agent_3_oracle_nav_randcoord_action_obstacle",
    "agent_4_oracle_nav_randcoord_action_obstacle",
    "agent_5_oracle_nav_randcoord_action_obstacle",
    "agent_6_oracle_nav_randcoord_action_obstacle",
    "agent_7_oracle_nav_randcoord_action_obstacle",
    "agent_8_oracle_nav_randcoord_action_obstacle",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--exp-config", required=True,
                   help="Habitat/TrackVLA config yaml (e.g. STT train config).")
    p.add_argument("--save-path", required=True,
                   help="Output root. Stage-2 should pass this as --video_folder.")
    p.add_argument("--split-num", type=int, default=8)
    p.add_argument("--split-id", type=int, default=0)
    p.add_argument("--scenes-dir", default=None,
                   help="If given, episodes whose scene file is missing under this root are skipped.")

    p.add_argument("--cuda-device", type=int, default=None,
                   help="Physical GPU id. Sets CUDA_VISIBLE_DEVICES BEFORE imports and "
                        "pins habitat_sim_v0.gpu_device_id to the masked-logical 0.")
    p.add_argument("--keep-extra-sensors", action="store_true",
                   help="By default we drop all sensors except agent_1 jaw_rgb (+ the task's "
                        "main_humanoid_detector for metrics) to keep render-time VRAM bounded. "
                        "Pass this flag to keep the full yaml sensor list (panoptic, every "
                        "third_rgb, every head_rgb on humans). Heavy.")
    p.add_argument("--max-steps", type=int, default=300,
                   help="Episode hard step cap (matches yaml max_episode_steps default).")
    p.add_argument("--episode-video-fps", type=int, default=1,
                   help="mp4 fps; must match Uni-NaVid stage-2 --video_fps (default 1).")

    # Oracle teacher thresholds.
    p.add_argument("--follow-radius", type=float, default=1.5,
                   help="Within this XZ-distance from the main human, oracle emits 'stop'.")
    p.add_argument("--angle-thresh", type=float, default=0.4,
                   help="If |bearing| (rad) <= thresh, oracle emits 'forward'; otherwise 'left'/'right'. "
                        "Default ~23 deg keeps the robot from oscillating when the human is mid-turn.")

    # Output layout.
    p.add_argument("--video-subdir", default="track_videos",
                   help="Subdirectory written into the JSON 'video' field. "
                        "Videos saved under save-path/<video-subdir>/.")
    p.add_argument("--output-config-dirname", default="configs",
                   help="Where the final JSON goes under save-path.")
    p.add_argument("--output-json-name", default="track_train.json",
                   help="Filename of the final Uni-NaVid-compatible JSON.")

    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _resolve_scene_path(scene_id: str, scenes_dir: str) -> str:
    if not scene_id:
        return scene_id
    if os.path.isabs(scene_id):
        return scene_id
    sid = scene_id.replace("\\", "/")
    for pref in ("data/scene_datasets/", "scene_datasets/"):
        if sid.startswith(pref):
            sid = sid[len(pref):]
            break
    return os.path.join(scenes_dir, sid)


def _safe_release(vw: cv2.VideoWriter) -> None:
    try:
        vw.release()
    except Exception:
        pass


def _hard_cuda_cleanup() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _slim_sensors(config) -> dict:
    """
    Drop everything we don't need for data collection so habitat-sim allocates
    fewer GPU textures and runs fewer render passes per env.step().

    Kept:
      * agent_1.sim_sensors.jaw_rgb_sensor          (the FPV we record to mp4)
      * agent_1_main_humanoid_detector_sensor       (task uses it for human_following metric)
      * agent_1_localization_sensor                 (cheap, sometimes referenced by task code)

    Dropped (default):
      * jaw_panoptic_sensor on agent_1              (categorical buffer; expensive)
      * third_rgb_sensor   on every agent           (extra render pass per agent per step)
      * head_rgb_sensor    on every humanoid agent  (8 extra render passes per step in STT)

    Returns a dict describing what was actually pruned, for the run manifest.
    """
    from omegaconf import OmegaConf

    OmegaConf.set_struct(config, False)

    # --- 1) Trim gym.obs_keys so the env doesn't request what we drop. ---
    keep_obs = {
        "agent_1_articulated_agent_jaw_rgb",
        "agent_1_main_humanoid_detector_sensor",
        "agent_1_other_humanoid_detector_sensor",  # used by some task metrics; cheap
        "agent_1_localization_sensor",
    }
    orig_obs = list(getattr(config.habitat.gym, "obs_keys", []) or [])
    new_obs = [k for k in orig_obs if k in keep_obs]
    config.habitat.gym.obs_keys = new_obs

    # --- 2) Trim each agent's sim_sensors. ---
    pruned_sensors = {}
    agents_cfg = config.habitat.simulator.agents
    drop_per_agent = {
        # On the controlled robot, drop panoptic + 3rd-person.
        "agent_1": ("jaw_panoptic_sensor", "third_rgb_sensor"),
    }
    # On every humanoid (following_humanoid setup), drop both RGBs (we don't use them).
    humanoid_drop = ("third_rgb_sensor", "head_rgb_sensor")

    for agent_name in list(agents_cfg.keys()):
        agent_cfg = agents_cfg[agent_name]
        sim_sensors = getattr(agent_cfg, "sim_sensors", None)
        if sim_sensors is None:
            continue
        if agent_name in drop_per_agent:
            drop_keys = drop_per_agent[agent_name]
        else:
            drop_keys = humanoid_drop
        gone = []
        for k in list(sim_sensors.keys()):
            if k in drop_keys:
                del sim_sensors[k]
                gone.append(k)
        if gone:
            pruned_sensors[agent_name] = gone

    return {
        "obs_keys_before": orig_obs,
        "obs_keys_after": new_obs,
        "pruned_sensors_by_agent": pruned_sensors,
    }


def _to_xyz(v) -> np.ndarray:
    """Coerce magnum.Vector3 / numpy / list to np.float32 (x, y, z)."""
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return np.array([float(v.x), float(v.y), float(v.z)], dtype=np.float32)
    arr = np.asarray(v, dtype=np.float32).flatten()
    return arr[:3]


def _robot_forward_xz(robot) -> tuple:
    """
    Return the robot's world-frame forward direction projected to XZ as (fx, fz).

    Why not robot.base_rot? In habitat-lab/.../articulated_agent_base.py:170
    `base_rot` is implemented as `self.sim_obj.rotation.angle()`, which is the
    UNSIGNED magnitude of the quaternion rotation -- it cannot tell facing
    direction (yaw=-30 deg and yaw=+30 deg both come back as +30 deg). Using it
    to build a forward vector is what made the robot rock left/right in place.

    Agent local forward in habitat-sim is -Z. We transform that local axis by
    the agent's full base_transformation (rotation part) to get the true world
    forward, regardless of yaw sign.
    """
    try:
        import magnum as mn
        fwd_local = mn.Vector3(0.0, 0.0, -1.0)
        fwd_world = robot.base_transformation.transform_vector(fwd_local)
        return float(fwd_world.x), float(fwd_world.z)
    except Exception:
        # Last-resort fallback (works only if base_rot happens to be signed).
        yaw = float(robot.base_rot)
        return math.sin(yaw), -math.cos(yaw)


def _oracle_action(sim, robot_pos: np.ndarray, fx: float, fz: float,
                   human_pos: np.ndarray, follow_radius: float,
                   angle_thresh: float) -> str:
    """
    Discrete action from the next shortest-path waypoint toward the human.

    Bearing convention (right-hand rule, +Y up):
        cross_y = fz*dx - fx*dz  > 0  =>  waypoint is to robot's LEFT
                                     < 0  =>  waypoint is to robot's RIGHT
    """
    rx, _, rz = float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])
    hx, _, hz = float(human_pos[0]), float(human_pos[1]), float(human_pos[2])

    if math.hypot(hx - rx, hz - rz) <= follow_radius:
        return "stop"

    # Try shortest path through the navmesh.
    wx, wz = hx, hz
    try:
        import habitat_sim
        path = habitat_sim.ShortestPath()
        path.requested_start = robot_pos.astype(np.float32)
        path.requested_end = human_pos.astype(np.float32)
        if sim.pathfinder.find_path(path) and len(path.points) >= 2:
            wp = path.points[1]
            wx, wz = float(wp[0]), float(wp[2])
    except Exception:
        # Fallback to direct line if pathfinder isn't ready.
        pass

    dx, dz = wx - rx, wz - rz
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return "stop"

    n = math.hypot(dx, dz)
    fn = math.hypot(fx, fz) + 1e-9
    cos_b = (fx * dx + fz * dz) / (n * fn + 1e-9)
    cross_y = fz * dx - fx * dz
    bearing = math.atan2(cross_y, cos_b)   # signed, in (-pi, pi]

    if abs(bearing) < angle_thresh:
        return "forward"
    return "left" if bearing > 0 else "right"


def _write_clip(frames, dst_path: Path, fps: int) -> bool:
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        return False
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        _safe_release(writer)
    return True


# -----------------------------------------------------------------------------
# Main collection loop
# -----------------------------------------------------------------------------
def _collect(args, config, dataset) -> int:
    import habitat
    from habitat_sim.gfx import LightInfo, LightPositionModel

    save_root = Path(args.save_path)
    video_subdir = args.video_subdir.strip().strip("/").replace("\\", "/")
    video_dir = save_root / video_subdir
    config_dir = save_root / args.output_config_dirname
    video_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = config_dir / args.output_json_name

    light_setup = [
        LightInfo(vector=[ 10.0, -2.0,  0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
        LightInfo(vector=[-10.0, -2.0,  0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
        LightInfo(vector=[  0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
        LightInfo(vector=[  0.0, -2.0,-10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
    ]

    samples = []
    skipped_episodes = 0

    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        sim = env.sim
        humanoid_main = sim.agents_mgr[0].articulated_agent
        robot = sim.agents_mgr[1].articulated_agent

        for _ in range(len(env.episodes)):
            obs = env.reset()
            sim.set_light_setup(light_setup)
            ep = env.current_episode
            instruction = ep.info.get("instruction", "")
            scene_key = Path(ep.scene_id).name.split(".")[0]
            ep_id = ep.episode_id

            frames = []           # list[np.uint8 RGB]
            tokens = []           # parallel list[str] of oracle actions

            try:
                step = 0
                while (not env.episode_over) and (step < int(args.max_steps)):
                    rgb = obs.get("agent_1_articulated_agent_jaw_rgb")
                    if rgb is None:
                        raise KeyError("Missing FPV sensor: agent_1_articulated_agent_jaw_rgb")
                    rgb = rgb[:, :, :3]
                    if rgb.dtype != np.uint8:
                        rgb = rgb.astype(np.uint8)
                    frames.append(rgb)

                    robot_pos = _to_xyz(robot.base_pos)
                    human_pos = _to_xyz(humanoid_main.base_pos)
                    fx, fz = _robot_forward_xz(robot)
                    token = _oracle_action(
                        sim, robot_pos, fx, fz, human_pos,
                        args.follow_radius, args.angle_thresh,
                    )
                    tokens.append(token)

                    action_dict = {
                        "action": DEFAULT_ACTION_TUPLE,
                        "action_args": {"agent_1_base_vel": ACTION_TO_VEL[token]},
                    }
                    obs = env.step(action_dict)
                    step += 1
            except Exception as e:
                print(f"[render_train_split] WARN ep {ep_id}: {e!r}; skipping.")
                skipped_episodes += 1
                _hard_cuda_cleanup()
                continue
            finally:
                _hard_cuda_cleanup()

            T = len(tokens)
            if T == 0:
                continue

            # One sample per 4-step chunk. For chunk c with step indices [t0..t0+3]:
            #   * video       = frames[: t0 + 1]               (history + "current" frame)
            #   * gpt label   = tokens[t0 : t0 + 4], padded with 'stop' if needed
            n_chunks = math.ceil(T / CHUNK)
            for ci in range(n_chunks):
                t0 = ci * CHUNK
                label = tokens[t0:t0 + CHUNK]
                if len(label) < CHUNK:
                    label = label + ["stop"] * (CHUNK - len(label))

                clip_end = min(t0 + 1, len(frames))
                clip = frames[:clip_end]
                if not clip:
                    continue

                clip_name = f"EP_{scene_key}_{ep_id}_chunk{ci:03d}.mp4"
                clip_abs = video_dir / clip_name
                if not _write_clip(clip, clip_abs, args.episode_video_fps):
                    continue

                samples.append({
                    "id": f"NAV_ID_TRACK_{scene_key}_{ep_id}_chunk{ci:03d}",
                    "video": f"{video_subdir}/{clip_name}",
                    "conversations": [
                        {"from": "human", "value": PROMPT_TEMPLATE.format(instruction=instruction)},
                        {"from": "gpt", "value": " ".join(label)},
                    ],
                    "meta": {
                        "scene": scene_key,
                        "episode_id": ep_id,
                        "chunk_index": ci,
                        "chunk_t0": t0,
                        "episode_steps": T,
                        "split_num": int(args.split_num),
                        "split_id": int(args.split_id),
                    },
                })

            # Hard release per-episode buffers. The frames list can hit ~260MB CPU RAM
            # for 300 steps @ 480x640x3 uint8; explicit clear keeps long shards bounded.
            frames.clear()
            tokens.clear()
            del frames, tokens
            _hard_cuda_cleanup()

            # Periodically flush partial JSON so a crash mid-shard still leaves usable data.
            if len(samples) and (len(samples) % 64 == 0):
                try:
                    with open(out_json_path, "w", encoding="utf-8") as fpart:
                        json.dump(samples, fpart, ensure_ascii=False, indent=2)
                except Exception:
                    pass

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    manifest = {
        "data_path_for_stage2": str(out_json_path.relative_to(save_root)).replace("\\", "/"),
        "video_folder_for_stage2": ".",
        "absolute_save_path": str(save_root.resolve()),
        "video_subdir_in_record": video_subdir,
        "num_samples": len(samples),
        "skipped_episodes": skipped_episodes,
        "chunk_size": CHUNK,
        "action_vocab": list(ACTION_VOCAB),
    }
    with open(config_dir / "train_layout_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(
        f"[render_train_split] wrote {len(samples)} samples (skipped {skipped_episodes} eps) "
        f"-> {out_json_path}"
    )
    return len(samples)


def main() -> None:
    args = parse_args()

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from evt_bench.default import get_config
    from habitat.datasets import make_dataset

    config = get_config(args.exp_config)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)

    # Route GPU. CUDA_VISIBLE_DEVICES masks devices, so the post-mask logical id is 0.
    sensor_prune_report = {"pruned": False}
    try:
        from omegaconf import OmegaConf
        OmegaConf.set_struct(config, False)
        if args.cuda_device is not None:
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
        if not args.keep_extra_sensors:
            sensor_prune_report = _slim_sensors(config)
            sensor_prune_report["pruned"] = True
    except Exception as e:
        print(f"[render_train_split] WARN sensor/gpu config patching failed: {e!r}")

    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)

    # Shard episodes by split.
    if args.split_num and args.split_num > 1:
        eps = list(getattr(dataset, "episodes", []))
        dataset.episodes = [ep for idx, ep in enumerate(eps) if idx % args.split_num == args.split_id]

    # Optionally drop episodes whose scene file is missing locally.
    if args.scenes_dir:
        scenes_dir = str(Path(args.scenes_dir).expanduser().resolve())
        kept = []
        for ep in dataset.episodes:
            sp = _resolve_scene_path(getattr(ep, "scene_id", None), scenes_dir)
            if sp and os.path.exists(sp):
                kept.append(ep)
        dataset.episodes = kept

    os.makedirs(args.save_path, exist_ok=True)
    with open(Path(args.save_path) / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "exp_config": args.exp_config,
            "teacher": "oracle_shortest_path",
            "action_vocab": list(ACTION_VOCAB),
            "chunk_size": CHUNK,
            "follow_radius": args.follow_radius,
            "angle_thresh": args.angle_thresh,
            "max_steps": int(args.max_steps),
            "episode_video_fps": int(args.episode_video_fps),
            "split_num": int(args.split_num),
            "split_id": int(args.split_id),
            "video_subdir": args.video_subdir,
            "output_config_dirname": args.output_config_dirname,
            "output_json_name": args.output_json_name,
            "fpv_sensor_key": "agent_1_articulated_agent_jaw_rgb",
            "dataset_split": str(getattr(config.habitat.dataset, "split", "")),
            "dataset_data_path": str(getattr(config.habitat.dataset, "data_path", "")),
            "cuda_device_physical": args.cuda_device,
            "habitat_sim_gpu_device_id": int(getattr(config.habitat.simulator.habitat_sim_v0, "gpu_device_id", 0)),
            "sensor_prune_report": sensor_prune_report,
        }, f, ensure_ascii=False, indent=2)

    _collect(args=args, config=config, dataset=dataset)


if __name__ == "__main__":
    main()
