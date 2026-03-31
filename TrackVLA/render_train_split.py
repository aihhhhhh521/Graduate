import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--exp-config", required=True, help="Habitat/TrackVLA config yaml (use STT train config)")
    p.add_argument("--model-path", required=True, help="Uni-NaVid (or compatible) model path")
    p.add_argument("--save-path", required=True, help="Output folder for collected data")
    p.add_argument("--split-num", type=int, default=8, help="Total number of shards (default 8)")
    p.add_argument("--split-id", type=int, default=0, help="Shard index [0, split-num)")
    p.add_argument("--scenes-dir", default=None, help="Root directory for scene datasets (optional, used to skip missing scenes)")

    # GPU controls
    p.add_argument("--cuda-device", type=int, default=None,
                   help="Physical GPU id to use. Sets CUDA_VISIBLE_DEVICES BEFORE imports.")

    p.add_argument("--max-steps", type=int, default=300, help="Episode hard step cap")
    p.add_argument("--episode-video-fps", type=int, default=1, help="Output episode video fps (default 1)")
                   
    p.add_argument("--output-config-dirname", default="configs",
                   help="Directory name under save-path for training config files (default: configs).")
    p.add_argument("--output-video-dirname", default="videos",
                   help="Directory name under save-path for training videos (default: videos).")
    p.add_argument("--output-json-name", default="track_train.json",
                   help="Final training json filename saved under output-config-dirname.")

    return p.parse_args()


def _resolve_scene_path(scene_id: str, scenes_dir: str) -> str:
    if not scene_id:
        return scene_id
    if os.path.isabs(scene_id):
        return scene_id
    sid = scene_id.replace("\\", "/")
    # Strip common redundant prefixes
    for pref in ("data/scene_datasets/", "scene_datasets/"):
        if sid.startswith(pref):
            sid = sid[len(pref):]
            break
    return os.path.join(scenes_dir, sid)

def _materialize_train_layout(save_path: str, config_dirname: str, video_dirname: str, output_json_name: str) -> None:
    root = Path(save_path)
    dataset_root = root / "uninavid_track_dataset"
    src_jsonl = dataset_root / "track_episodes.jsonl"

    out_config_dir = root / config_dirname
    out_video_dir = root / video_dirname
    out_config_dir.mkdir(parents=True, exist_ok=True)
    out_video_dir.mkdir(parents=True, exist_ok=True)

    if not src_jsonl.exists():
        raise FileNotFoundError(f"episode jsonl not found: {src_jsonl}")

    samples = []
    missing_videos = 0
    moved_videos = 0
    kept_videos = 0

    with open(src_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            video_rel = rec.get("video", "")
            if not video_rel:
                continue

            src_abs = dataset_root / video_rel
            if not src_abs.exists():
                missing_videos += 1
                continue

            video_name = src_abs.name
            dst_abs = out_video_dir / video_name
            if src_abs.resolve() != dst_abs.resolve():
                if not dst_abs.exists():
                    shutil.move(str(src_abs), str(dst_abs))
                    moved_videos += 1
                else:
                    kept_videos += 1

            rec["video"] = video_name
            samples.append(rec)

    out_json = out_config_dir / output_json_name
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    layout_manifest = {
        "data_path": str(out_json.relative_to(root)),
        "video_folder": str(out_video_dir.relative_to(root)),
        "num_samples": len(samples),
        "moved_videos": moved_videos,
        "kept_videos": kept_videos,
        "missing_videos": missing_videos,
        "source_jsonl": str(src_jsonl.relative_to(root)),
    }
    with open(out_config_dir / "train_layout_manifest.json", "w", encoding="utf-8") as f:
        json.dump(layout_manifest, f, ensure_ascii=False, indent=2)

    print(
        f"[render_train_split] finalized train layout: "
        f"data_path={layout_manifest['data_path']} video_folder={layout_manifest['video_folder']} "
        f"samples={layout_manifest['num_samples']} missing_videos={missing_videos}"
    )


def _safe_release(vw: cv2.VideoWriter) -> None:
    try:
        vw.release()
    except Exception:
        pass


def _collect_with_origin_agent(args, config, dataset) -> None:
    from habitat.datasets import make_dataset  # noqa: F401
    import habitat
    from habitat_sim.gfx import LightInfo, LightPositionModel
    from agent_uninavid_origin import UniNaVid_Agent

    save_root = Path(args.save_path)
    dataset_root = save_root / "uninavid_track_dataset"
    raw_video_dir = dataset_root / "raw_videos"
    raw_video_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = dataset_root / "track_episodes.jsonl"

    # fresh file for this shard run
    if jsonl_path.exists():
        jsonl_path.unlink()

    episodes = list(getattr(dataset, "episodes", []))

    agent = UniNaVid_Agent(model_path=args.model_path, result_path=str(save_root), exp_save="collect")

    # capture raw model textual output each step (teacher action tokens)
    agent._last_model_output = ""
    _orig_predict = agent.predict_inference

    def _predict_and_capture(prompt):
        out = _orig_predict(prompt)
        agent._last_model_output = out
        return out

    agent.predict_inference = _predict_and_capture

    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        sim = env.sim
        agent.reset()

        for _ in range(len(episodes)):
            obs = env.reset()
            instruction = env.current_episode.info.get("instruction", "")
            info = env.get_metrics()

            light_setup = [
                LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
            ]
            sim.set_light_setup(light_setup)

            scene_key = Path(env.current_episode.scene_id).name.split(".")[0]
            sample_id = f"NAV_ID_TRACK_{scene_key}_{env.current_episode.episode_id}"
            video_rel = f"raw_videos/EP_{scene_key}_{env.current_episode.episode_id}.mp4"
            video_abs = dataset_root / video_rel

            action_tokens = []
            writer = None

            try:
                step = 0
                while (not env.episode_over) and (step < int(args.max_steps)):
                    rgb = obs.get("agent_1_articulated_agent_jaw_rgb")
                    if rgb is None:
                        raise KeyError("Missing required FPV key: agent_1_articulated_agent_jaw_rgb")
                    rgb = rgb[:, :, :3]
                    if rgb.dtype != np.uint8:
                        rgb = rgb.astype(np.uint8)

                    if writer is None:
                        h, w = rgb.shape[:2]
                        video_abs.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(video_abs),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            float(args.episode_video_fps),
                            (w, h),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Failed to open writer: {video_abs}")
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                    act_vec = agent.act(obs, info, instruction, env.current_episode.episode_id)
                    raw = str(getattr(agent, "_last_model_output", "")).strip().lower()
                    first = raw.split()[0] if raw else "stop"
                    if first not in {"forward", "left", "right", "back", "stop"}:
                        first = "stop"
                    action_tokens.append(first)

                    action_dict = {
                        "action": (
                            "agent_0_humanoid_navigate_action",
                            "agent_1_base_velocity",
                            "agent_2_oracle_nav_randcoord_action_obstacle",
                            "agent_3_oracle_nav_randcoord_action_obstacle",
                            "agent_4_oracle_nav_randcoord_action_obstacle",
                            "agent_5_oracle_nav_randcoord_action_obstacle",
                        ),
                        "action_args": {"agent_1_base_vel": act_vec},
                    }
                    obs = env.step(action_dict)
                    info = env.get_metrics()
                    step += 1

            finally:
                if writer is not None:
                    _safe_release(writer)
                try:
                    agent.reset(env.current_episode)
                except Exception:
                    pass

            rec = {
                "id": sample_id,
                "video": video_rel,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "Imagine you are a robot programmed for navigation tasks. "
                            "You have been given a video of historical observations and an image of the current observation <image>. "
                            f"Your assigned task is: '{instruction}'. "
                            "Analyze this series of images to determine your next four actions. "
                            "The predicted action should be one of the following: forward, left, right, back, or stop."
                        ),
                    },
                    {"from": "gpt", "value": " ".join(action_tokens)},
                ],
                "meta": {
                    "steps": len(action_tokens),
                    "split_num": int(args.split_num),
                    "split_id": int(args.split_id),
                    "exp_config": args.exp_config,
                },
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main() -> None:
    args = parse_args()

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    # Helps fragmentation (matches the OOM hint) - safe no-op if already set
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from evt_bench.default import get_config
    from habitat.datasets import make_dataset
    
    config = get_config(args.exp_config)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)

    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)

    # Shard episodes
    if args.split_num and args.split_num > 1:
        eps = list(getattr(dataset, "episodes", []))
        dataset.episodes = [ep for idx, ep in enumerate(eps) if idx % args.split_num == args.split_id]

    # Optionally skip missing scenes (prevents hard crashes)
    if args.scenes_dir:
        scenes_dir = str(Path(args.scenes_dir).expanduser().resolve())
        kept = []
        for ep in dataset.episodes:
            scene_id = getattr(ep, "scene_id", None)
            scene_path = _resolve_scene_path(scene_id, scenes_dir)
            if scene_path and os.path.exists(scene_path):
                kept.append(ep)
        dataset.episodes = kept

    os.makedirs(args.save_path, exist_ok=True)
    with open(Path(args.save_path) / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "exp_config": args.exp_config,
                "model_path": args.model_path,
                "split_num": int(args.split_num),
                "split_id": int(args.split_id),
                "scenes_dir": args.scenes_dir,
                "cuda_device": args.cuda_device,
                "episode_video_fps": int(args.episode_video_fps),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    _collect_with_origin_agent(args=args, config=config, dataset=dataset)
    _materialize_train_layout(
        save_path=args.save_path,
        config_dirname=args.output_config_dirname,
        video_dirname=args.output_video_dirname,
        output_json_name=args.output_json_name,
    )

if __name__ == "__main__":
    main()
