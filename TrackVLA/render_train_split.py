"""Render / collect a sharded split of episodes.
Dual-GPU support (recommended):
- Use --cuda-visible-devices "A,B" to expose two physical GPUs.
- Use --habitat-gpu-id 0 (within visible devices) for Habitat-Sim rendering.
- Use --model-gpu-id 1 (within visible devices) for UniNaVid inference.

Backwards compatible:
- --cuda-device <int> still works and exposes a single GPU.

This script intentionally delays heavy imports until AFTER CUDA_VISIBLE_DEVICES is set.
"""

import argparse
import os
from pathlib import Path
import random
import numpy as np

import json
import shutil

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--exp-config", required=True, help="Habitat/TrackVLA config yaml")
    p.add_argument("--model-path", required=True, help="Uni-NaVid (or compatible) model path")
    p.add_argument("--save-path", required=True, help="Output folder for collected data")
    p.add_argument("--split-num", type=int, default=1, help="Total number of shards")
    p.add_argument("--split-id", type=int, default=0, help="Shard index [0, split-num)")
    p.add_argument("--scenes-dir", default=None, help="Root directory for scene datasets (optional, used to skip missing scenes)")

    # GPU controls
    p.add_argument("--cuda-device", type=int, default=None,
                   help="(Legacy) Physical GPU id to use. Sets CUDA_VISIBLE_DEVICES to this single id BEFORE imports.")
    p.add_argument("--cuda-visible-devices", default=None,
                   help='Comma-separated physical GPU ids to expose, e.g. "7,8". Overrides --cuda-device if set.')
    p.add_argument("--habitat-gpu-id", type=int, default=0,
                   help="Habitat-Sim gpu_device_id inside *visible* devices (default 0).")
    p.add_argument("--model-gpu-id", type=int, default=0,
                   help="UniNaVid torch cuda_device inside *visible* devices (default 0). For dual-GPU, set to 1.")

    # Model memory controls
    p.add_argument("--device-map", default=None, help="HF device_map (default: None). If you set auto, it may use ALL visible GPUs.")
    p.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes)")
    p.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    p.add_argument("--max-new-tokens", type=int, default=32, help="Max generation length for action decoding")
    p.add_argument("--do-sample", action="store_true", help="Enable sampling for generation (default off)")
    p.add_argument("--online-cache-prune-mode", default="step_window",
                   help="UniNaVid online visual cache prune mode (default: step_window, recommended for OOM prevention).")
    p.add_argument("--restart-env-every-episodes", type=int, default=10,
                   help="Recreate TrackEnv every N episodes to prevent habitat-sim renderer GPU memory growth/leaks (default 50).")
    p.add_argument("--reuse-action-horizon", action="store_true",
                   help="If set, reuse next-k predicted actions (infer every k steps). Default off = infer every step (normal UniNaVid).")
    p.add_argument("--episode-only", action="store_true",
                   help="Collect ONE full episode video + ONE JSONL record per episode (recommended). Disables clip outputs.")
    p.add_argument("--clip-mode", action="store_true",
                   help="(Legacy) Collect many short clips + clip-level JSONL (sampled_500 style). Not recommended for your current goal.")
                   
    p.add_argument("--output-config-dirname", default="configs",
                   help="Directory name under save-path for training config files (default: configs).")
    p.add_argument("--output-video-dirname", default="videos",
                   help="Directory name under save-path for training videos (default: videos).")
    p.add_argument("--output-json-name", default="track_train.json",
                   help="Final training json filename saved under output-config-dirname.")

    return p.parse_args()


def _resolve_scene_path(scene_id: str, scenes_dir: str) -> str:
    """Resolve episode.scene_id against scenes_dir.

    Handles common cases:
    - scene_id is absolute
    - scene_id is relative like "hm3d/train/.../scene.basis.glb"
    - scene_id is mistakenly prefixed with "data/scene_datasets/..." (will strip that prefix)
    """
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
    """Convert collected episode JSONL to UniNaVid-train-ready JSON and split config/videos folders.

    Expected source (generated by agent_uninavid_withmetrics):
      {save_path}/uninavid_track_dataset/track_episodes.jsonl
      {save_path}/uninavid_track_dataset/raw_videos/*.mp4
    """
    root = Path(save_path)
    dataset_root = root / "uninavid_track_dataset"
    src_jsonl = dataset_root / "track_episodes.jsonl"
    src_video_dir = dataset_root / "raw_videos"

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

def main() -> None:
    args = parse_args()

    # Must be set BEFORE torch/habitat import to take effect
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    elif args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    # Helps fragmentation (matches the OOM hint) - safe no-op if already set
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Optional hint for other libs (harmless if unused)
    os.environ["TRACKVLA_CUDA_DEVICE"] = str(int(args.model_gpu_id))

    # Heavy imports AFTER env is set
    try:
        from evt_bench.default import get_config  # TrackVLA / EVT-style config
    except Exception:
        from habitat.config.default import get_config  # habitat-lab fallback

    from habitat.datasets import make_dataset
    # Use collector that writes FPV episode videos + UniNaVid-style JSONL.
    from agent_uninavid_withmetrics import evaluate_agent
    
    config = get_config(args.exp_config)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)

    # Set habitat-sim GPU id if present
    try:
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(args.habitat_gpu_id)
    except Exception:
        pass

    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)

    # Shard episodes
    if args.split_num and args.split_num > 1:
        eps = list(getattr(dataset, "episodes", []))
        keep = []
        for idx, ep in enumerate(eps):
            if idx % args.split_num == args.split_id:
                keep.append(ep)
        dataset.episodes = keep

    # Optionally skip missing scenes (prevents hard crashes)
    if args.scenes_dir:
        scenes_dir = str(Path(args.scenes_dir).expanduser().resolve())
        kept = []
        missing = 0
        for ep in dataset.episodes:
            try:
                scene_id = getattr(ep, "scene_id")
            except Exception:
                kept.append(ep)
                continue
            scene_path = _resolve_scene_path(scene_id, scenes_dir)
            if os.path.exists(scene_path):
                kept.append(ep)
            else:
                missing += 1
        dataset.episodes = kept
        print(f"[render_train_split] episodes kept={len(kept)} missing_scenes={missing} scenes_dir={scenes_dir}")

    os.makedirs(args.save_path, exist_ok=True)
    # record experiment config + runtime args for reproducibility
    run_manifest = {
        "exp_config": args.exp_config,
        "model_path": args.model_path,
        "split_num": int(args.split_num),
        "split_id": int(args.split_id),
        "scenes_dir": args.scenes_dir,
        "cuda_visible_devices": args.cuda_visible_devices,
        "cuda_device": args.cuda_device,
        "habitat_gpu_id": args.habitat_gpu_id,
        "model_gpu_id": args.model_gpu_id,
        "online_cache_prune_mode": args.online_cache_prune_mode,
        "reuse_action_horizon": bool(args.reuse_action_horizon),
    }
    with open(Path(args.save_path) / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)

    # run once on train split shard with original evaluate_agent signature
    evaluate_agent(
        config=config,
        model_path=args.model_path,
        dataset_split=dataset,
        save_path=args.save_path,
         cuda_device=int(args.model_gpu_id),
        device_map=args.device_map,
        load_4bit=bool(args.load_4bit),
        load_8bit=bool(args.load_8bit),
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        save_episode_video=True,
        save_clip_video=False,
        write_clip_jsonl=False,
        write_episode_jsonl=True,
        episode_jsonl_name="track_episodes.jsonl",
        save_debug_video=False,
        restart_env_every_episodes=int(args.restart_env_every_episodes),
        action_horizon=4,
        history_len=20,
        reuse_action_horizon=bool(args.reuse_action_horizon),
        online_cache_prune_mode=str(args.online_cache_prune_mode),
    )
    _materialize_train_layout(
        save_path=args.save_path,
        config_dirname=args.output_config_dirname,
        video_dirname=args.output_video_dirname,
        output_json_name=args.output_json_name,
    )

if __name__ == "__main__":
    main()
