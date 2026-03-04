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
    p.add_argument("--restart-env-every-episodes", type=int, default=50,
                   help="Recreate TrackEnv every N episodes to prevent habitat-sim renderer GPU memory growth/leaks (default 50).")
    p.add_argument("--episode-only", action="store_true",
                   help="Collect ONE full episode video + ONE JSONL record per episode (recommended). Disables clip outputs.")
    p.add_argument("--clip-mode", action="store_true",
                   help="(Legacy) Collect many short clips + clip-level JSONL (sampled_500 style). Not recommended for your current goal.")

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
    from agent_uninavid import evaluate_agent

    config = get_config(args.exp_config)

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

    # Run collection
        # Run collection
    # Default: episode-only output (one mp4 + one jsonl per episode) unless user explicitly wants clip mode.
    # Default behavior: episode-only. Use --clip-mode to restore legacy per-inference clip outputs.
    episode_only = (not bool(args.clip_mode))

    evaluate_agent(
        config=config,
        model_path=args.model_path,
        dataset_split=dataset,
        save_path=args.save_path,
        cuda_device=int(args.model_gpu_id),
        device_map=args.device_map,
        load_4bit=(True if args.load_4bit else None),
        load_8bit=(True if args.load_8bit else None),
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        # Output format control
        save_episode_video=True,
        save_clip_video=False if episode_only else True,
        write_clip_jsonl=False if episode_only else True,
        write_episode_jsonl=True,
        episode_jsonl_name="track_episodes.jsonl",
        # Habitat-sim GPU memory/leak mitigation
        restart_env_every_episodes=int(args.restart_env_every_episodes),
    )


if __name__ == "__main__":
    main()
