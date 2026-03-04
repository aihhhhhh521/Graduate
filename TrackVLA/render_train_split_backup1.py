#!/usr/bin/env python3
"""Render / collect a sharded split of episodes.

Key additions:
- --cuda-device: select which physical GPU to use (via CUDA_VISIBLE_DEVICES)
- Options are passed through to agent_uninavid.evaluate_agent

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
                  help="Physical GPU id to use. Sets CUDA_VISIBLE_DEVICES to this single id BEFORE imports.")
    p.add_argument("--habitat-gpu-id", type=int, default=0,
                  help="Habitat-Sim gpu_device_id inside *visible* devices (usually 0).")

    # Model memory controls
    p.add_argument("--device-map", default=None, help="HF device_map (default: auto)")
    p.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes)")
    p.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    p.add_argument("--max-new-tokens", type=int, default=32, help="Max generation length for action decoding")
    p.add_argument("--do-sample", action="store_true", help="Enable sampling for generation (default off)")

    return p.parse_args()


def _resolve_scene_path(scene_id: str, scenes_dir: str) -> str:
    """Robustly resolve episode.scene_id against scenes_dir.

    Handles common cases:
    - scene_id is already an absolute path
    - scene_id is relative like 'hm3d/train/.../scene.basis.glb'
    - scene_id is mistakenly prefixed with 'data/scene_datasets/...'
    """
    if os.path.isabs(scene_id):
        return scene_id

    # Normalize windows slashes
    sid = scene_id.replace('\\', '/')

    # If scene_id already contains the canonical prefix, strip it
    prefix = "data/scene_datasets/"
    if sid.startswith(prefix):
        sid = sid[len(prefix):]

    return os.path.join(scenes_dir, sid)


def main() -> None:
    args = parse_args()

    # Must be set BEFORE torch/habitat import to take effect
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    # Optional hint for other libs (harmless if unused)
    os.environ.setdefault("TRACKVLA_CUDA_DEVICE", "0")  # relative id within visible devices

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
        # Common habitat-lab path
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.habitat_gpu_id
    except Exception:
        pass

    # Build dataset
    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)

    # Shard episodes
    if args.split_num <= 0:
        raise ValueError("--split-num must be > 0")
    if not (0 <= args.split_id < args.split_num):
        raise ValueError("--split-id must satisfy 0 <= split-id < split-num")

    episodes = list(dataset.episodes)
    episodes = episodes[args.split_id :: args.split_num]
    dataset.episodes = episodes

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

    # Pass model options to agent
    evaluate_agent(
        config=config,
        model_path=args.model_path,
        dataset_split=dataset,
        save_path=args.save_path,
        cuda_device=0,  # relative id within visible devices
        device_map=args.device_map,
        load_4bit=(True if args.load_4bit else None),
        load_8bit=(True if args.load_8bit else None),
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
    )


if __name__ == "__main__":
    main()
