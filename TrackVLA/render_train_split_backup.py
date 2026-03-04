#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render UniNaVid expert trajectories on TrackVLA train split,
and skip missing scenes (e.g., MP3D not prepared).
"""

import os
import os.path as osp
import random
import numpy as np

from evt_bench.default import get_config
from habitat.datasets import make_dataset
from agent_uninavid import evaluate_agent
from typing import Optional


def scene_exists(scenes_dir: str, scene_id: str) -> bool:
    if not scene_id:
        return False
        
    if scenes_dir:
        p = osp.join(scenes_dir, scene_id)
        if osp.exists(p):
            return True
            
    for ds in ("hm3d", "mp3d"):
        key = ds + "/"
        idx = scene_id.find(key)
        if idx != -1 and scenes_dir:
            tail = scene_id[idx:]
            p = osp.join(scenes_dir, tail)
            if osp.exists(p):
                return True
    
    return False


def render_train_split(
    exp_config: str,
    model_path: str,
    save_path: str,
    split_num: int = 1,
    split_id: int = 0,
    scenes_dir: Optional[str] = None,
):
    print(f"[INFO] Loading config: {exp_config}")
    config = get_config(exp_config)

    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)

    # 默认 scene 根目录
    if scenes_dir is None:
        scenes_dir = getattr(config.habitat.dataset, "scenes_dir", None) or "data/scene_datasets"
    scenes_dir = osp.abspath(scenes_dir)
    
    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    dataset_split = dataset.get_splits(split_num)[split_id]
    
    episodes = getattr(dataset_split, "episodes", None)
    assert episodes is not None, "dataset_split has no episodes field"
    print(f"[INFO] Loaded episodes: {len(episodes)}")
    print(f"[INFO] scenes_dir={scenes_dir}")
    print(f"[INFO] sample scene_id={episodes[0].scene_id if len(episodes)>0 else 'N/A'}")

    # 过滤掉缺失 scene 的 episode（你只有 HM3D 时，这一步能避免中断）
    kept = []
    dropped = 0
    for ep in episodes:
        if scene_exists(scenes_dir, ep.scene_id):
            kept.append(ep)
        else:
            dropped += 1
    dataset_split.episodes = kept  # habitat Dataset 支持 episodes 赋值
    print(f"[INFO] Filtered episodes: kept={len(kept)}, dropped={dropped}")
    
    if len(kept) == 0:
        raise RuntimeError(
            f"After filtering, 0 episodes kept, scene_dir={scenes_dir}."
            f"Example scene_id={episodes[0].scene_id if len(episodes)>0 else 'N/A'}"
        )

    os.makedirs(save_path, exist_ok=True)
    evaluate_agent(config, model_path, dataset_split, save_path)
    print("[INFO] Rendering complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--split-num", type=int, default=1)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--scenes-dir", type=str, default=None)
    args = parser.parse_args()

    render_train_split(
        exp_config=args.exp_config,
        model_path=args.model_path,
        save_path=args.save_path,
        split_num=args.split_num,
        split_id=args.split_id,
        scenes_dir=args.scenes_dir,
    )
