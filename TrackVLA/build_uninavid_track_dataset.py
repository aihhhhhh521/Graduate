#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import json
import argparse


def jsonl_to_json(jsonl_path: str, output_json: str, dataset_root: str):
    samples = []
    missing = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # 校验视频是否存在
            video_rel = rec.get("video", "")
            video_abs = osp.join(dataset_root, video_rel)
            if not osp.exists(video_abs):
                missing += 1
                continue

            samples.append(rec)

    os.makedirs(osp.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Loaded {len(samples)} samples, skipped_missing_video={missing}")
    print(f"[INFO] Saved to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True, help="same as save_path used in rendering")
    parser.add_argument("--jsonl", type=str, default="track_episodes.jsonl")
    parser.add_argument("--output-json", type=str, default="track_dataset.json")
    args = parser.parse_args()

    jsonl_to_json(
        jsonl_path=osp.join(args.dataset_root, args.jsonl),
        output_json=osp.join(args.dataset_root, args.output_json),
        dataset_root=args.dataset_root,
    )
