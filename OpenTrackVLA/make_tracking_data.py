#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import subprocess
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class EpisodePaths:
    seed_dir: Path
    run_dir: Path
    stem: str  # without suffixes, e.g., "2" for 2.mp4 / 2_info.json
    mp4: Optional[Path]
    info_json: Path


def find_ffmpeg_executable() -> Optional[str]:
    """Return path to ffmpeg if available, else None."""
    return shutil.which("ffmpeg")


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def list_sorted_images(directory: Path) -> List[Path]:
    image_paths = [p for p in directory.glob("*.jpg")]
    image_paths.sort(key=lambda p: natural_sort_key(p.name))
    return image_paths


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def extract_frames_ffmpeg(ffmpeg_path: str, mp4_path: Path, out_dir: Path, quality: int = 2) -> List[Path]:
    """Extract all frames from mp4 using ffmpeg. Returns list of frame paths."""
    ensure_dir(out_dir)
    pattern = str(out_dir / "frame_%05d.jpg")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(mp4_path),
        "-q:v",
        str(quality),
        str(pattern),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return list_sorted_images(out_dir)


def sample_indices(num_items: int, target_count: int) -> List[int]:
    """Evenly sample indices from [0, num_items-1] to length target_count (<= num_items)."""
    if target_count <= 0 or num_items <= 0:
        return []
    if target_count >= num_items:
        return list(range(num_items))
    # Even spacing
    import numpy as np
    positions = np.linspace(0, num_items - 1, target_count)
    return [int(round(p)) for p in positions]


def pad_to_length(items: List, length: int) -> List:
    if length <= 0:
        return []
    if not items:
        # replicate a placeholder value (should not happen normally)
        return [items for _ in range(length)]  # type: ignore
    if len(items) >= length:
        return items[:length]
    padded = list(items)
    last = items[-1]
    while len(padded) < length:
        padded.append(last)
    return padded


def load_episode_info(info_json_path: Path) -> List[dict]:
    with open(info_json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {info_json_path}, found: {type(data)}")
    return data


def load_episode_status(status_json_path: Path) -> Optional[dict]:
    try:
        with open(status_json_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def build_actions_from_info(steps: List[dict]) -> List[List[float]]:
    """Extract raw base velocity as [forward, lateral, yaw_rate] per step."""
    actions: List[List[float]] = []
    for step in steps:
        base_velocity = step.get("base_velocity") or [0.0, 0.0, 0.0]
        if not isinstance(base_velocity, list) or len(base_velocity) < 3:
            base_velocity = [0.0, 0.0, 0.0]
        fwd, lat, yaw = float(base_velocity[0]), float(base_velocity[1]), float(base_velocity[2])
        actions.append([fwd, lat, yaw])
    return actions


def integrate_future_trajectory(
    actions: List[List[float]], start_index: int, horizon: int, dt: float = 1.0
) -> List[List[float]]:
    """Integrate future base velocities into a local trajectory starting at [0,0,0].

    Returns a list of [x, y, theta], including the origin as the first point.
    Uses actions[start_index : start_index + horizon + 1] (clamped to available actions).
    """
    x, y, theta = 0.0, 0.0, 0.0
    trajectory: List[List[float]] = []
    if horizon <= 0 or start_index >= len(actions):
        return trajectory
    end_index = min(len(actions) - 1, start_index + horizon)
    for k in range(start_index, end_index + 1):
        vx, vy, wz = actions[k]
        # Rotate body-frame linear velocities into the initial local frame by current heading
        dx_global = vx * math.cos(theta) - vy * math.sin(theta)
        dy_global = vx * math.sin(theta) + vy * math.cos(theta)
        # accumulate displacement in local frame
        x += dx_global * dt
        y += dy_global * dt
        theta += wz * dt
        trajectory.append([x, y, theta])
    return trajectory


def build_indicator_curve(
    actions: List[List[float]], start_index: int, horizon: int, dt: float
) -> List[List[float]]:
    """Return a short local curve anchored at origin as [[x,y], ...].

    Integrates actions from start_index for `horizon` steps (clamped),
    accumulating displacement in the local robot frame.
    """
    x, y, theta = 0.0, 0.0, 0.0
    curve_xy: List[List[float]] = []
    if horizon <= 0 or start_index >= len(actions):
        return curve_xy
    end_index = min(len(actions) - 1, start_index + horizon)
    # Always include origin as the first point for stability
    curve_xy.append([0.0, 0.0])
    for k in range(start_index, end_index + 1):
        vx, vy, wz = actions[k]
        dx_global = vx * math.cos(theta) - vy * math.sin(theta)
        dy_global = vx * math.sin(theta) + vy * math.cos(theta)
        x += dx_global * dt
        y += dy_global * dt
        theta += wz * dt
        curve_xy.append([x, y])
    return curve_xy


def make_episode_json(
    rel_frame_paths: List[str],
    actions_7d: List[List[float]],
    episode_id: str,
    instruction: str,
) -> dict:
    return {
        "episode_id": episode_id,
        "frames": rel_frame_paths,
        "actions": actions_7d,
        "instruction": instruction,
    }


def collect_episode_pairs(input_root: Path) -> List[EpisodePaths]:
    """Find (<k>.mp4, <k>_info.json) pairs under input_root."""
    episodes: List[EpisodePaths] = []

    for seed_dir in sorted(input_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        for run_dir in sorted(seed_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            # Find all *_info.json files inside run_dir
            for info_json in sorted(run_dir.glob("*_info.json")):
                stem = info_json.name[:-10]  # remove _info.json
                mp4 = (run_dir / f"{stem}.mp4")
                episodes.append(
                    EpisodePaths(
                        seed_dir=seed_dir,
                        run_dir=run_dir,
                        stem=stem,
                        mp4=mp4 if mp4.exists() else None,
                        info_json=info_json,
                    )
                )
    return episodes


def should_keep_episode(run_dir: Path, stem: str, only_success: bool) -> bool:
    if not only_success:
        return True
    status_path = run_dir / f"{stem}.json"
    status = load_episode_status(status_path)
    if not status:
        return False
    # keep if success flag indicates success
    success_val = status.get("success")
    finish = status.get("finish")
    status_str = str(status.get("status", "")).lower()
    return bool(finish) or (isinstance(success_val, (int, float)) and success_val > 0) or ("success" in status_str)


def main():
    parser = argparse.ArgumentParser(description="Make TrackVLA training data from mass_train outputs")
    parser.add_argument("--input_root", type=str, required=True, help="Path to mass_train root (e.g., exp_results/mass_train)")
    parser.add_argument("--output_root", type=str, required=True, help="Output root for training data (e.g., data/track)")
    parser.add_argument("--max_frames", type=int, default=32, help="[Deprecated] Ignored. All frames will be used.")
    parser.add_argument("--only_success", action="store_true", help="Keep only successful episodes if status json exists")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help=(
            "Instruction to use for all samples; if omitted, use per-episode status JSON "
            "when available, otherwise a sensible default."
        ),
    )
    parser.add_argument("--history", type=int, default=31, help="Number of previous frames for each sample window")
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Path to aggregated dataset JSON (default: <output_root>/dataset.json)",
    )
    parser.add_argument("--horizon", type=int, default=8, help="Future action horizon to integrate for trajectory")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step per action for integration")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    frames_root = output_root / "frames"
    ensure_dir(frames_root)
    jsonl_root = output_root / "jsonl"
    ensure_dir(jsonl_root)
    # Aggregated dataset will be written at the end
    out_file = Path(args.out_file).resolve() if args.out_file else (output_root / "dataset.json")
    ensure_dir(out_file.parent)

    ffmpeg_path = find_ffmpeg_executable()
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg or add it to PATH.")

    episodes = collect_episode_pairs(input_root)
    total = len(episodes)
    kept = 0
    num_samples = 0
    jsonl_files_written = 0
    episodes_no_samples = 0
    skipped_no_video = 0
    skipped_no_frames = 0
    all_samples: List[dict] = []

    for ep in episodes:
        kept += 1
        if not should_keep_episode(ep.run_dir, ep.stem, args.only_success):
            continue

        if ep.mp4 is None or not ep.mp4.exists():
            skipped_no_video += 1
            continue

        try:
            steps = load_episode_info(ep.info_json)
        except Exception as e:
            print(f"[WARN] Failed to load info for {ep.info_json}: {e}")
            continue

        # Determine instruction: if CLI provides one, always use it; otherwise try per-episode JSON, else fallback
        instruction_text = args.instruction.strip() if isinstance(args.instruction, str) and args.instruction.strip() else None
        status_path = ep.run_dir / f"{ep.stem}.json"
        if instruction_text is None:
            status = load_episode_status(status_path)
            if status:
                instr_candidate = status.get("instruction")
                if isinstance(instr_candidate, str) and instr_candidate.strip():
                    instruction_text = instr_candidate.strip()
        if instruction_text is None:
            instruction_text = "Follow the target person without collision."

        # Paths for frames extraction
        rel_frames_dir = Path(ep.seed_dir.name) / ep.run_dir.name / ep.stem
        abs_frames_dir = frames_root / rel_frames_dir
        try:
            frame_paths = extract_frames_ffmpeg(ffmpeg_path, ep.mp4, abs_frames_dir)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] ffmpeg failed for {ep.mp4}: {e}")
            continue

        if not frame_paths:
            skipped_no_frames += 1
            continue

        # Use ALL frames; align actions length to number of frames
        desired_len = len(frame_paths)
        if desired_len == 0:
            skipped_no_frames += 1
            continue

        actions_full = build_actions_from_info(steps)
        if len(actions_full) >= desired_len:
            actions = actions_full[:desired_len]
        else:
            actions = pad_to_length(actions_full, desired_len)

        # Build relative paths for JSON for all frames (no skipping)
        rel_frame_paths = [str((Path("frames") / rel_frames_dir / p.name).as_posix()) for p in frame_paths]

        # Build sliding-window samples
        history = max(0, int(args.history))
        episode_samples: List[dict] = []
        if len(rel_frame_paths) > 0:
            for j in range(0, len(rel_frame_paths)):
                if history > 0:
                    start_idx = max(0, j - history)
                    images_window = rel_frame_paths[start_idx:j]
                else:
                    images_window = []
                current_frame = rel_frame_paths[j]
                # Compute future trajectory from j-th to j+horizon-th action (inclusive)
                horizon = int(args.horizon)
                dt = float(args.dt)
                # Require that we have full horizon in the ORIGINAL steps (no padding)
                if j + horizon > len(actions_full) - 1:
                    continue
                # Integrate and slice from original actions
                traj = integrate_future_trajectory(actions_full, start_index=j, horizon=horizon, dt=dt)
                # Build indicator curve (x,y only), same horizon and dt
                #indicator_xy = build_indicator_curve(actions_full, start_index=j, horizon=horizon, dt=dt)
                # Include the corresponding future actions in the sample (exact horizon+1 length)
                end_index = j + horizon
                future_actions = actions_full[j : end_index + 1]
                step_info = steps[j] if j < len(steps) else {}
                collision_flag = bool(step_info.get("collision", False))
                target_distance = step_info.get("target_distance", step_info.get("dis_to_human", 0.0))
                sample = {
                    "images": images_window,
                    "current": current_frame,
                    "instruction": instruction_text,
                    "trajectory": traj,
                    "actions": future_actions,
                    "collision": collision_flag,
                    "target_distance": float(target_distance) if target_distance is not None else 0.0,
                }
                episode_samples.append(sample)

        # Write per-episode JSONL and update aggregates
        if episode_samples:
            # Mirror input structure under jsonl root: <seed>/<run>/<stem>.jsonl
            rel_jsonl_dir = Path(ep.seed_dir.name) / ep.run_dir.name
            abs_jsonl_dir = jsonl_root / rel_jsonl_dir
            ensure_dir(abs_jsonl_dir)
            jsonl_path = abs_jsonl_dir / f"{ep.stem}.jsonl"
            with open(jsonl_path, "w") as f:
                for s in episode_samples:
                    f.write(json.dumps(s) + "\n")
            jsonl_files_written += 1
            all_samples.extend(episode_samples)
            num_samples += len(episode_samples)
        else:
            episodes_no_samples += 1

    # Write aggregated dataset JSON (if any samples)
    if all_samples:
        with open(out_file, "w") as f:
            json.dump(all_samples, f)

    print(f"Found episodes: {total}")
    print(f"Written samples: {num_samples}")
    print(f"Per-episode JSONL files written: {jsonl_files_written}")
    if episodes_no_samples:
        print(f"Episodes with no samples: {episodes_no_samples}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Skipped (no frames): {skipped_no_frames}")
    if all_samples:
        print(f"Aggregated dataset file: {out_file}")
    print(f"Output dataset root: {output_root}")


if __name__ == "__main__":
    main()

