# coding: utf-8
"""
Online per-step benchmark for Uni-NaVid (with optional SR/TR/CR logging).

This script focuses on *per-step* latency/Hz and token-budget stats, and can also
attach *task performance* metrics (SR/TR/CR) if you provide per-step environment
telemetry (distance/angle/collision or a tracking_success flag).

What this script can measure (per step):
- latency (ms/step) and Hz
- visual-token budget/structure from Uni-NaVid runtime stats
- (optional) full-step latency including model.generate (vision + LLM decode)
- (optional) action correctness against GT actions in train-format JSON
- (optional) SR/TR/CR (official definitions) when env telemetry is provided

Benchmark modes:
1) episode_encode: one-shot encode_images over the whole video (legacy, NOT online)
2) online_encode: feed frames one by one, time encode_images only (online visual+merge)
3) online_full : feed frames one by one, time model.generate per step (online end-to-end)

Outputs:
- <output_dir>/step_stats.jsonl   (one JSON per step; episode SR/TR/CR is filled in after episode ends)
- <output_dir>/summary.json       (aggregate p50/p90/mean for ms/step + Hz + (optional) SR/TR/CR)

Env-metric JSONL (optional):
If you pass --env_metrics_jsonl, it must contain one JSON object per line like:

{
  "episode_id": "xxx",
  "step_idx": 12,
  "dist_to_target_m": 2.4,
  "angle_to_target_deg": 18.0,
  "collision": false,
  "done": false,
  "done_reason": "none",
  "tracking_success": true
}

Only episode_id + step_idx are required keys; others are optional, but SR/TR/CR
needs at least:
- collision or done_reason
- (tracking_success) OR (dist_to_target_m + angle_to_target_deg)

Official SR/TR/CR (EVT-Bench definition):
- SR: episode success if at the end the agent faces the target and is 1–3m away.
- TR: proportion of steps with successful tracking (S/L).
- CR: proportion of episodes terminated due to collision.
(See TrackVLA appendix / EVT-Bench metric definitions.)

Notes on FPS:
- video_fps only controls how densely you sample frames from the recorded video.
  It does NOT change the model’s per-step compute, but it changes how many steps
  exist per second of video and therefore the workload to hit e.g. 5Hz online.
- For apples-to-apples token/latency comparison against training, keep 1 fps.
  For deployability, also try 2/5 fps and check if ms/step <= 1000/fps.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from uninavid.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import (
    NAVIGATION_IDENTIFIER,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    VIDEO_SPECIAL_TOKEN,
    VIDEO_SPECIAL_TOKEN_END,
    IMAGE_SPECIAL_TOKEN,
    IMAGE_SPECIAL_TOKEN_END,
    IMAGE_SEPARATOR,
)
from uninavid.conversation import conv_templates

# Avoid WANDB noise in benchmark runs
os.environ.setdefault("WANDB_MODE", "offline")

ALLOWED_ACTIONS = {"forward", "left", "right", "stop"}


def load_video_frames(video_path: str, target_fps: float = 1.0, max_frames: Optional[int] = None) -> List[Image.Image]:
    """Decode a video-like file into a list of PIL RGB frames."""
    # Case 1: .npy / .npz arrays (T, H, W, 3)
    if video_path.endswith(".npy") or video_path.endswith(".npz"):
        arr = np.load(video_path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            key0 = list(arr.files)[0]
            arr = arr[key0]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"Expected (T,H,W,3) array, got {arr.shape}")
        frames = [Image.fromarray(np.asarray(arr[i], dtype=np.uint8)) for i in range(arr.shape[0])]
        return frames[:max_frames] if max_frames else frames

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(native_fps) or native_fps <= 0:
        native_fps = target_fps

    stride = max(1, int(round(native_fps / float(target_fps))))

    frames: List[Image.Image] = []
    frame_idx = 0
    kept = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            kept += 1
            if max_frames is not None and kept >= int(max_frames):
                break
        frame_idx += 1
    cap.release()
    return frames


def extract_instruction(sample: Dict[str, Any]) -> str:
    """Get the human instruction text from a LLaVA-style sample."""
    convs = sample.get("conversations", None)
    if isinstance(convs, list):
        for t in convs:
            if isinstance(t, dict) and str(t.get("from", "")).lower() in {"human", "user"}:
                v = t.get("value", "")
                if isinstance(v, str) and v.strip():
                    return v
    for k in ["instruction", "prompt", "text"]:
        v = sample.get(k, "")
        if isinstance(v, str) and v.strip():
            return v
    return ""


def extract_gt_actions(sample: Dict[str, Any]) -> List[str]:
    """Parse GT actions from the 'gpt' turn if present (space-separated)."""
    convs = sample.get("conversations", None)
    if not isinstance(convs, list):
        return []
    for t in convs:
        if isinstance(t, dict) and str(t.get("from", "")).lower() in {"gpt", "assistant"}:
            v = t.get("value", "")
            if isinstance(v, str):
                toks = [x.strip().lower() for x in v.split() if x.strip()]
                return [x for x in toks if x in ALLOWED_ACTIONS]
    return []


def build_prompt_input_ids(tokenizer, instruction: str, conv_mode: str = "vicuna_v1") -> torch.Tensor:
    """
    Build LLM input_ids for navigation inference.

    We build a Vicuna-style conversation:
      USER: <instruction> (instruction should already contain <image> token)
      ASSISTANT: (to be generated)
    """
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Convert <image> placeholder to IMAGE_TOKEN_INDEX positions.
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

    # Wrap IMAGE_TOKEN_INDEX with Uni-NaVid navigation control tokens so that
    # tokenization matches training (imgsp_v1 behavior).
    img_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten().tolist()
    if len(img_pos) > 0:
        def tid(s: str) -> List[int]:
            return tokenizer(s, add_special_tokens=False).input_ids

        v_beg = tid(VIDEO_SPECIAL_TOKEN)
        v_end = tid(VIDEO_SPECIAL_TOKEN_END)
        i_beg = tid(IMAGE_SPECIAL_TOKEN)
        i_end = tid(IMAGE_SPECIAL_TOKEN_END)
        i_sep = tid(IMAGE_SEPARATOR)
        nav = tid(NAVIGATION_IDENTIFIER)

        new_ids: List[int] = []
        for tok in input_ids[0].tolist():
            if tok == IMAGE_TOKEN_INDEX:
                new_ids += v_beg + i_sep + [IMAGE_TOKEN_INDEX] + v_end + i_beg + i_end + nav
            else:
                new_ids.append(tok)
        input_ids = torch.tensor([new_ids], dtype=torch.long)

    return input_ids


def parse_pred_actions(text: str) -> List[str]:
    toks = [x.strip().lower() for x in text.replace("\n", " ").split() if x.strip()]
    return [x for x in toks if x in ALLOWED_ACTIONS]


def percentiles(arr: List[float], ps=(50, 90, 95)) -> Dict[str, float]:
    if not arr:
        return {f"p{p}": float("nan") for p in ps}
    s = sorted(arr)
    out = {}
    for p in ps:
        k = (len(s) - 1) * (p / 100.0)
        f = int(np.floor(k))
        c = int(np.ceil(k))
        if f == c:
            out[f"p{p}"] = float(s[f])
        else:
            out[f"p{p}"] = float(s[f] * (c - k) + s[c] * (k - f))
    return out


def load_env_metrics_jsonl(path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Load per-step env telemetry from jsonl.
    Returns dict[(episode_id, step_idx)] = record.
    """
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not path:
        return out
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                raise ValueError(f"Bad JSON at {path}:{ln}")
            epi = str(r.get("episode_id", ""))
            if epi == "":
                continue
            if "step_idx" not in r:
                continue
            try:
                st = int(r["step_idx"])
            except Exception:
                continue
            out[(epi, st)] = r
    return out


def compute_tracking_success(info: Dict[str, Any], tr_angle_max_deg: float, tr_dist_max_m: float) -> Optional[bool]:
    """
    Try to determine whether the target is successfully tracked at this step.
    Priority:
      1) info["tracking_success"] (bool)
      2) compute from (angle_to_target_deg, dist_to_target_m) if present
    """
    if "tracking_success" in info:
        v = info["tracking_success"]
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
    ang = info.get("angle_to_target_deg", None)
    dist = info.get("dist_to_target_m", None)
    if ang is None or dist is None:
        return None
    try:
        ang = float(ang)
        dist = float(dist)
    except Exception:
        return None
    if not np.isfinite(ang) or not np.isfinite(dist):
        return None
    return (abs(ang) <= float(tr_angle_max_deg)) and (dist <= float(tr_dist_max_m))


def compute_episode_success(last_info: Dict[str, Any], sr_angle_max_deg: float, sr_dist_min_m: float, sr_dist_max_m: float) -> Optional[bool]:
    """
    EVT-Bench SR: successful if by the end the agent remains oriented toward the target
    and maintains a safe distance of 1–3 meters.
    Priority:
      1) last_info["success"] (bool)
      2) compute from (angle_to_target_deg, dist_to_target_m)
    """
    if "success" in last_info:
        v = last_info["success"]
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)

    ang = last_info.get("angle_to_target_deg", None)
    dist = last_info.get("dist_to_target_m", None)
    if ang is None or dist is None:
        return None
    try:
        ang = float(ang)
        dist = float(dist)
    except Exception:
        return None
    if not np.isfinite(ang) or not np.isfinite(dist):
        return None
    return (abs(ang) <= float(sr_angle_max_deg)) and (float(sr_dist_min_m) <= dist <= float(sr_dist_max_m))


def is_collision_termination(info: Dict[str, Any]) -> Optional[bool]:
    """
    Determine if this episode terminated due to collision.
    Priority:
      1) info["done_reason"] == "collision"
      2) info["collision"] == True AND info["done"] == True
    """
    dr = str(info.get("done_reason", "")).lower()
    if dr in {"collision", "collide", "crash"}:
        return True
    if "collision" in info and "done" in info:
        try:
            return bool(info["collision"]) and bool(info["done"])
        except Exception:
            return None
    if "collision" in info:
        try:
            return bool(info["collision"])
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser("Uni-NaVid online per-step benchmark (train-format data)")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--video_folder", type=str, default="", help="Root folder for relative video paths.")
    parser.add_argument("--output_dir", type=str, default="exp_results/online_bench")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--benchmark_mode", type=str, default="online_full",
                        choices=["episode_encode", "online_encode", "online_full"])

    parser.add_argument("--video_fps", type=float, default=1.0, help="Sampling FPS for decoding the recorded video.")
    parser.add_argument("--max_frames_per_video", type=int, default=None)
    parser.add_argument("--max_steps_per_episode", type=int, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--only_nav_id", action="store_true",
                        help="If set, only evaluate samples whose id contains 'NAV_ID'.")

    # Online compression knobs (requires your patched model)
    parser.add_argument("--compress_type", type=str, default=None,
                        help="e.g. 'grid:2', 'grid:1', or your custom scheme. Set None to keep model default.")
    parser.add_argument("--online_length_threshold", type=int, default=None,
                        help="e.g. 64. Online historical token budget threshold.")
    parser.add_argument("--online_similarity_threshold", type=float, default=None,
                        help="e.g. 0.985. Similarity threshold for online token merging.")

    # Generation knobs (online_full)
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=2, help="Warmup steps per episode (not recorded).")

    # Env metrics attachment
    parser.add_argument("--env_metrics_jsonl", type=str, default="",
                        help="Optional jsonl of per-step telemetry to compute SR/TR/CR.")
    parser.add_argument("--sr_angle_max_deg", type=float, default=45.0)
    parser.add_argument("--sr_dist_min_m", type=float, default=1.0)
    parser.add_argument("--sr_dist_max_m", type=float, default=3.0)
    parser.add_argument("--tr_angle_max_deg", type=float, default=45.0)
    parser.add_argument("--tr_dist_max_m", type=float, default=7.5)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------
    # Load model
    # -------------------------
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, None, model_name)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Apply config knobs (if supported by your model code)
    if hasattr(model, "config"):
        if args.compress_type is not None:
            setattr(model.config, "compress_type", args.compress_type)
        if args.online_length_threshold is not None:
            setattr(model.config, "online_length_threshold", int(args.online_length_threshold))
        if args.online_similarity_threshold is not None:
            setattr(model.config, "online_similarity_threshold", float(args.online_similarity_threshold))
        setattr(model.config, "run_type", "eval")

    # Inner backbone holds caches
    backbone_inner = model.get_model()

    # -------------------------
    # Load episodes (train-format JSON)
    # -------------------------
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("Expected list of samples in JSON")

    episodes = []
    for sample in data:
        if not isinstance(sample, dict):
            continue
        epi_id = str(sample.get("id", ""))
        if args.only_nav_id and "NAV_ID" not in epi_id:
            continue

        # locate video relpath
        rel = sample.get("video_path_fpv") or sample.get("video") or sample.get("video_file") or sample.get("video_relpath") or ""
        rel = str(rel)
        if not rel:
            continue

        video_path = rel
        if args.video_folder and not os.path.isabs(video_path):
            video_path = os.path.join(args.video_folder, video_path)
        episodes.append({"id": epi_id, "video_path": video_path, "sample": sample})

    if args.max_episodes is not None:
        episodes = episodes[: int(args.max_episodes)]
    if len(episodes) == 0:
        raise RuntimeError("No episodes found. Check --data_path/--video_folder and filtering options.")

    # -------------------------
    # Load optional env telemetry
    # -------------------------
    env_metrics = load_env_metrics_jsonl(args.env_metrics_jsonl) if args.env_metrics_jsonl else {}

    # -------------------------
    # Output paths
    # -------------------------
    step_out_path = os.path.join(args.output_dir, "step_stats.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.json")

    ms_step_all: List[float] = []
    hz_all: List[float] = []
    ms_encode_all: List[float] = []
    ms_generate_all: List[float] = []

    # Aggregate SR/TR/CR across episodes (only if env metrics available)
    n_epi_with_env = 0
    n_epi_success = 0
    n_epi_collision = 0
    tr_steps_success = 0
    tr_steps_total = 0

    with open(step_out_path, "w", encoding="utf-8") as wf:
        for epi_idx, epi in enumerate(episodes):
            epi_id = epi["id"]
            video_path = epi["video_path"]
            sample = epi["sample"]

            print(f"\n[{epi_idx+1}/{len(episodes)}] id={epi_id}")
            print(f"  video={video_path}")

            try:
                frames = load_video_frames(video_path, target_fps=args.video_fps, max_frames=args.max_frames_per_video)
            except Exception as e:
                print(f"  [WARN] load_video_frames failed: {e}")
                continue
            if len(frames) == 0:
                print("  [WARN] empty frames, skip.")
                continue

            pixel_values = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(device=device, dtype=model.dtype)

            T = int(pixel_values.shape[0])
            if args.max_steps_per_episode is not None:
                T = min(T, int(args.max_steps_per_episode))

            # reset caches and runtime stats
            if hasattr(backbone_inner, "initialize_online_inference_nav_feat_cache"):
                backbone_inner.initialize_online_inference_nav_feat_cache()
            if hasattr(backbone_inner, "new_frames"):
                backbone_inner.new_frames = 0
            if hasattr(model, "reset_runtime_stats"):
                model.reset_runtime_stats()
            if hasattr(model, "update_prompt"):
                model.update_prompt([[NAVIGATION_IDENTIFIER]])

            instruction = extract_instruction(sample)
            if not instruction:
                instruction = f"{DEFAULT_IMAGE_TOKEN}\nPlease determine the next four actions."
            input_ids = build_prompt_input_ids(tokenizer, instruction, conv_mode=args.conv_mode).to(device)

            conv = conv_templates[args.conv_mode].copy()
            stop_str = conv.sep2 if conv.sep_style.name == "TWO" else conv.sep
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            gt_actions = extract_gt_actions(sample)

            # Determine how many valid steps we can score for action correctness
            # (if GT has length, we can only score steps < len(gt_actions))
            # We still benchmark all frames for latency.
            # For 4-action supervision, step t corresponds to GT slice t:t+4.
            gt_len = len(gt_actions)

            # --- warmup (not recorded)
            warmup = min(args.warmup_steps, max(0, T - 1))
            if warmup > 0:
                for t in range(warmup):
                    if hasattr(backbone_inner, "new_frames"):
                        backbone_inner.new_frames = 1
                    with torch.no_grad():
                        if args.benchmark_mode == "online_full":
                            _ = model.generate(
                                input_ids,
                                images=[pixel_values[t:t+1]],
                                prompts=[[NAVIGATION_IDENTIFIER]],
                                image_counts=[1],
                                long_video=False,
                                do_sample=False,
                                num_beams=args.num_beams,
                                max_new_tokens=args.max_new_tokens,
                                use_cache=True,
                                stopping_criteria=[stopping_criteria],
                            )
                        else:
                            _ = model.encode_images(
                                images=pixel_values[t:t+1],
                                prompts=[[NAVIGATION_IDENTIFIER]],
                                image_counts=[1],
                                long_video=False,
                            )

            # per-episode buffers so we can attach episode SR/TR/CR to each step record
            ep_records: List[Dict[str, Any]] = []

            # per-episode env counters (for SR/TR/CR)
            ep_has_env = False
            ep_tr_succ = 0
            ep_tr_total = 0
            ep_collision = False
            last_env_info: Optional[Dict[str, Any]] = None
            last_step_with_env: Optional[int] = None

            # --- measured steps
            for t in range(warmup, T):
                record: Dict[str, Any] = {"episode_id": epi_id, "step_idx": int(t), "video_path": video_path}

                t_encode_ms = None
                t_gen_ms = None
                t_total_ms = None

                if hasattr(backbone_inner, "new_frames"):
                    backbone_inner.new_frames = 1

                if args.benchmark_mode == "episode_encode":
                    # legacy: run once at the end (not per-step)
                    # We still record placeholders for consistency.
                    t_total_ms = float("nan")

                elif args.benchmark_mode == "online_encode":
                    if torch.cuda.is_available():
                        st = torch.cuda.Event(enable_timing=True)
                        ed = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        st.record()
                    with torch.no_grad():
                        _ = model.encode_images(
                            images=pixel_values[t:t+1],
                            prompts=[[NAVIGATION_IDENTIFIER]],
                            image_counts=[1],
                            long_video=False,
                        )
                    if torch.cuda.is_available():
                        ed.record()
                        torch.cuda.synchronize()
                        t_encode_ms = float(st.elapsed_time(ed))
                    else:
                        t_encode_ms = float("nan")

                    t_total_ms = t_encode_ms
                    record["encode_ms"] = t_encode_ms

                elif args.benchmark_mode == "online_full":
                    if torch.cuda.is_available():
                        st = torch.cuda.Event(enable_timing=True)
                        ed = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        st.record()
                    with torch.no_grad():
                        out_ids = model.generate(
                            input_ids,
                            images=[pixel_values[t:t+1]],
                            prompts=[[NAVIGATION_IDENTIFIER]],
                            image_counts=[1],
                            long_video=False,
                            do_sample=False,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                        )
                    if torch.cuda.is_available():
                        ed.record()
                        torch.cuda.synchronize()
                        t_gen_ms = float(st.elapsed_time(ed))
                    else:
                        t_gen_ms = float("nan")

                    t_total_ms = t_gen_ms
                    record["generate_ms"] = t_gen_ms

                    # decode predicted actions
                    try:
                        gen_text = tokenizer.decode(out_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
                    except Exception:
                        gen_text = ""
                    pred_actions = parse_pred_actions(gen_text)
                    record["pred_actions"] = pred_actions[:8]

                    # action correctness against GT (if GT available)
                    if gt_len > 0 and t < gt_len:
                        gt_next = gt_actions[t:t+4]
                        record["gt_actions_next4"] = gt_next
                        k4 = min(len(gt_next), len(pred_actions), 4)
                        if k4 > 0:
                            acc4 = sum(int(pred_actions[i] == gt_next[i]) for i in range(k4)) / float(k4)
                            record["acc_next4"] = float(acc4)
                            record["k_next4"] = int(k4)
                            record["exact_match_next4"] = bool(all(pred_actions[i] == gt_next[i] for i in range(k4)) and k4 == len(gt_next))
                        # 1-step correctness (next action)
                        if len(pred_actions) >= 1 and len(gt_next) >= 1:
                            record["acc_next1"] = float(pred_actions[0] == gt_next[0])

                # runtime stats (visual tokens)
                if hasattr(model, "get_runtime_stats"):
                    stats = model.get_runtime_stats()
                    steps = stats.get("vis_tokens_llm_steps", [])
                    struct = stats.get("vis_tokens_llm_structure", [])
                    if steps:
                        record["vis_tokens_step"] = int(steps[-1])
                    if struct:
                        record["vis_structure_step"] = struct[-1]

                # env metrics attachment (optional)
                info = env_metrics.get((epi_id, int(t)), None)
                if info is not None:
                    ep_has_env = True
                    last_env_info = info
                    last_step_with_env = int(t)

                    # copy a few raw fields if they exist
                    for k in ["dist_to_target_m", "angle_to_target_deg", "collision", "done", "done_reason", "tracking_success", "success"]:
                        if k in info:
                            record[k] = info[k]

                    # TR step: whether successfully tracking at this step
                    tr_ok = compute_tracking_success(info, args.tr_angle_max_deg, args.tr_dist_max_m)
                    if tr_ok is not None:
                        record["tr_step_success"] = int(bool(tr_ok))
                        ep_tr_succ += int(bool(tr_ok))
                        ep_tr_total += 1

                    # collision (episode-level)
                    col = is_collision_termination(info)
                    if col is True:
                        ep_collision = True

                # latency -> Hz
                if t_total_ms is not None and np.isfinite(t_total_ms) and t_total_ms > 0:
                    hz = 1000.0 / float(t_total_ms)
                else:
                    hz = float("nan")

                record["total_ms"] = t_total_ms
                record["hz"] = hz

                ep_records.append(record)

                if t_total_ms is not None and np.isfinite(t_total_ms):
                    ms_step_all.append(float(t_total_ms))
                    hz_all.append(float(hz))
                if t_encode_ms is not None and np.isfinite(t_encode_ms):
                    ms_encode_all.append(float(t_encode_ms))
                if t_gen_ms is not None and np.isfinite(t_gen_ms):
                    ms_generate_all.append(float(t_gen_ms))

            # -------------------------
            # Compute episode SR/TR/CR (only if env metrics present)
            # -------------------------
            ep_sr: Optional[float] = None
            ep_tr: Optional[float] = None
            ep_cr: Optional[float] = None

            if ep_has_env and last_env_info is not None:
                # TR: proportion of successful tracking steps among steps with telemetry
                if ep_tr_total > 0:
                    ep_tr = float(ep_tr_succ) / float(ep_tr_total)

                # CR: 1 if collision termination, else 0
                ep_cr = float(bool(ep_collision))

                # SR: success if end-of-episode matches SR condition
                sr_ok = compute_episode_success(last_env_info, args.sr_angle_max_deg, args.sr_dist_min_m, args.sr_dist_max_m)
                if sr_ok is not None:
                    ep_sr = float(bool(sr_ok))

                # Update global counters
                n_epi_with_env += 1
                if ep_sr is not None and ep_sr >= 0.5:
                    n_epi_success += 1
                if ep_cr is not None and ep_cr >= 0.5:
                    n_epi_collision += 1
                if ep_tr_total > 0:
                    tr_steps_success += ep_tr_succ
                    tr_steps_total += ep_tr_total

            # attach episode metrics to every step record
            for r in ep_records:
                if ep_sr is not None:
                    r["SR"] = ep_sr
                if ep_tr is not None:
                    r["TR"] = ep_tr
                if ep_cr is not None:
                    r["CR"] = ep_cr
                if ep_has_env and last_step_with_env is not None:
                    r["last_step_with_env"] = int(last_step_with_env)

                wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -------------------------
    # Summary
    # -------------------------
    summary: Dict[str, Any] = {
        "n_steps": len(ms_step_all),
        "ms_step_mean": float(np.mean(ms_step_all)) if ms_step_all else float("nan"),
        **{f"ms_step_{k}": v for k, v in percentiles(ms_step_all).items()},
        "hz_mean": float(np.mean(hz_all)) if hz_all else float("nan"),
        **{f"hz_{k}": v for k, v in percentiles(hz_all).items()},
    }
    if ms_encode_all:
        summary.update({
            "ms_encode_mean": float(np.mean(ms_encode_all)),
            **{f"ms_encode_{k}": v for k, v in percentiles(ms_encode_all).items()},
        })
    if ms_generate_all:
        summary.update({
            "ms_generate_mean": float(np.mean(ms_generate_all)),
            **{f"ms_generate_{k}": v for k, v in percentiles(ms_generate_all).items()},
        })

    if n_epi_with_env > 0:
        summary.update({
            "n_episodes_with_env": int(n_epi_with_env),
            "SR": float(n_epi_success) / float(n_epi_with_env),
            "CR": float(n_epi_collision) / float(n_epi_with_env),
            # weighted TR across all telemetry steps
            "TR": float(tr_steps_success) / float(tr_steps_total) if tr_steps_total > 0 else float("nan"),
            "TR_steps_total": int(tr_steps_total),
        })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] step stats: {step_out_path}")
    print(f"[Done] summary  : {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
