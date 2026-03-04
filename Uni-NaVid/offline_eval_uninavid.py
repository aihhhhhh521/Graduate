
# coding: utf-8
"""
Offline visual-token evaluation script for Uni-NaVid.

This version is aligned with the *training* data format:
- `--data_path` points to the same JSON file you pass to training (e.g. track dataset).
- Each sample in that JSON is expected to contain at least:
    - "id": unique id string (navigation samples usually contain "NAV_ID")
    - one of: "video_path_fpv" / "video" / "video_file" / "video_relpath"
      which is a path to video or pre-extracted frames, usually *relative*
      to `--video_folder` (or absolute if `--video_folder` is empty).

For each navigation episode, we:
1. Decode the video into a frame sequence (1 FPS by default, configurable).
   - Supports:
     * normal video files readable by OpenCV (.mp4, .avi, ...)
     * .npy/.npz files containing (T, H, W, 3) pre-extracted frames
2. Run ONLY the vision encoder + historical frame compression module
   (`encode_images`), without decoding text, so the token statistics
   correspond exactly to the visual tokens handed to the LLM.
3. Read the runtime visual-token statistics from the backbone model
   (which must already be instrumented with `reset_runtime_stats`,
   `get_runtime_stats`, etc., as in your modified `uninavid_arch.py`).
4. Aggregate:
   - total visual tokens per episode
   - per-step token list
   - bucket statistics (<=64 / 64–320 / >320 visual tokens)
   - global averages across all evaluated episodes.

Example usage:

CUDA_VISIBLE_DEVICES=0 \\
python offline_eval_uninavid_train.py \\
  --model_path /path/to/uninavid-7b-full-224-video-fps-1-grid-2 \\
  --data_path /path/to/open_uninavid_track_train.json \\
  --video_folder /path/to/video/root \\
  --output_dir ./offline_eval_stats \\
  --max_episodes 100 \\
  --video_fps 1

You can also select the device via `--device` (e.g. `cuda:0`).
"""

import os
import json
import argparse
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import NAVIGATION_IDENTIFIER, DEFAULT_IMAGE_TOKEN


from typing import List, Dict, Any

def format_llm_token_structure(
    episode_index: int,
    episode_id: str,
    vis_structure: List[Dict[str, Any]],
    semantic_tokens: int,
    total_visual_tokens: int,
    total_llm_tokens: int,
) -> List[str]:
    """Build a human-readable token-structure summary for one episode.

    The layout roughly follows a tree style like:

        ================================
        Episode 0 - id: ...
        Total LLM tokens: ...
        <|vision_bos|>
        ├─ VIDEO_HIST × 4
        ├─ VIDEO_HIST × 4
        ├─ VIDEO_NAV × 64
        <|vision_eos|>
        <|text_semantic|>
        ├─ TEXT × 128
        <|text_eos|>

    `vis_structure` is a list of per-step dicts recorded by the backbone:
        {"history_blocks": [len_0, len_1, ...],
         "nav_tokens": int,
         "total_visual_tokens": int}
    In the current offline eval, there is usually a single step per episode,
    but we keep the loop generic.
    """
    lines: List[str] = []
    sep_line = "=" * 40
    lines.append(sep_line)
    lines.append(f"Episode {episode_index} - id: {episode_id}")
    lines.append(
        f"Total LLM tokens: {int(total_llm_tokens)} (visual={int(total_visual_tokens)}, semantic={int(semantic_tokens)})"
    )
    lines.append("<|vision_bos|>")

    if not vis_structure:
        lines.append("  (no visual structure stats available)")
    else:
        for step_idx, step in enumerate(vis_structure):
            blocks = step.get("history_blocks", []) or []
            nav_tokens = int(step.get("nav_tokens", 0))
            if len(vis_structure) > 1:
                lines.append(f"  # step {step_idx}")
            for b in blocks:
                try:
                    b_int = int(b)
                except Exception:
                    continue
                lines.append(f"├─ VIDEO_HIST × {b_int}")
            if nav_tokens > 0:
                lines.append(f"├─ VIDEO_NAV × {nav_tokens}")

    lines.append("<|vision_eos|>")
    lines.append("<|text_semantic|>")
    lines.append(f"├─ TEXT × {int(semantic_tokens)}")
    lines.append("<|text_eos|>")
    return lines

def load_video_frames(
    video_path: str,
    target_fps: float = 1.0,
    max_frames: int = None,
) -> List[Image.Image]:
    """
    Decode a video-like file into a list of PIL RGB frames.

    Supported formats:
      - standard video files readable by OpenCV (.mp4, .avi, ...)
      - .npy / .npz files containing an array of shape (T, H, W, 3)

    For real videos we sample approximately at `target_fps`.
    For .npy / .npz we assume frames are already sampled, so we only
    optionally truncate to `max_frames`.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()

    # Case 1: pre-extracted frames stored as (T, H, W, 3) in .npy / .npz
    if ext in [".npy", ".npz"]:
        data = np.load(video_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            # If multiple arrays are stored, take the first one by default.
            first_key = list(data.files)[0]
            data = data[first_key]

        if data.ndim != 4 or data.shape[-1] != 3:
            raise ValueError(
                f"Unsupported array shape in {video_path}: expected (T, H, W, 3), got {data.shape}"
            )

        frames: List[Image.Image] = []
        num_frames = data.shape[0]
        limit = num_frames if max_frames is None else min(num_frames, max_frames)

        for i in range(limit):
            frame = data[i]
            # Assume the array is in RGB order; if your data is BGR,
            # you can swap channels here.
            frame_uint8 = np.asarray(frame, dtype=np.uint8)
            frames.append(Image.fromarray(frame_uint8))

        return frames

    # Case 2: standard video file, use OpenCV + FPS-based stride sampling.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(native_fps) or native_fps <= 0:
        native_fps = target_fps

    # How many native frames we skip between two kept frames.
    stride = max(int(round(native_fps / max(target_fps, 1e-6))), 1)

    frames: List[Image.Image] = []
    frame_idx = 0
    kept = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


def episode_iter_from_train_json(
    data_path: str,
    video_folder: str,
) -> List[Dict[str, Any]]:
    """
    Load training-style JSON and return a list of episode descriptors.

    Expected JSON structure (same as training):
        [
          {
            "id": "...NAV_ID...",
            "video_path_fpv" or "video" or "video_file" or "video_relpath": "...",
            ...
          },
          ...
        ]

    We:
      - keep only samples whose id contains "NAV_ID" (navigation episodes),
      - resolve the video path under `video_folder` if it is not empty.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "data" in obj:
        data_list = obj["data"]
    else:
        data_list = obj

    episodes: List[Dict[str, Any]] = []
    for sample in data_list:
        # Try several possible keys for the video-relative path,
        # following the patterns used in training code.
        vid_rel = (
            sample.get("video_path_fpv")
            or sample.get("video")
            or sample.get("video_file")
            or sample.get("video_relpath")
        )
        if vid_rel is None:
            continue

        epi_id = str(sample.get("id", ""))

        # Only keep navigation episodes by default.
        if "NAV_ID" not in epi_id:
            continue

        if video_folder:
            video_path = os.path.join(video_folder, vid_rel)
        else:
            video_path = vid_rel

        episodes.append(
            {
                "id": epi_id,
                "video_path": video_path,
                "sample": sample,
            }
        )

    return episodes


def classify_buckets_from_steps(per_step: List[int]) -> Dict[str, int]:
    """
    Given a list of visual tokens per step, aggregate into three buckets:
        <=64, 64–320, >320.
    """
    buckets = {"<=64": 0, "64-320": 0, ">320": 0}
    for v in per_step:
        if v <= 64:
            buckets["<=64"] += 1
        elif v <= 320:
            buckets["64-320"] += 1
        else:
            buckets[">320"] += 1
    return buckets



def compute_semantic_tokens(sample: Dict[str, Any], tokenizer) -> int:
    """Approximate number of *text* (semantic) tokens for this episode.

    We use the raw "conversations" field (if present) from the training JSON:
    - Concatenate all turns' "value".
    - Remove DEFAULT_IMAGE_TOKEN placeholders, since those are replaced by visual tokens.
    - Tokenize with the same tokenizer used for training/inference.
    This is an approximation of how many text tokens the LLM processes per episode.
    """
    if sample is None or tokenizer is None:
        return 0

    text_pieces: List[str] = []

    # Most Uni-NaVid training JSONs follow the LLaVA-style "conversations" schema.
    convs = sample.get("conversations", None)
    if isinstance(convs, list) and len(convs) > 0:
        for turn in convs:
            if not isinstance(turn, dict):
                continue
            val = str(turn.get("value", ""))
            if not val:
                continue
            # Remove image placeholders: they correspond to visual tokens, not semantic tokens.
            if DEFAULT_IMAGE_TOKEN in val:
                val = val.replace(DEFAULT_IMAGE_TOKEN, "")
            text_pieces.append(val)
    else:
        # Fallback: try a few common text fields.
        for key in ["text", "input", "prompt"]:
            if key in sample and isinstance(sample[key], str):
                val = sample[key]
                if DEFAULT_IMAGE_TOKEN in val:
                    val = val.replace(DEFAULT_IMAGE_TOKEN, "")
                text_pieces.append(val)
                break

    if not text_pieces:
        return 0

    full_text = "\n".join(text_pieces).strip()
    if not full_text:
        return 0

    try:
        token_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
        return int(token_ids.shape[0])
    except Exception:
        # Be robust to any tokenizer issues.
        return 0


def estimate_llm_flops_per_step(n_tokens: int, config) -> float:
    """Rough FLOPs estimate for one LLM *forward* pass on `n_tokens` tokens.

    We use a standard transformer cost approximation:
        per-layer FLOPs ~= 4 * n * d^2          (QKV + output projections)
                         + 2 * n * d * d_ff     (FFN up & down projections)
                         + 2 * n^2 * d         (self-attention matmuls)
    where:
        - n = total sequence length (semantic + visual tokens)
        - d = hidden_size
        - d_ff = intermediate_size (default 4 * d if missing)

    Total FLOPs ~= num_layers * per-layer FLOPs.

    Note: this ignores embedding, layernorm, and vision-tower FLOPs,
    and only approximates the LLM core; it is meant for relative comparison.
    """
    if n_tokens <= 0 or config is None:
        return 0.0

    d_model = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
    n_layer = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layers", None)
    if d_model is None or n_layer is None:
        return 0.0

    d_ff = getattr(config, "intermediate_size", None)
    if d_ff is None:
        d_ff = 4 * d_model

    n = float(n_tokens)
    d = float(d_model)
    dff = float(d_ff)

    # QKV + output projections
    flops_proj = 4.0 * n * d * d
    # FFN
    flops_ffn = 2.0 * n * d * dff
    # Self-attention matmuls
    flops_attn = 2.0 * n * n * d

    flops_per_layer = flops_proj + flops_ffn + flops_attn
    total_flops = float(n_layer) * flops_per_layer
    return total_flops


def main():
    parser = argparse.ArgumentParser("Offline Uni-NaVid visual-token evaluation (train-format data)")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the finetuned Uni-NaVid model (same as training).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Training-style JSON file (same as `--data_path` in training).",
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        required=True,
        help=(
            "Root folder where sample['video_path_fpv']/['video'] is located "
            "(same as training `--video_folder`). "
            "If your JSON already stores absolute paths, you can pass an empty string here."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save aggregated statistics (JSON).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help=(
            "Device for inference, e.g. 'cuda:0' or 'cpu'. "
            "You can also control visible GPUs via CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional limit on the number of episodes to evaluate.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index in the episode list (for sharding / partial runs).",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=1.0,
        help="Target FPS when sampling frames from video (should match training `--video_fps`).",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=None,
        help="Optional cap on frames per video to avoid extremely long sequences.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model & processors
    # ------------------------------------------------------------------
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None,
        model_name,
    )

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    # Backbone (MetaModel) used for encode_images / runtime stats.
    # `model` is the full Uni-NaVid LLM (LlavaLlamaAttForCausalLM mixin).
    # For visual encoding + historical compression we need:
    #   - the outer model (`model`) to call `encode_images` (defined in UniNaVIDMetaForCausalLM)
    #   - the inner vision/navigation backbone (`model.get_model()`) to manage caches.
    backbone = model
    backbone_inner = model.get_model()


    # ------------------------------------------------------------------
    # 2. Build episode list from training JSON
    # ------------------------------------------------------------------
    episodes = episode_iter_from_train_json(args.data_path, args.video_folder)

    if args.start_index > 0:
        episodes = episodes[args.start_index :]

    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    if not episodes:
        print("No navigation episodes found in the provided data_path.")
        return

    print(f"Loaded {len(episodes)} navigation episodes from training-style JSON.")

    # ------------------------------------------------------------------
    # 3. Iterate episodes and collect statistics
    # ------------------------------------------------------------------
    all_stats = []
    global_token_sum = 0
    global_frame_sum = 0
    global_step_sum = 0
    global_buckets = {"<=64": 0, "64-320": 0, ">320": 0}
    global_sem_token_sum = 0
    global_llm_token_sum = 0
    global_flop_sum = 0.0

    for epi_idx, epi in enumerate(episodes):
        epi_id = epi.get("id", f"episode_{epi_idx}")
        video_path = epi["video_path"]

        print(f"\n[{epi_idx + 1}/{len(episodes)}] Episode id={epi_id}")
        print(f"  Video: {video_path}")

        try:
            frames = load_video_frames(
                video_path,
                target_fps=args.video_fps,
                max_frames=args.max_frames_per_video,
            )
        except Exception as e:
            print(f"  [WARN] Failed to load video frames: {e}")
            continue

        if len(frames) == 0:
            print("  [WARN] No frames decoded, skip.")
            continue

        # Convert frames to model pixel_values: (num_frames, 3, H, W)
        with torch.no_grad():
            pixel_values = image_processor.preprocess(
                frames,
                return_tensors="pt",
            )["pixel_values"]  # shape: (T, 3, H, W)

        pixel_values = pixel_values.to(device=device, dtype=model.dtype)
        num_frames = pixel_values.shape[0]

        # ------------------------------------------------------------------
        # Run ONLY visual encoder + historical frame compression.
        # We do NOT call model.generate, so we only measure visual tokens.
        # ------------------------------------------------------------------
        # 1) Reset caches & runtime statistics for this episode.
        # Initialize the navigation feature cache on the *inner* model.
        if hasattr(backbone_inner, "initialize_online_inference_nav_feat_cache"):
            backbone_inner.initialize_online_inference_nav_feat_cache()
            # Also reset the number of new frames.
            backbone_inner.new_frames = 0
        if hasattr(backbone, "reset_runtime_stats"):
            backbone.reset_runtime_stats()

        # 2) Mark this as navigation so that the NAV-specific branch is used.
        if hasattr(backbone, "update_prompt"):
            backbone.update_prompt([[NAVIGATION_IDENTIFIER]])

        # 3) Ensure run_type is "eval" so that the evaluation-time branch
        #    (with historical compression + stats collection) is active.
        if hasattr(backbone, "config"):
            setattr(backbone.config, "run_type", "eval")

        # 4) Tell online_process_tensor how many *new* frames are present.
        # Tell the inner backbone how many new frames this episode contains.
        if hasattr(backbone_inner, "new_frames"):
            backbone_inner.new_frames = int(num_frames)

        # 5) Call encode_images once with the whole sequence.
        with torch.no_grad():
            _ = backbone.encode_images(
                images=pixel_values,
                prompts=[[NAVIGATION_IDENTIFIER]],
                image_counts=[int(num_frames)],
                long_video=False,
            )

        # 6) Fetch runtime statistics.
        if hasattr(backbone, "get_runtime_stats"):
            stats = backbone.get_runtime_stats()
        else:
            stats = None

        if not stats:
            print("  [WARN] Backbone returned no runtime stats; "
                  "please ensure your uninavid_arch.py is instrumented.")
            continue

        # Our instrumented uninavid_arch.py uses these keys:
        #   - "vis_tokens_llm_steps": list[int], per-step visual tokens
        #   - "vis_tokens_llm_total": int, accumulated tokens over episode
        #   - "vis_tokens_llm_buckets": dict, bucketized counts
        per_step = stats.get("vis_tokens_llm_steps", [])
        total_tokens = stats.get("vis_tokens_llm_total", sum(per_step))
        buckets = stats.get("vis_tokens_llm_buckets", classify_buckets_from_steps(per_step))
        vis_structure = stats.get("vis_tokens_llm_structure", [])

        # Normalize bucket keys to a standard form.
        normalized_buckets = {"<=64": 0, "64-320": 0, ">320": 0}
        for k, v in buckets.items():
            kl = str(k).lower()
            v_int = int(v)
            if "<" in kl and "64" in kl:
                normalized_buckets["<=64"] += v_int
            elif "64" in kl and "320" in kl:
                normalized_buckets["64-320"] += v_int
            elif "320" in kl or ">" in kl:
                normalized_buckets[">320"] += v_int

        # 7) Compute semantic-token count and approximate LLM FLOPs for this episode.
        sample = epi.get("sample", None)
        sem_tokens = compute_semantic_tokens(sample, tokenizer)
        total_llm_tokens = int(total_tokens) + int(sem_tokens)
        approx_llm_flops = estimate_llm_flops_per_step(total_llm_tokens, model.config)

        # Build a human-readable token-structure summary for this episode.
        token_structure_lines = format_llm_token_structure(
            episode_index=epi_idx,
            episode_id=epi_id,
            vis_structure=vis_structure,
            semantic_tokens=sem_tokens,
            total_visual_tokens=total_tokens,
            total_llm_tokens=total_llm_tokens,
        )

        episode_stat = {
            "episode_index": epi_idx,
            "episode_id": epi_id,
            "video_path": video_path,
            "num_frames": int(num_frames),
            "visual_tokens_total": int(total_tokens),
            "visual_tokens_per_step": [int(x) for x in per_step],
            "visual_token_buckets": normalized_buckets,
            "semantic_tokens_total": int(sem_tokens),
            "total_llm_tokens": int(total_llm_tokens),
            "approx_llm_flops": float(approx_llm_flops),
            "llm_token_structure": token_structure_lines,
        }
        all_stats.append(episode_stat)

        # Accumulate global statistics.
        global_token_sum += int(total_tokens)
        global_frame_sum += int(num_frames)
        global_step_sum += len(per_step)
        global_sem_token_sum += int(sem_tokens)
        global_llm_token_sum += int(total_llm_tokens)
        global_flop_sum += float(approx_llm_flops)
        for k in global_buckets:
            global_buckets[k] += normalized_buckets.get(k, 0)

        # Also print the detailed token structure to console for inspection.
        print("Token structure for this episode:")
        for line in token_structure_lines:
            print(" ", line)
        print()

        # Print per-episode summary to console.
        avg_tokens_per_frame = total_tokens / max(num_frames, 1)
        print(f"  Frames: {num_frames}")
        print(f"  Total visual tokens after compression: {total_tokens}")
        print(f"  Avg tokens per frame: {avg_tokens_per_frame:.2f}")
        print(
            "  Buckets (<=64 / 64–320 / >320): "
            f"{normalized_buckets['<=64']} / "
            f"{normalized_buckets['64-320']} / "
            f"{normalized_buckets['>320']}"
        )

    if not all_stats:
        print("No valid episodes evaluated; nothing to save.")
        return

    # ------------------------------------------------------------------
    # 4. Global aggregated statistics
    # ------------------------------------------------------------------
    num_episodes = len(all_stats)
    avg_tokens_per_episode = global_token_sum / num_episodes
    avg_tokens_per_frame_global = global_token_sum / max(global_frame_sum, 1)
    avg_tokens_per_step_global = global_token_sum / max(global_step_sum, 1)
    avg_sem_tokens_per_episode = global_sem_token_sum / num_episodes
    avg_llm_tokens_per_episode = global_llm_token_sum / num_episodes
    avg_flops_per_episode = global_flop_sum / num_episodes
    avg_flops_per_step_global = global_flop_sum / max(global_step_sum, 1)

    summary = {
        "num_episodes": num_episodes,
        "total_visual_tokens": int(global_token_sum),
        "total_semantic_tokens": int(global_sem_token_sum),
        "total_llm_tokens": int(global_llm_token_sum),
        "total_llm_flops": float(global_flop_sum),
        "total_frames": int(global_frame_sum),
        "total_steps": int(global_step_sum),
        "avg_tokens_per_episode": float(avg_tokens_per_episode),
        "avg_tokens_per_frame": float(avg_tokens_per_frame_global),
        "avg_tokens_per_step": float(avg_tokens_per_step_global),
        "avg_semantic_tokens_per_episode": float(avg_sem_tokens_per_episode),
        "avg_total_llm_tokens_per_episode": float(avg_llm_tokens_per_episode),
        "avg_llm_flops_per_episode": float(avg_flops_per_episode),
        "avg_llm_flops_per_step": float(avg_flops_per_step_global),
        "global_buckets": global_buckets,
    }

    output = {
        "summary": summary,
        "episodes": all_stats,
    }

    out_path = os.path.join(args.output_dir, "offline_visual_token_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n===== Global summary =====")
    print(f"  Episodes evaluated: {num_episodes}")
    print(f"  Avg visual tokens / episode: {avg_tokens_per_episode:.2f}")
    print(f"  Avg semantic tokens / episode: {avg_sem_tokens_per_episode:.2f}")
    print(f"  Avg total LLM tokens / episode: {avg_llm_tokens_per_episode:.2f}")
    print(f"  Avg visual tokens / frame: {avg_tokens_per_frame_global:.2f}")
    print(f"  Avg visual tokens / step:  {avg_tokens_per_step_global:.2f}")
    print(f"  Avg LLM FLOPs / episode: {avg_flops_per_episode/1e9:.3f} GFLOPs")
    print(f"  Avg LLM FLOPs / step:    {avg_flops_per_step_global/1e9:.3f} GFLOPs")
    print(
        "  Global bucket counts (<=64 / 64–320 / >320): "
        f"{global_buckets['<=64']} / "
        f"{global_buckets['64-320']} / "
        f"{global_buckets['>320']}"
    )
    print(f"\nDetailed per-episode stats saved to: {out_path}")


if __name__ == "__main__":
    main()
