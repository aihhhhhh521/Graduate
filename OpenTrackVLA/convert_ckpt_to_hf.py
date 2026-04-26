#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from open_trackvla_hf import OpenTrackVLAConfig, OpenTrackVLAForWaypoint


def _resolve_ckpt_path(explicit: Optional[Path], ckpt_dir: Path) -> Path:
    if explicit is not None:
        if explicit.is_file():
            return explicit
        raise FileNotFoundError(f"Checkpoint path does not exist: {explicit}")
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory missing: {ckpt_dir}")
    cands = sorted(ckpt_dir.glob("model_epoch*.pt"))
    if not cands:
        raise FileNotFoundError(f"No model_epoch*.pt files inside {ckpt_dir}")
    return cands[-1]


def _load_config_blob(ckpt_blob: Dict[str, Any], fallback_path: Path) -> Dict[str, Any]:
    ck_cfg = ckpt_blob.get("config")
    if isinstance(ck_cfg, dict):
        return ck_cfg
    cfg_file = fallback_path if fallback_path.is_file() else fallback_path / "config.json"
    if cfg_file.is_file():
        with open(cfg_file, "r") as fh:
            return json.load(fh)
    raise RuntimeError("Could not find a config dictionary inside the checkpoint or ckpt_dir/config.json")


def _build_hf_config(overrides: Dict[str, Any], args: argparse.Namespace) -> OpenTrackVLAConfig:
    def _maybe(key: str, cast, default):
        if getattr(args, key) is not None:
            return cast(getattr(args, key))
        return cast(overrides.get(key, default))

    # Historical checkpoints may store `no_tanh_actions`; convert to `use_tanh_actions`
    no_tanh = overrides.get("no_tanh_actions", not overrides.get("use_tanh_actions", True))
    hf_cfg = OpenTrackVLAConfig(
        llm_name=args.llm_name or overrides.get("llm_name", "Qwen/Qwen3-0.6B"),
        freeze_llm=bool(overrides.get("freeze_llm", True)),
        n_waypoints=int(overrides.get("n_waypoints", 8)),
        max_time=int(overrides.get("max_time", 4096)),
        beta_nav=float(overrides.get("beta_nav", 10.0)),
        use_angle_tvi=bool(overrides.get("use_angle_tvi", False)),
        use_tanh_actions=not bool(no_tanh),
        alpha_xy=overrides.get("alpha_xy"),
        vision_feat_dim=_maybe("vision_feat_dim", int, 1536),
    )
    return hf_cfg


def convert_checkpoint(
    ckpt_path: Path,
    out_dir: Path,
    ckpt_dir: Path,
    overwrite: bool,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty. Pass --overwrite to replace its contents.")

    print(f"[convert] Loading checkpoint: {ckpt_path}")
    obj = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError("Checkpoint must be a dict containing 'model_state'")
    state = obj.get("model_state") or obj.get("model_state_dict")
    if state is None:
        raise KeyError("Checkpoint is missing 'model_state' / 'model_state_dict'")

    overrides = _load_config_blob(obj, ckpt_dir)
    hf_config = _build_hf_config(overrides, args)
    print(f"[convert] Instantiating HuggingFace wrapper with llm={hf_config.llm_name}")
    hf_model = OpenTrackVLAForWaypoint(hf_config)
    missing, unexpected = hf_model.model.load_state_dict(state, strict=False)
    if missing:
        print(f"[convert][warn] Missing keys: {missing}")
    if unexpected:
        print(f"[convert][warn] Unexpected keys: {unexpected}")

    print(f"[convert] Saving HuggingFace checkpoint under {out_dir}")
    hf_model.save_pretrained(str(out_dir))

    meta = {
        "source_checkpoint": str(ckpt_path),
        "converted_with": os.path.basename(__file__),
        "config_overrides": overrides,
        "hf_config": hf_config.to_dict(),
        "epoch": obj.get("epoch"),
        "step": obj.get("step"),
    }
    with open(out_dir / "checkpoint_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print("[convert] Done.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert training checkpoints to HuggingFace format.")
    ap.add_argument("--ckpt-path", type=str, default=None, help="Explicit path to model_epoch*.pt")
    ap.add_argument("--ckpt-dir", type=str, default="ckpts", help="Directory to search when --ckpt-path is omitted")
    ap.add_argument("--out-dir", type=str, default="open_trackvla_hf", help="Destination HuggingFace directory")
    ap.add_argument("--vision-feat-dim", type=int, default=None, help="Override vision feature dim")
    ap.add_argument("--llm-name", type=str, default=None, help="Override llm_name stored in the checkpoint config")
    ap.add_argument("--overwrite", action="store_true", help="Allow replacing existing files in --out-dir")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve() if args.ckpt_path else None
    resolved_ckpt = _resolve_ckpt_path(ckpt_path, ckpt_dir)
    out_dir = Path(args.out_dir).expanduser().resolve()
    convert_checkpoint(resolved_ckpt, out_dir, ckpt_dir, args.overwrite, args)


if __name__ == "__main__":
    main()

