#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from cache_gridpool import VisionFeatureCacher, VisionCacheConfig, adapt_siglip_grid, grid_pool_tokens


def _list_image_files(dir_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _detect_layout_and_collect(data_root: Path) -> Tuple[List[List[Path]], List[List[str]], str]:
    """Return (files_by_view, relpaths_by_view, layout_tag).

    - files_by_view: [V][T] absolute Paths
    - relpaths_by_view: [V][T] relative path parts (to mirror under cache)
    - layout_tag: one of {"frames", "views", "flat"}
    """
    root = data_root
    frames_dir = root / "frames"
    if frames_dir.exists() and frames_dir.is_dir():
        # Choose seed
        seed_env = os.getenv("TRACKVLA_SEED", "").strip()
        seed_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
        if not seed_dirs:
            # Treat each leaf directory under frames/ as an independent view (video clips, etc.)
            files_by_view = []
            rels_by_view = []

            # Include images directly under frames_dir if present
            root_imgs = _list_image_files(frames_dir)
            if root_imgs:
                files_by_view.append(root_imgs)
                rels_by_view.append([str(Path("frames") / p.name) for p in root_imgs])

            # Walk subdirectories to find image sets
            subdirs = sorted([p for p in frames_dir.rglob('*') if p.is_dir()])
            for subdir in subdirs:
                imgs = _list_image_files(subdir)
                if not imgs:
                    continue
                rel_prefix = Path("frames") / subdir.relative_to(frames_dir)
                files_by_view.append(imgs)
                rels_by_view.append([str(rel_prefix / p.name) for p in imgs])

            if not files_by_view:
                raise RuntimeError(f"No frame images found under {frames_dir}")
            return files_by_view, rels_by_view, "frames"

        if seed_env:
            if seed_env.isdigit():
                seed_choice = f"seed_{seed_env}"
            else:
                seed_choice = seed_env
            seed_dirs = [p for p in seed_dirs if p.name == seed_choice]
            if not seed_dirs:
                raise RuntimeError(f"Requested seed not found: {seed_choice}")

        # Choose scene
        scene_env = os.getenv("TRACKVLA_SCENE", "").strip()
        scenes = []
        for seed_path in seed_dirs:
            for s in sorted([p for p in seed_path.iterdir() if p.is_dir()]):
                if (not scene_env) or (s.name == scene_env):
                    scenes.append((seed_path.name, s))
        if not scenes:
            raise RuntimeError(f"No scene directories found under selected seeds of {frames_dir}")

        # Each camera directory is a view
        files_by_view: List[List[Path]] = []
        rels_by_view: List[List[str]] = []
        for seed_name, scene_path in scenes:
            cam_dirs = sorted([p for p in scene_path.iterdir() if p.is_dir()])
            if not cam_dirs:
                # Fallback: images directly under scene
                imgs = _list_image_files(scene_path)
                if imgs:
                    files_by_view.append(imgs)
                    rels_by_view.append([str(Path("frames")/seed_name/scene_path.name/p.name) for p in imgs])
                continue
            for cam_dir in cam_dirs:
                imgs = _list_image_files(cam_dir)
                if not imgs:
                    continue
                files_by_view.append(imgs)
                rels_by_view.append([str(Path("frames")/seed_name/scene_path.name/cam_dir.name/p.name) for p in imgs])

        if not files_by_view:
            raise RuntimeError("No images found in frames layout")
        return files_by_view, rels_by_view, "frames"

    # Layout 2: <root>/<view>/*
    view_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if view_dirs:
        files_by_view = []
        rels_by_view = []
        for vd in view_dirs:
            imgs = _list_image_files(vd)
            if not imgs:
                continue
            files_by_view.append(imgs)
            rels_by_view.append([str(Path(vd.name)/p.name) for p in imgs])
        if not files_by_view:
            raise RuntimeError(f"No images found under any subdirectory of {root}")
        return files_by_view, rels_by_view, "views"

    # Layout 3: <root>/*
    imgs = _list_image_files(root)
    if not imgs:
        raise RuntimeError(f"No images found in {root}")
    return [imgs], [[p.name for p in imgs]], "flat"


@torch.inference_mode()
def _encode_single(pil: Image.Image, enc: VisionFeatureCacher) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reproduce train_planner encoder contract
    tok_dino, Hp, Wp = enc._encode_dino([pil])
    tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))
    Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)       # (1, P, C)
    Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)
    Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)
    return Vcoarse[0].cpu().float(), Vfine[0].cpu().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True, help='Dataset root to scan for frames')
    ap.add_argument('--cache_root', type=str, default=None, help='Where to mirror cache; defaults to <data_root>/vision_cache')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--image_size', type=int, default=384)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    cache_root = Path(args.cache_root).resolve() if args.cache_root else (data_root / 'vision_cache')
    os.makedirs(cache_root, exist_ok=True)

    files_by_view, rels_by_view, _layout = _detect_layout_and_collect(data_root)

    # Initialize encoder (CPU when multi-worker; here single-process heuristic)
    use_cuda = torch.cuda.is_available()
    enc = VisionFeatureCacher(VisionCacheConfig(image_size=args.image_size, batch_size=args.batch_size, device=('cuda' if use_cuda else 'cpu')))
    enc.eval()

    # Iterate views and frames; skip existing token files
    total = 0
    done = 0
    for v_idx in range(len(files_by_view)):
        files = files_by_view[v_idx]
        rels = rels_by_view[v_idx]
        for t_idx in range(len(files)):
            abs_img = files[t_idx]
            rel_path = Path(rels[t_idx])
            token_dir = cache_root / rel_path.parent
            base = rel_path.stem
            vf_path = token_dir / f"{base}_vfine.pt"
            vc_path = token_dir / f"{base}_vcoarse.pt"
            total += 1
            vf_exists = vf_path.exists()
            vc_exists = vc_path.exists()
            if vf_exists and vc_exists:
                continue
            token_dir.mkdir(parents=True, exist_ok=True)
            try:
                # Fast path: if Vfine exists but Vcoarse is missing, derive Vcoarse by pooling
                # 8x8 -> 2x2 without re-encoding heavy towers.
                if vf_exists and (not vc_exists):
                    try:
                        vf = torch.load(str(vf_path), map_location='cpu')  # (64, C)
                        vf = vf.float() if vf.dtype != torch.float32 else vf
                        vc = grid_pool_tokens(vf.unsqueeze(0), 8, 8, out_tokens=4)[0].cpu()  # (4, C)
                        try:
                            torch.save(vc.half(), str(vc_path))
                        except Exception:
                            torch.save(vc, str(vc_path))
                        done += 1
                        print(total, done)
                        continue
                    except Exception as e:
                        # Fall back to full re-encode on any failure
                        pass

                pil = Image.open(str(abs_img)).convert('RGB')
                vc, vf = _encode_single(pil, enc)
                if not vf_exists:
                    try:
                        torch.save(vf.half(), str(vf_path))
                    except Exception:
                        torch.save(vf, str(vf_path))
                if not vc_exists:
                    try:
                        torch.save(vc.half(), str(vc_path))
                    except Exception:
                        torch.save(vc, str(vc_path))
                done += 1
            except Exception as e:
                print(f"[warn] failed on {abs_img}: {e}")
            print (total, done)

    print(f"Completed precache: generated {done} / {total} frame token pairs under {cache_root}")


if __name__ == '__main__':
    main()


