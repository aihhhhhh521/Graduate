#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import os, json, math, argparse, time, csv
from pathlib import Path
from contextlib import nullcontext
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModel
from cache_gridpool import VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens, adapt_siglip_grid


# ----------------------- utils -----------------------

# Silence tokenizers fork warnings in dataloader workers
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_tokens_file(path: str) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location='cpu')
    except Exception:
        # PyTorch 2.6 defaults to weights_only=True; retry allowing full unpickling for trusted caches
        try:
            obj = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise e
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for k in ("V", "Vfine", "Vcoarse", "tokens", "feat", "features"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]
                if t.dim() == 3 and t.size(0) == 1:
                    t = t[0]
                return t.float()
    raise ValueError(f"Unrecognized token file: {path}")


def integrate_actions_to_waypoints(actions: np.ndarray, n_waypoints: int, dt: float = 0.2) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 1: a = a[None, :]
    T, D = a.shape
    vx = a[:, 0].astype(np.float32)
    vy = a[:, 1].astype(np.float32) if D > 1 else np.zeros_like(vx)
    wz = a[:, 2].astype(np.float32) if D > 2 else np.zeros_like(vx)

    x = np.zeros(T, dtype=np.float32)
    y = np.zeros(T, dtype=np.float32)
    th = np.zeros(T, dtype=np.float32)

    for t in range(1, T):
        th[t] = th[t-1] + wz[t-1] * dt
        c, s = np.cos(th[t-1]), np.sin(th[t-1])
        x[t] = x[t-1] + (c * vx[t-1] - s * vy[t-1]) * dt
        y[t] = y[t-1] + (s * vx[t-1] + c * vy[t-1]) * dt

    traj = np.stack([x, y, th], axis=-1)
    if n_waypoints <= 1: return traj[-1:]
    idx = np.linspace(0, T-1, n_waypoints).round().astype(int)
    return traj[idx]


# Flexible loader for JSON/JSONL datasets and directories
def _read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            examples.append(json.loads(s))
    return examples


def load_examples_from_path(train_path: str) -> List[Dict[str, Any]]:
    """Load dataset examples from a JSON list, a JSONL file, or a directory (recursively).
    - If directory: recursively loads all .jsonl files in subfolders.
    - If file: supports .json (list of dicts) or .jsonl (one JSON per line).
    """
    p = Path(train_path)
    if p.is_dir():
        jsonl_files = sorted(p.rglob('*.jsonl'))
        if len(jsonl_files) == 0:
            raise FileNotFoundError(f"No .jsonl files found under directory: {train_path}")
        all_items: List[Dict[str, Any]] = []
        for fp in jsonl_files:
            all_items.extend(_read_jsonl_file(str(fp)))
        return all_items
    if p.is_file():
        if p.suffix.lower() == '.jsonl':
            return _read_jsonl_file(str(p))
        if p.suffix.lower() == '.json':
            with open(p, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list at top-level: {train_path}")
            return data
        raise ValueError(f"Unsupported file type: {train_path}")
    raise FileNotFoundError(f"Path does not exist: {train_path}")

def _cleanup_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make checkpoints usable across single-GPU and DDP.

    - If keys are prefixed with 'module.' (DDP/DataParallel), strip that prefix.
    - Otherwise return unchanged.
    """
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

# ----------------------- Sanity checks -----------------------

def _dataset_sanity_report(ds: 'JsonTrackingDataset', cfg: 'TrainConfig', max_items: int = 512):
    try:
        import numpy as _np
        n = min(max_items, len(ds))
        xs, ys, thetas = [], [], []
        mask_cov = []
        yaw_hist_present = 0
        yaw_curr_present = 0
        img_ok = 0
        for i in range(n):
            ex = ds.get_example(i)
            # compute target waypoints without triggering token encoding
            if 'waypoints' in ex:
                wp = _np.asarray(ex['waypoints'], dtype=_np.float32)
            elif 'actions' in ex:
                dt = float(ex.get('dt', ds.cfg.default_dt))
                wp = integrate_actions_to_waypoints(_np.asarray(ex['actions'], dtype=_np.float32), cfg.n_waypoints, dt)
            else:
                continue
            xs.append(wp[:, 0])
            if wp.shape[1] >= 2:
                ys.append(wp[:, 1])
            if wp.shape[1] >= 3:
                thetas.append(wp[:, 2])
            # valid mask coverage
            if 'valid_mask' in ex and isinstance(ex['valid_mask'], list):
                mv = _np.asarray(ex['valid_mask'], dtype=bool)
                mask_cov.append(float(mv.mean()))
            elif 'valid_idx' in ex and isinstance(ex['valid_idx'], list):
                mv = _np.zeros(cfg.n_waypoints, dtype=bool)
                idx = _np.asarray(ex['valid_idx'], dtype=int)
                mv[_np.clip(idx, 0, cfg.n_waypoints-1)] = True
                mask_cov.append(float(mv.mean()))
            # yaw fields presence
            if 'yaw_hist' in ex:
                yaw_hist_present += 1
            if 'yaw_curr' in ex:
                yaw_curr_present += 1
            # quick current image existence check (without IO)
            cur_rel = Path(ex.get('current', ''))
            if str(cur_rel):
                cur_abs = cur_rel if cur_rel.is_absolute() else (ds.base_root / cur_rel)
                if cur_abs.exists():
                    img_ok += 1
        if xs:
            x = _np.concatenate(xs)
            x_mu, x_sd = float(_np.mean(x)), float(_np.std(x))
        else:
            x_mu = x_sd = float('nan')
        if ys:
            y = _np.concatenate(ys)
            y_mu, y_sd = float(_np.mean(y)), float(_np.std(y))
        else:
            y_mu = y_sd = float('nan')
        th_sd = float(_np.std(_np.concatenate(thetas))) if thetas else float('nan')
        cov_mu = float(_np.mean(mask_cov)) if mask_cov else float('nan')
        print(f"[SANITY] GT x(mean={x_mu:.3f}, std={x_sd:.3f}) y(mean={y_mu:.3f}, std={y_sd:.3f}) theta(std={th_sd:.3f}) | mask_cov_mean={cov_mu:.3f}")
        print(f"[SANITY] yaw_hist_present={yaw_hist_present}/{n} yaw_curr_present={yaw_curr_present}/{n} | current_img_exists={img_ok}/{n}")
        # simple warnings
        if _np.isfinite(y_sd) and y_sd < 0.05:
            print("[SANITY][warn] GT lateral std is very small; model may learn straight lines.")
        if _np.isfinite(cov_mu) and cov_mu < 0.2:
            print("[SANITY][warn] Many waypoints are invalid; training signal may be sparse.")
    except Exception as _e:
        print(f"[SANITY] skipped due to error: {_e}")


# ----------------------- TVI + projector + planner -----------------------

class TVIEmbedder(nn.Module):
    """Temporal-Viewpoint Indicator with token insertion.
    - make_time_token(t, kind_id, view_id)
    - make_angle_token(theta, kind_id, view_id) -> uses [sinθ, cosθ] projection
    kind_id: 0 = coarse/history, 1 = fine/current.
    """
    def __init__(self, d_model: int, max_time: int = 4096, max_views: int = 1):
        super().__init__()
        self.time_emb   = nn.Embedding(max_time, d_model)
        self.view_emb   = nn.Embedding(max_views, d_model)
        self.kind_emb   = nn.Embedding(2, d_model)
        self.angle_proj = nn.Linear(2, d_model)

    def make_time_token(self, t_scalar: int, kind_id: int, view_id: int = 0,
                        device: Optional[torch.device] = None) -> torch.Tensor:
        tok = self.time_emb.weight[t_scalar] + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok.to(device) if device is not None else tok

    def make_angle_token(self, theta: float, kind_id: int, view_id: int = 0,
                         device: Optional[torch.device] = None) -> torch.Tensor:
        # project [sinθ, cosθ] into d_model
        theta = (theta + math.pi) % (2*math.pi) - math.pi
        sincos = torch.tensor([math.sin(theta), math.cos(theta)],
                              dtype=self.angle_proj.weight.dtype,
                              device=device)
        ang = F.linear(sincos, self.angle_proj.weight, self.angle_proj.bias)
        tok = ang + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok


class CrossModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x): return self.net(x)


class PlannerHead3L(nn.Module):
    """Three-layer MLP A_θ mapping E_A^T → normalized waypoints â ∈ [-1,1]."""
    def __init__(self, d_model: int, n_waypoints: int, action_dims: int, use_tanh: bool = True):
        super().__init__()
        hid = d_model * 2
        out_dim = n_waypoints * action_dims
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, out_dim)
        )
        self.nw = n_waypoints
        self.ad = action_dims
        self.use_tanh = use_tanh
    def forward(self, act_h: torch.Tensor) -> torch.Tensor:
        y = self.mlp(act_h)
        if self.use_tanh:
            y = torch.tanh(y)                 # bound to [-1,1]
        return y.view(-1, self.nw, self.ad)   # (B, M, D_action)


# ----------------------- Model -----------------------

@dataclass
class ModelConfig:
    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = False
    n_waypoints: int = 8
    max_time: int = 4096
    beta_nav: float = 10.0
    use_angle_tvi: bool = False     # single-cam default: off
    # Action/target configuration
    use_tanh_actions: bool = True   # allow removing tanh cap via flag
    alpha_xy: Optional[float] = 2.0  # Optional scalar to scale XY only; yaw stays unscaled


class OpenTrackVLA(nn.Module):
    def __init__(self, cfg: ModelConfig, vision_feat_dim: int):
        super().__init__()
        self.cfg = cfg
        self.llm = AutoModel.from_pretrained(cfg.llm_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
        self.llm.requires_grad_(not cfg.freeze_llm)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        self.D = self.llm.config.hidden_size
        self.proj = CrossModalityProjector(vision_feat_dim, self.D)
        # Always keep projector trainable regardless of LLM freeze
        self.proj.requires_grad_(True)
        self.tvi = TVIEmbedder(self.D, max_time=cfg.max_time)
        self.act_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.normal_(self.act_token, std=0.02)
        # Determine target/action dimensionality
        action_dims = 3
        self.action_dims = action_dims
        self.planner = PlannerHead3L(self.D, cfg.n_waypoints, action_dims, use_tanh=cfg.use_tanh_actions)
        # Always keep planner trainable regardless of LLM freeze
        self.planner.requires_grad_(True)
        # Freeze unused TVI heads so DDP doesn't expect grads for them
        if not cfg.use_angle_tvi:
            for p in self.tvi.angle_proj.parameters():
                p.requires_grad = False
        if cfg.alpha_xy is not None:
            vec = [1.0] * action_dims
            if action_dims >= 2:
                vec[0] = cfg.alpha_xy
                vec[1] = cfg.alpha_xy
            alpha = torch.tensor(vec, dtype=torch.float32).view(1, 1, -1)
        else:
            vec = [1.0] * action_dims
            alpha = torch.tensor(vec, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("alpha_task", alpha)

    def _embed_text(self, instructions: List[str], device):
        tok = self.tokenizer(instructions, return_tensors='pt', padding=True, truncation=True, max_length=128)
        tok = {k: v.to(device) for k, v in tok.items()}
        emb = self.llm.get_input_embeddings()(tok['input_ids'])
        return emb, tok['attention_mask']

    def _interleave_tvi(self, tokens: torch.Tensor, t_idx: torch.Tensor, kind_id: int,
                        yaw_per_frame: Optional[torch.Tensor] = None, use_angle: bool = False) -> torch.Tensor:
        """Insert TVI time (and optional angle) token before each frame's token block.
        tokens: (B, N, D_llm); t_idx: (B, N) frame ids; yaw_per_frame: (B, F) or None.
        Returns: (B, N + (1 or 2)*F, D_llm)
        """
        B, N, D = tokens.shape
        out_list = []
        for b in range(B):
            tb = t_idx[b]
            xb = tokens[b]
            items = []
            i = 0
            fcount = 0
            while i < N:
                tcur = int(tb[i].item())
                j = i + 1
                while j < N and int(tb[j].item()) == tcur:
                    j += 1
                time_tok = self.tvi.make_time_token(tcur, kind_id, device=xb.device).unsqueeze(0)
                items.append(time_tok)
                if use_angle:
                    theta = 0.0
                    if yaw_per_frame is not None and fcount < yaw_per_frame.size(1):
                        theta = float(yaw_per_frame[b, fcount].item())
                    angle_tok = self.tvi.make_angle_token(theta, kind_id, device=xb.device).unsqueeze(0)
                    items.append(angle_tok)
                items.append(xb[i:j])
                i = j
                fcount += 1
            out_list.append(torch.cat(items, dim=0))
        return torch.stack(out_list, dim=0)

    def forward(self,
                coarse_tokens, coarse_tidx,
                fine_tokens, fine_tidx,
                instructions,
                yaw_hist: Optional[torch.Tensor] = None,
                yaw_curr: Optional[torch.Tensor] = None):
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)
        # project to LLM space
        vis_c = self.proj(coarse_tokens.to(device))   # (B, Nc, D)
        vis_f = self.proj(fine_tokens.to(device))     # (B, Nf, D)
        # insert TVI tokens per frame
        vis_c = self._interleave_tvi(
            vis_c, coarse_tidx.to(device), kind_id=0,
            yaw_per_frame=yaw_hist, use_angle=self.cfg.use_angle_tvi
        )
        vis_f = self._interleave_tvi(
            vis_f, fine_tidx.to(device), kind_id=1,
            yaw_per_frame=yaw_curr, use_angle=self.cfg.use_angle_tvi
        )
        txt_emb, txt_mask = self._embed_text(instructions, device)  # (B, Ltxt, D), (B, Ltxt)
        extra = []
        act = self.act_token.expand(B, 1, -1)
        pieces = [txt_emb] + ([extra[0]] if extra else []) + [vis_c, vis_f, act]
        seq = torch.cat(pieces, dim=1).to(self.llm.dtype)
        extra_len = (extra[0].size(1) if extra else 0)
        attn = torch.cat([
            txt_mask,
            torch.ones(B, extra_len + vis_c.size(1) + vis_f.size(1) + 1, dtype=torch.long, device=device)  # +1 for ACT
        ], dim=1)
        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)
        h_act = out.last_hidden_state[:, -1, :]        # E_A^T (ACT is last)
        # Cast to float32 to match planner LayerNorm/Linear parameter dtype
        h_act = h_act.float()
        a_hat = self.planner(h_act)                # normalized [-1,1]
        tau_pred = a_hat * self.alpha_task             # absolute units
        return tau_pred


# ----------------------- Dataset -----------------------

@dataclass
class DataConfig:
    train_json: str
    n_waypoints: int = 8
    history: int = 31
    default_dt: float = 0.1
    cache_root: Optional[str] = None


class JsonTrackingDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        p = Path(cfg.train_json)
        # Determine base root to prefix relative paths in JSON examples
        # Robustly ascend until 'frames' directory is found (common layout: <root>/frames/...)
        candidate = p if p.is_dir() else p.parent
        max_up = 4
        while max_up >= 0 and not (candidate / 'frames').exists():
            if candidate.parent == candidate:
                break
            candidate = candidate.parent
            max_up -= 1
        self.base_root = candidate
        # Determine cache root for token files (defaults to <base_root>/vision_cache)
        self.cache_root = Path(cfg.cache_root) if cfg.cache_root is not None else (self.base_root / "vision_cache")
        # Lazy online encoder (created on first use)
        self._online_encoder: Optional[VisionFeatureCacher] = None
        # Dataset storage: either eager JSON list or lazy JSONL/directory index
        self._lazy = False
        self._index: Optional[List[Tuple[str, int]]] = None  # list of (file_path, byte_offset) per example
        self.examples: Optional[List[Dict[str, Any]]] = None
        if p.is_file() and p.suffix.lower() == '.json':
            data = load_examples_from_path(cfg.train_json)
            assert isinstance(data, list) and len(data) > 0, "JSON file must contain a non-empty list"
            self.examples = data
        else:
            # Build lazy index over .jsonl file or directory of .jsonl files
            files: List[Path] = []
            if p.is_file() and p.suffix.lower() == '.jsonl':
                files = [p]
            elif p.is_dir():
                files = sorted(p.rglob('*.jsonl'))
            if len(files) == 0:
                raise FileNotFoundError(f"No .jsonl files found under: {cfg.train_json}")
            self._lazy = True
            self._index = []
            for fp in files:
                try:
                    with open(fp, 'rb') as f:
                        pos = 0
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            if line.strip():
                                self._index.append((str(fp), pos))
                            pos += len(line)
                except Exception as _e:
                    raise _e
            if len(self._index) == 0:
                raise RuntimeError(f"No examples indexed from .jsonl sources under: {cfg.train_json}")
        # Target history length is configured; per-sample sequences will be padded/trimmed to this length
        H_target = int(self.cfg.history)
        self.coarse_frames = H_target

    def __len__(self):
        if self.examples is not None:
            return len(self.examples)
        if self._index is not None:
            return len(self._index)
        return 0

    def _load_tokens(self, path: str) -> torch.Tensor:
        return load_tokens_file(path)

    def _get_online_encoder(self) -> VisionFeatureCacher:
        if self._online_encoder is None:
            # Use CPU when running with multiple workers to avoid GPU contention
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            use_cuda = torch.cuda.is_available() and (worker_info is None)
            cfg = VisionCacheConfig(image_size=384, batch_size=8, device=('cuda' if use_cuda else 'cpu'))
            self._online_encoder = VisionFeatureCacher(cfg)
            self._online_encoder.eval()
        return self._online_encoder

    @torch.inference_mode()
    def _encode_image_tokens(self, img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._get_online_encoder()
        pil = Image.open(str(img_path)).convert('RGB')
        tok_dino, Hp, Wp = enc._encode_dino([pil])               # (1, P, C_dino)
        tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))    # (1, P, C_sigl)
        Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)         # (1, P, C_total)
        Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)  # (1, 64, C_total)
        Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4) # (1, 4,  C_total)
        return Vcoarse[0].cpu().float(), Vfine[0].cpu().float()

    def _read_indexed_example(self, idx: int) -> Dict[str, Any]:
        assert self._lazy and self._index is not None
        fp, off = self._index[idx]
        with open(fp, 'rb') as f:
            f.seek(off)
            line = f.readline()
        return json.loads(line.decode('utf-8'))

    def get_example(self, idx: int) -> Dict[str, Any]:
        if self._lazy:
            return self._read_indexed_example(idx)
        assert self.examples is not None
        return self.examples[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.get_example(idx)
        H = self.coarse_frames

        # fine current (load first to determine feature dimensionality for zero padding)
        curr_path = Path(ex['current'])
        abs_curr_img = curr_path if curr_path.is_absolute() else (self.base_root / curr_path)
        try:
            rel_curr = abs_curr_img.relative_to(self.base_root)
        except ValueError:
            rel_curr = abs_curr_img
        curr_token_dir = self.cache_root / rel_curr.parent
        curr_token_name = rel_curr.stem + "_vfine.pt"
        curr_tok_path = curr_token_dir / curr_token_name
        try:    
            fine_tokens = self._load_tokens(str(curr_tok_path))  # (64, C)
        except Exception as e:            
            # Fallback: encode online and save for reuse
            curr_token_dir.mkdir(parents=True, exist_ok=True)
            vc, vf = self._encode_image_tokens(abs_curr_img)
            print (curr_tok_path)
            try:
                torch.save(vf.half(), str(curr_tok_path))
            except Exception as e:
                print (str(e))
                pass
            fine_tokens = vf
        fine_tidx = torch.full((fine_tokens.size(0),), fill_value=H, dtype=torch.long)

        # coarse history with left-padding using earliest available token (edge padding)
        imgs_src = ex.get('images', [])
        imgs_trim = imgs_src[-H:]
        missing = H - len(imgs_trim)
        coarse_list, coarse_tidx = [], []
        first_tok: Optional[torch.Tensor] = None
        current_vc: Optional[torch.Tensor] = None
        for t in range(H):
            if t < missing:
                # placeholder; will fill with first_tok after we load at least one real token
                tok = None
            else:
                img_p = imgs_trim[t - missing]
                # Map image path -> corresponding Vcoarse token path within cache_root
                rp = Path(img_p)
                abs_img = rp if rp.is_absolute() else (self.base_root / rp)
                try:
                    rel_img = abs_img.relative_to(self.base_root)
                except ValueError:
                    rel_img = abs_img
                token_dir = self.cache_root / rel_img.parent
                token_name = rel_img.stem + "_vcoarse.pt"
                tok_path = token_dir / token_name
                try:
                    tok = self._load_tokens(str(tok_path))
                except Exception as e:
                    # Fallback: encode online from image and save for reuse
                    token_dir.mkdir(parents=True, exist_ok=True)
                    vc, vf = self._encode_image_tokens(abs_img)
                    try:
                        torch.save(vc.half(), str(tok_path))
                    except Exception:
                        pass
                    tok = vc
                if first_tok is None:
                    first_tok = tok
            # Fill left padding with first available token (edge padding);
            # if none, use current frame's coarse tokens; fallback to zeros if that also fails
            if tok is None:
                if first_tok is not None:
                    tok = first_tok
                else:
                    # Try to obtain current coarse tokens
                    try:
                        if current_vc is None:
                            # Attempt to load cached current coarse tokens alongside fine
                            cur_coarse_name = rel_curr.stem + "_vcoarse.pt"
                            cur_coarse_path = curr_token_dir / cur_coarse_name
                            try:
                                current_vc = self._load_tokens(str(cur_coarse_path))
                            except Exception:
                                # Encode from current image if not cached
                                vc_tmp, _ = self._encode_image_tokens(abs_curr_img)
                                current_vc = vc_tmp
                        tok = current_vc
                    except Exception:
                        tok = torch.zeros(4, fine_tokens.size(1), dtype=torch.float32)
            coarse_list.append(tok)
            coarse_tidx.append(torch.full((tok.size(0),), fill_value=t, dtype=torch.long))
        coarse_tokens = torch.cat(coarse_list, dim=0)      # (H*4, C)
        coarse_tidx   = torch.cat(coarse_tidx, dim=0)      # (H*4,)

        # yaw (optional; used only if angle TVI is enabled)
        yaw_hist = torch.tensor(ex.get('yaw_hist', [0.0]*H), dtype=torch.float32)            # (H,)
        yaw_curr = torch.tensor(ex.get('yaw_curr', 0.0), dtype=torch.float32).view(1)        # (1,)

        # targets
        if 'waypoints' in ex:
            wp = torch.tensor(ex['waypoints'], dtype=torch.float32)
        else:
            assert 'actions' in ex, "JSON needs either 'waypoints' or 'actions'"
            dt = float(ex.get('dt', self.cfg.default_dt))
            traj = integrate_actions_to_waypoints(np.asarray(ex['actions'], dtype=np.float32), self.cfg.n_waypoints, dt)
            wp = torch.from_numpy(traj)

        # validity mask/idx (optional)
        if 'valid_mask' in ex:
            vm = torch.tensor(ex['valid_mask'], dtype=torch.bool)
        elif 'valid_idx' in ex:
            vm = torch.zeros(self.cfg.n_waypoints, dtype=torch.bool)
            vm[torch.tensor(ex['valid_idx'], dtype=torch.long)] = True
        else:
            vm = torch.ones(self.cfg.n_waypoints, dtype=torch.bool)

        item: Dict[str, Any] = {
            'coarse_tokens': coarse_tokens,
            'coarse_tidx':   coarse_tidx,
            'fine_tokens':   fine_tokens,
            'fine_tidx':     fine_tidx,
            'yaw_hist':      yaw_hist,     # (H,)
            'yaw_curr':      yaw_curr,     # (1,)
            'waypoints':     wp,           # (M, D_action)
            'valid_mask':    vm,           # (M,)
            'instruction':   ex.get('instruction', 'follow the person'),
            'current_path':  str(abs_curr_img),
        }
        return item


# ----------------------- Loss & Train -----------------------

def mse_masked(pred: torch.Tensor, target: torch.Tensor, mask_waypoints: torch.Tensor) -> torch.Tensor:
    """Mean squared error over selected waypoints (absolute units)."""
    assert pred.shape == target.shape
    B, M, D = pred.shape
    mask = mask_waypoints.view(B, M, 1).expand(B, M, D)
    se = (pred - target).pow(2)
    se = se[mask]
    return se.mean() if se.numel() > 0 else pred.new_tensor(0.0)


def _compute_total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        return float(total_norm.item())
    total = torch.zeros([], device=device)
    for p in parameters:
        param_norm = p.grad.detach().data.norm(norm_type)
        total += param_norm.pow(norm_type)
    total = total.pow(1.0 / norm_type)
    return float(total.item())


@dataclass
class TrainConfig:
    train_json: str
    out_dir: str = './ckpts_qwen4'
    n_waypoints: int = 8
    history: int = 31
    llm_name: str = "Qwen/Qwen3-0.6B"
    epochs: int = 1
    batch_size: int = 12
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixed_precision: bool = True
    vision_feat_dim: int = 1536
    seed: int = 0
    num_workers: int = 4
    # model
    use_angle_tvi: bool = False
    beta_nav: float = 10.0
    cache_root: Optional[str] = None
    distributed: bool = False
    dist_backend: str = 'nccl'
    alpha_xy: Optional[float] = 2.0
    # logging
    log_every: int = 10
    csv_logging: bool = True
    # trajectory saving
    save_trajectories: bool = False
    traj_subdir: str = 'trajectories'
    # evaluation
    val_json: Optional[str] = None
    eval_every: int = 0
    eval_batches: int = 8
    final_wp_threshold: float = 0.2
    # single-episode evaluation
    episode_json: Optional[str] = None
    episode_eval_every: int = 0
    episode_threshold: float = 0.2
    episode_max_frames: int = 256
    # modeling options
    no_tanh_actions: bool = True
    # checkpoint retention
    max_ckpts: int = 2
    # resume
    resume: bool = False
    resume_ckpt: Optional[str] = None
    # inference
    infer_json: Optional[str] = None
    infer_ckpt: Optional[str] = None
    infer_out: str = './infer_out'
    infer_batches: int = 0
    infer_vis: bool = False
    infer_save_npz: bool = True


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    instr = [b['instruction'] for b in batch]
    return {
        'coarse_tokens': torch.stack([b['coarse_tokens'] for b in batch], dim=0),
        'coarse_tidx':   torch.stack([b['coarse_tidx']   for b in batch], dim=0),
        'fine_tokens':   torch.stack([b['fine_tokens']   for b in batch], dim=0),
        'fine_tidx':     torch.stack([b['fine_tidx']     for b in batch], dim=0),
        'yaw_hist':      torch.stack([b['yaw_hist']      for b in batch], dim=0),   # (B,H)
        'yaw_curr':      torch.stack([b['yaw_curr']      for b in batch], dim=0),   # (B,1)
        'waypoints':     torch.stack([b['waypoints']     for b in batch], dim=0),
        'valid_mask':    torch.stack([b['valid_mask']    for b in batch], dim=0),
        'instruction':   instr,
        'current_path':  [b['current_path'] for b in batch]
    }


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    # Distributed setup
    is_cuda = torch.cuda.is_available()
    use_ddp = bool(cfg.distributed) and is_cuda
    if use_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)

        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method='env://',
            device_id=device,  # or device_id=local_rank
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()

    else:
        device = torch.device('cuda' if is_cuda else 'cpu')
        local_rank = 0
        rank = 0
        world_size = 1

    ds = JsonTrackingDataset(DataConfig(train_json=cfg.train_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
    if rank == 0:
        _dataset_sanity_report(ds, cfg)
    # Auto-compute scalar XY scaling (alpha_xy) from dataset if not provided
    if cfg.alpha_xy is None:
        try:
            import numpy as _np
            vals = []
            sample_n = min(4000, len(ds))
            for i in range(sample_n):
                ex = ds.get_example(i)
                arr = None
                if 'waypoints' in ex:
                    arr = _np.asarray(ex['waypoints'], dtype=_np.float32)
                elif 'actions' in ex:
                    dt = float(ex.get('dt', ds.cfg.default_dt))
                    arr = integrate_actions_to_waypoints(_np.asarray(ex['actions'], dtype=_np.float32), cfg.n_waypoints, dt)
                if arr is None:
                    continue
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.shape[1] >= 2:
                    r = _np.linalg.norm(arr[:, :2], axis=-1)
                    vals.append(r)
            if vals:
                allr = _np.concatenate(vals)
                alpha_est = float(_np.percentile(allr, 95))
                alpha_est = max(alpha_est, 1e-3)
                cfg.alpha_xy = alpha_est
                if rank == 0:
                    print(f"[auto_alpha_xy] alpha_xy set to {alpha_est:.3f} from dataset percentiles ({cfg.train_json})")
        except Exception as _e:
            if rank == 0:
                print(f"[auto_alpha_xy] skipped due to error: {_e}")
    # Auto-detect vision_feat_dim from dataset if needed
    if rank == 0:
        try:
            sample_item = ds[0]
            detected_dim = None
            if 'fine_tokens' in sample_item:
                detected_dim = sample_item['fine_tokens'].shape[-1]
            elif 'coarse_tokens' in sample_item:
                detected_dim = sample_item['coarse_tokens'].shape[-1]
            if detected_dim is not None and detected_dim != cfg.vision_feat_dim:
                print(f"[AUTO_DIM] Detected vision_feat_dim={detected_dim} from dataset (config had {cfg.vision_feat_dim}), updating...")
                cfg.vision_feat_dim = detected_dim
        except Exception as e:
            print(f"[AUTO_DIM] Failed to auto-detect vision_feat_dim: {e}, using config value {cfg.vision_feat_dim}")
    
    # Broadcast the updated vision_feat_dim to all ranks in DDP
    if use_ddp:
        vision_feat_dim_tensor = torch.tensor([cfg.vision_feat_dim], dtype=torch.int32, device=device)
        dist.broadcast(vision_feat_dim_tensor, src=0)
        cfg.vision_feat_dim = int(vision_feat_dim_tensor.item())

    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=(sampler is None), num_workers=cfg.num_workers,
                    pin_memory=True, collate_fn=collate_batch, sampler=sampler)

    model = OpenTrackVLA(
        ModelConfig(
            llm_name=cfg.llm_name,
            n_waypoints=cfg.n_waypoints,
            beta_nav=cfg.beta_nav,
            use_angle_tvi=cfg.use_angle_tvi,
            use_tanh_actions=(not cfg.no_tanh_actions),
            alpha_xy=cfg.alpha_xy,
        ),
        vision_feat_dim=cfg.vision_feat_dim,
    ).to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Log trainable vs frozen parameters (rank 0 only)
    if rank == 0:
        try:
            from collections import defaultdict
            model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            total_params = sum(p.numel() for p in model_inspect.parameters())
            trainable_params = sum(p.numel() for p in model_inspect.parameters() if p.requires_grad)
            pct = (trainable_params / max(1, total_params)) * 100.0
            print(f"[PARAMS] total={total_params:,} trainable={trainable_params:,} ({pct:.2f}%)")
            group_counts = defaultdict(lambda: [0, 0])  # [total, trainable]
            for name, p in model_inspect.named_parameters():
                head = name.split('.')[0]
                n = p.numel()
                group_counts[head][0] += n
                if p.requires_grad:
                    group_counts[head][1] += n
            summary = ' '.join([f"{k}:{v[1]}/{v[0]}" for k, v in group_counts.items()])
            print(f"[PARAMS groups] {summary}")
            tn = [n for n, p in model_inspect.named_parameters() if p.requires_grad][:16]
            print(f"[TRAINABLE names (first 16)] {tn}")
        except Exception as _e:
            print(f"[PARAMS] logging skipped due to error: {_e}")

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Mixed precision configuration
    amp_enabled = cfg.mixed_precision and is_cuda
    amp_dtype = torch.bfloat16  # switch to torch.float16 if you want fp16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_enabled and amp_dtype == torch.float16))

    # Optionally resume from checkpoint
    start_epoch = 0
    step = 0
    if cfg.resume:
        try:
            import glob as _glob
            ckpt_path = cfg.resume_ckpt
            if ckpt_path is None:
                pts = sorted(_glob.glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p))
                ckpt_path = pts[-1] if pts else None
            if ckpt_path and os.path.exists(ckpt_path):
                obj = torch.load(ckpt_path, map_location=device)
                msd = obj.get('model_state', None)
                if msd:
                    msd = _cleanup_state_dict_keys(msd)
                    model_to_load = model.module if isinstance(
                        model, torch.nn.parallel.DistributedDataParallel
                    ) else model
                    model_to_load.load_state_dict(msd, strict=False)
                osd = obj.get('optim_state', None)
                if osd:
                    optim.load_state_dict(osd)
                ssd = obj.get('scaler_state', None)
                if ssd and scaler.is_enabled():
                    scaler.load_state_dict(ssd)
                start_epoch = int(obj.get('epoch', 0))
                step = int(obj.get('step', 0))
                if rank == 0:
                    print(f"[RESUME] Loaded {ckpt_path} | epoch={start_epoch} step={step}")
        except Exception as _e:
            if rank == 0:
                print(f"[RESUME] Skipped due to error: {_e}")

    if rank == 0:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # continue step counter if resumed
    ema_loss: Optional[float] = None
    ema_nav: Optional[float] = None
    last_log_time = time.time()
    epoch_start_time = last_log_time
    for epoch in range(cfg.epochs):
        model.train()
        if use_ddp and sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            coarse_tokens = batch['coarse_tokens'].to(device)
            coarse_tidx   = batch['coarse_tidx'].to(device)
            fine_tokens   = batch['fine_tokens'].to(device)
            fine_tidx     = batch['fine_tidx'].to(device)
            yaw_hist      = batch['yaw_hist'].to(device)   # (B,H)
            yaw_curr      = batch['yaw_curr'].to(device)   # (B,1)
            gt_wp         = batch['waypoints'].to(device)
            valid_mask    = batch['valid_mask'].to(device)
            instr         = batch['instruction']

            optim.zero_grad(set_to_none=True)
            amp_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
            with amp_ctx:
                tau_pred = model(
                    coarse_tokens, coarse_tidx,
                    fine_tokens, fine_tidx,
                    instr,
                    yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                    yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                )
                # Option A: compute loss in normalized space — divide XY by alpha, keep yaw unscaled
                # Get alpha vector from model (shape: (1,1,D_action)) and clamp to avoid div-by-zero
                model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                alpha_vec = getattr(model_inspect, 'alpha_task', None)
                if alpha_vec is None:
                    # Fallback: no scaling information; compute loss in absolute space
                    pred_norm = tau_pred
                    gt_norm = gt_wp
                else:
                    # Normalize only XY dims (0,1); leave others (e.g., yaw) unscaled
                    pred_norm = tau_pred
                    gt_norm = gt_wp
                    if pred_norm.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                        ax = alpha_vec[..., 0:2].clamp_min(1e-6)
                        pred_norm = pred_norm.clone()
                        gt_norm = gt_norm.clone()
                        pred_norm[..., 0:2] = pred_norm[..., 0:2] / ax
                        gt_norm[..., 0:2] = gt_norm[..., 0:2] / ax
                L_nav = mse_masked(pred_norm, gt_norm, valid_mask)
                L_QA = tau_pred.new_tensor(0.0)
                loss = cfg.beta_nav * L_nav + L_QA
            # Save trajectories per-step (rank 0)
            if rank == 0 and cfg.save_trajectories:
                try:
                    traj_root = os.path.join(cfg.out_dir, cfg.traj_subdir)
                    os.makedirs(traj_root, exist_ok=True)
                    with torch.no_grad():
                        pred_np = tau_pred.detach().float().cpu().numpy()
                        gt_np = gt_wp.detach().float().cpu().numpy()
                        vm_np = valid_mask.detach().cpu().numpy()
                    Bcur = pred_np.shape[0]
                    for bi in range(Bcur):
                        fpath = os.path.join(traj_root, f"ep{epoch:02d}_st{step+1:06d}_b{bi:03d}.npz")
                        # step+1 so filenames are 1-indexed and align with printed step after increment
                        np.savez_compressed(
                            fpath,
                            pred=pred_np[bi],
                            gt=gt_np[bi],
                            valid_mask=vm_np[bi],
                            instruction=instr[bi],
                            epoch=epoch,
                            step=step+1
                        )
                except Exception:
                    pass
            scaler.scale(loss).backward()
            grad_norm_before = 0.0
            if cfg.grad_clip is not None:
                scaler.unscale_(optim)
                grad_norm_before = _compute_total_grad_norm([p for p in model.parameters() if getattr(p, 'grad', None) is not None])
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update()

            step += 1
            if rank == 0 and (step % cfg.log_every == 0):
                now = time.time()
                dt = now - last_log_time
                last_log_time = now
                B = coarse_tokens.size(0)
                lr = optim.param_groups[0]['lr']
                with torch.no_grad():
                    tp = tau_pred.detach().float()
                    gwp = gt_wp.detach().float()
                    vm = valid_mask.detach().float()
                    # Scale predictions to absolute units for logging if alpha is available
                    tp_abs = tp
                    try:
                        model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                        alpha_vec = getattr(model_inspect, 'alpha_task', None)
                        if alpha_vec is not None and alpha_vec.size(-1) >= tp.size(-1):
                            av = alpha_vec.to(tp.device, tp.dtype)
                            tp_abs = tp * av
                    except Exception:
                        pass
                    pred_mean = tp_abs.mean().item()
                    pred_std = tp_abs.std().item()
                    pred_absmax = tp_abs.abs().max().item()
                    gt_mean = gwp.mean().item()
                    gt_std = gwp.std().item()
                    mask_cov = vm.mean().item()
                    mse_total = (tp_abs - gwp).pow(2).mean().item()
                    ad = tp_abs.size(-1)
                    per_dim_mse = []
                    for d in range(min(4, ad)):
                        per_dim_mse.append(float((tp_abs[..., d] - gwp[..., d]).pow(2).mean().item()))
                loss_val = float(loss.detach().item())
                nav_val = float(L_nav.detach().item())
                ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)
                ema_nav = nav_val if ema_nav is None else (0.98 * ema_nav + 0.02 * nav_val)

                mem_alloc_mb = mem_peak_mb = 0.0
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    mem_alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
                    mem_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

                msg_lines = [
                    f"epoch {epoch} step {step} | lr={lr:.2e} | loss={loss_val:.4f} (ema {ema_loss:.4f}) | L_nav={nav_val:.4f} (ema {ema_nav:.4f})",
                    f"  mask_cov={mask_cov:.3f} | grad_norm_preclip={grad_norm_before:.3f} | step_time={dt:.3f}s | throughput={B/dt:.2f} it/s",
                    f"  pred(mean={pred_mean:.3f}, std={pred_std:.3f}, absmax={pred_absmax:.3f}) | gt(mean={gt_mean:.3f}, std={gt_std:.3f})",
                    f"  mse_total={mse_total:.5f} | per_dim_mse={per_dim_mse} | mem_alloc={mem_alloc_mb:.1f}MB peak={mem_peak_mb:.1f}MB"
                ]
                print("\n".join(msg_lines), flush=True)

                # Debug preview: print GT and Pred waypoints (absolute), and normalized XY if alpha is set
                """
                try:
                    import numpy as _np
                    model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    alpha_vec = getattr(model_inspect, 'alpha_task', None)
                    pred_abs_b0 = tp[0].detach().cpu().numpy()
                    gt_abs_b0 = gwp[0].detach().cpu().numpy()
                    print("[WAYPOINTS abs][b0] pred=", _np.array2string(pred_abs_b0, precision=3, floatmode='fixed'))
                    print("[WAYPOINTS abs][b0]   gt=", _np.array2string(gt_abs_b0, precision=3, floatmode='fixed'))
                    if alpha_vec is not None and pred_abs_b0.shape[1] >= 2 and alpha_vec.size(-1) >= 2:
                        ax = alpha_vec[0, 0, 0:2].clamp_min(1e-6).detach().float().cpu()
                        pred_n_xy = (tp[0, :, 0:2].detach().cpu() / ax).numpy()
                        gt_n_xy = (gwp[0, :, 0:2].detach().cpu() / ax).numpy()
                        print(f"[WAYPOINTS norm][b0] alpha_xy={ax.numpy().tolist()} pred_xy=", _np.array2string(pred_n_xy, precision=3, floatmode='fixed'))
                        print("[WAYPOINTS norm][b0]   gt_xy=", _np.array2string(gt_n_xy, precision=3, floatmode='fixed'))
                except Exception:
                    pass
                """
                if cfg.csv_logging:
                    csv_path = os.path.join(cfg.out_dir, 'train_log.csv')
                    header = [
                        'epoch','step','lr','loss','loss_ema','L_nav','L_nav_ema','mask_cov',
                        'grad_norm_preclip','step_time','throughput_it_per_s','pred_mean','pred_std','pred_absmax',
                        'gt_mean','gt_std','mse_total','mem_alloc_mb','mem_peak_mb'
                    ] + [f'mse_dim_{i}' for i in range(len(per_dim_mse))]
                    write_header = not os.path.exists(csv_path)
                    try:
                        with open(csv_path, 'a', newline='') as f:
                            w = csv.writer(f)
                            if write_header:
                                w.writerow(header)
                            row = [
                                epoch, step, lr, loss_val, ema_loss, nav_val, ema_nav, mask_cov,
                                grad_norm_before, dt, (B/dt), pred_mean, pred_std, pred_absmax,
                                gt_mean, gt_std, mse_total, mem_alloc_mb, mem_peak_mb
                            ] + per_dim_mse
                            w.writerow(row)
                    except Exception:
                        pass

                # Visualize GT vs Pred on current images 
                if rank == 0 and step % 100 == 0:
                    try:
                        vis_dir = os.path.join(cfg.out_dir, 'vis')
                        os.makedirs(vis_dir, exist_ok=True)
                        with torch.no_grad():
                            # Ensure predictions are in absolute units for visualization
                            pred_draw = tau_pred.detach().float()
                            pred_np = pred_draw.cpu().numpy()
                            gt_np = gwp.detach().float().cpu().numpy()
                            cur_paths = batch.get('current_path', [])
                        Bcur = pred_np.shape[0]
                        for bi in range(min(Bcur, 4)):
                            cur_path = cur_paths[bi] if isinstance(cur_paths, list) and bi < len(cur_paths) else None
                            if cur_path is None or (not os.path.exists(cur_path)):
                                continue
                            pil_img = Image.open(cur_path).convert('RGB')
                            draw = ImageDraw.Draw(pil_img)
                            w, h = pil_img.size
                            base_x = w // 2
                            base_y = int(h * 0.86)
                            def to_pxxy(traj):
                                pts = []
                                for i in range(min(traj.shape[0], 64)):
                                    x, y = float(traj[i, 0]), float(traj[i, 1])
                                    # y is left-positive in robot frame → screen x grows to right ⇒ subtract
                                    px = base_x - int(y * 120)
                                    py = base_y - int(x * 120)
                                    pts.append((px, py))
                                return pts
                            pts_pred = to_pxxy(pred_np[bi])
                            pts_gt   = to_pxxy(gt_np[bi])
                            # outline
                            for seq, color in ((pts_gt, (0,0,0)), (pts_pred, (0,0,0))):
                                for i2 in range(1, len(seq)):
                                    draw.line([seq[i2-1], seq[i2]], fill=color, width=10)
                            # body
                            for i2 in range(1, len(pts_gt)):
                                draw.line([pts_gt[i2-1], pts_gt[i2]], fill=(255, 200, 0), width=6)
                            for i2 in range(1, len(pts_pred)):
                                draw.line([pts_pred[i2-1], pts_pred[i2]], fill=(0, 255, 200), width=6)
                            # start points
                            if pts_gt:
                                r0 = 6
                                sx, sy = pts_gt[0]
                                draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(255,255,255))
                            if pts_pred:
                                r0 = 6
                                sx, sy = pts_pred[0]
                                draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(0,255,0))
                            try:
                                from inspect import isclass
                                model_cfg = (model.module.cfg if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.cfg)
                            except Exception:
                                pass
                            out_path = os.path.join(vis_dir, f"ep{epoch:02d}_st{step:06d}_b{bi:03d}.jpg")
                            pil_img.save(out_path)
                            print(f"[VIS] saved {out_path}")
                    except Exception:
                        pass

            if step % 100 == 0 and rank == 0:
                ckpt = os.path.join(cfg.out_dir, f"model_epoch{epoch:02d}_step{step:06d}.pt")
                # Always save the *underlying* model (so no 'module.' prefix)
                model_to_save = model.module if isinstance(
                    model, torch.nn.parallel.DistributedDataParallel
                ) else model

                torch.save(
                {
                    'epoch': epoch,
                    'model_state': model_to_save.state_dict(),
                    'optim_state': optim.state_dict(),
                    'scaler_state': (scaler.state_dict() if scaler.is_enabled() else None),
                    'config': cfg.__dict__,
                    'step': step,
                },
                ckpt,
                )
                try:
                    from glob import glob
                    pts = sorted(glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p), reverse=True)
                    if cfg.max_ckpts is not None and cfg.max_ckpts > 0 and len(pts) > cfg.max_ckpts:
                        for old in pts[cfg.max_ckpts:]:
                            try:
                                os.remove(old)
                            except Exception:
                                pass
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        torch.cuda.reset_peak_memory_stats(device)
                    except Exception:
                        pass

            # Periodic evaluation on a small held-out set (rank 0 only)
            if (cfg.eval_every and (step % cfg.eval_every == 0) and rank == 0 and cfg.val_json):
                try:
                    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    model_eval.eval()
                    with torch.inference_mode():
                        vds = JsonTrackingDataset(DataConfig(train_json=cfg.val_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
                        vdl = DataLoader(vds, batch_size=cfg.batch_size, shuffle=False, num_workers=min(2, cfg.num_workers), pin_memory=True, collate_fn=collate_batch)
                        total_mse = 0.0
                        total_count = 0
                        final_errors: List[float] = []
                        hits = 0
                        max_batches = max(1, cfg.eval_batches)
                        bdone = 0
                        for vbatch in vdl:
                            coarse_tokens = vbatch['coarse_tokens'].to(device)
                            coarse_tidx   = vbatch['coarse_tidx'].to(device)
                            fine_tokens   = vbatch['fine_tokens'].to(device)
                            fine_tidx     = vbatch['fine_tidx'].to(device)
                            yaw_hist      = vbatch['yaw_hist'].to(device)
                            yaw_curr      = vbatch['yaw_curr'].to(device)
                            gt_wp         = vbatch['waypoints'].to(device)
                            valid_mask    = vbatch['valid_mask'].to(device)
                            instr         = vbatch['instruction']

                            pred = model_eval(
                                coarse_tokens, coarse_tidx,
                                fine_tokens, fine_tidx,
                                instr,
                                yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                                yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                            )
                            # masked MSE in normalized space (Option A): divide XY by alpha
                            model_inspect = model_eval
                            alpha_vec = getattr(model_inspect, 'alpha_task', None)
                            if alpha_vec is not None and pred.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                                ax = alpha_vec[..., 0:2].clamp_min(1e-6)
                                pred_n = pred.clone()
                                gt_n = gt_wp.clone()
                                pred_n[..., 0:2] = pred_n[..., 0:2] / ax
                                gt_n[..., 0:2] = gt_n[..., 0:2] / ax
                                mse = mse_masked(pred_n, gt_n, valid_mask).item()
                            else:
                                mse = mse_masked(pred, gt_wp, valid_mask).item()
                            total_mse += mse * pred.size(0)
                            total_count += pred.size(0)
                            # final waypoint EPE and hit rate (first two dims assumed spatial)
                            pred_xy = pred[:, -1, :2].float()
                            gt_xy = gt_wp[:, -1, :2].float()
                            epe = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # (B,)
                            final_errors.extend(epe.cpu().tolist())
                            hits += (epe <= cfg.final_wp_threshold).sum().item()

                            bdone += 1
                            if bdone >= max_batches:
                                break
                    mean_mse = total_mse / max(1, total_count)
                    if len(final_errors) > 0:
                        import numpy as _np
                        epe_mean = float(_np.mean(final_errors))
                        epe_median = float(_np.median(final_errors))
                        hit_rate = float(hits / len(final_errors))
                    else:
                        epe_mean = epe_median = hit_rate = float('nan')
                    print(f"[VAL] step {step} | masked_MSE={mean_mse:.5f} | final_EPE_mean={epe_mean:.4f} | final_EPE_median={epe_median:.4f} | hit@{cfg.final_wp_threshold}={hit_rate:.3f}", flush=True)
                except Exception as _e:
                    print(f"[VAL] evaluation skipped due to error: {_e}")
                finally:
                    model.train()

            # Single-episode evaluation (rank 0 only)
            if (cfg.episode_eval_every and (step % cfg.episode_eval_every == 0) and rank == 0 and cfg.episode_json):
                try:
                    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    model_eval.eval()
                    with torch.inference_mode():
                        ds_tmp = JsonTrackingDataset(DataConfig(train_json=cfg.episode_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
                        if len(ds_tmp) == 0:
                            raise RuntimeError('episode_json produced no examples')
                        max_frames = min(cfg.episode_max_frames, len(ds_tmp))
                        epe_list: List[float] = []
                        hits = 0
                        for i in range(max_frames):
                            item = ds_tmp[i]
                            coarse_tokens = item['coarse_tokens'].unsqueeze(0).to(device)
                            coarse_tidx   = item['coarse_tidx'].unsqueeze(0).to(device)
                            fine_tokens   = item['fine_tokens'].unsqueeze(0).to(device)
                            fine_tidx     = item['fine_tidx'].unsqueeze(0).to(device)
                            yaw_hist      = item['yaw_hist'].unsqueeze(0).to(device)
                            yaw_curr      = item['yaw_curr'].unsqueeze(0).to(device)
                            gt_wp         = item['waypoints'].unsqueeze(0).to(device)
                            instr         = [item['instruction']]

                            pred = model_eval(
                                coarse_tokens, coarse_tidx,
                                fine_tokens, fine_tidx,
                                instr,
                                yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                                yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                            )
                            pred_xy = pred[:, -1, :2].float()
                            gt_xy = gt_wp[:, -1, :2].float()
                            epe = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # (1,)
                            e = float(epe.item())
                            epe_list.append(e)
                            if e <= cfg.episode_threshold:
                                hits += 1
                        if len(epe_list) > 0:
                            import numpy as _np
                            epe_mean = float(_np.mean(epe_list))
                            epe_median = float(_np.median(epe_list))
                            follow_rate = float(hits / len(epe_list))
                        else:
                            epe_mean = epe_median = follow_rate = float('nan')
                    print(f"[EPISODE] step {step} | frames={len(epe_list)} | EPE_mean={epe_mean:.4f} | EPE_median={epe_median:.4f} | follow@{cfg.episode_threshold}={follow_rate:.3f}", flush=True)
                except Exception as _e:
                    print(f"[EPISODE] evaluation skipped due to error: {_e}")
                finally:
                    model.train()

    # Optional inference after training when requested (rank 0 only)
    if rank == 0 and cfg.infer_json:
        try:
            _run_inference(cfg)
        except Exception as _e:
            print(f"[INFER] failed: {_e}")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"[TRAIN] Finished all epochs. last_step={step}")


# ----------------------- Inference -----------------------

@torch.inference_mode()
def _run_inference(cfg: TrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Resolve checkpoint
    ckpt_path = cfg.infer_ckpt
    try:
        if ckpt_path is None:
            from glob import glob
            pts = sorted(glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p))
            ckpt_path = pts[-1] if pts else None
    except Exception:
        ckpt_path = None
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"No checkpoint found for inference (looked at --infer_ckpt and {cfg.out_dir})")

    # Load checkpoint config
    obj = torch.load(ckpt_path, map_location=device)
    ck = obj if isinstance(obj, dict) else {}
    ck_cfg = ck.get('config', {})

    # Build model config mirroring training
    n_waypoints = int(ck_cfg.get('n_waypoints', cfg.n_waypoints))
    use_angle_tvi = bool(ck_cfg.get('use_angle_tvi', cfg.use_angle_tvi))
    no_tanh_actions = bool(ck_cfg.get('no_tanh_actions', cfg.no_tanh_actions))
    vision_feat_dim = int(ck_cfg.get('vision_feat_dim', cfg.vision_feat_dim))
    alpha_xy        = ck_cfg.get('alpha_xy', getattr(cfg, 'alpha_xy', None))
    llm_name        = str(ck_cfg.get('llm_name', getattr(cfg, 'llm_name', "Qwen/Qwen3-8B")))
    print (llm_name)

    model = OpenTrackVLA(
        ModelConfig(
            llm_name=llm_name,
            n_waypoints=n_waypoints,
            beta_nav=float(ck_cfg.get('beta_nav', cfg.beta_nav)),
            use_angle_tvi=use_angle_tvi,
            use_tanh_actions=(not no_tanh_actions),
            alpha_xy=alpha_xy,
        ),
        vision_feat_dim=vision_feat_dim,
    ).to(device).eval()
    msd = ck.get('model_state', None)
    if msd:
        model.load_state_dict(msd, strict=False)

    # Data
    if cfg.infer_json is None:
        raise ValueError('--infer_json is required for inference')
    vds = JsonTrackingDataset(DataConfig(train_json=cfg.infer_json, n_waypoints=n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
    vdl = DataLoader(vds, batch_size=cfg.batch_size, shuffle=False, num_workers=min(2, cfg.num_workers), pin_memory=True, collate_fn=collate_batch)

    # Output dirs
    os.makedirs(cfg.infer_out, exist_ok=True)
    vis_dir = os.path.join(cfg.infer_out, 'vis')
    npz_dir = os.path.join(cfg.infer_out, 'npz')
    if cfg.infer_vis:
        os.makedirs(vis_dir, exist_ok=True)
    if cfg.infer_save_npz:
        os.makedirs(npz_dir, exist_ok=True)

    batches_limit = max(0, int(cfg.infer_batches))
    bdone = 0
    for bidx, batch in enumerate(vdl):
        coarse_tokens = batch['coarse_tokens'].to(device)
        coarse_tidx   = batch['coarse_tidx'].to(device)
        fine_tokens   = batch['fine_tokens'].to(device)
        fine_tidx     = batch['fine_tidx'].to(device)
        yaw_hist      = batch['yaw_hist'].to(device)
        yaw_curr      = batch['yaw_curr'].to(device)
        instr         = batch['instruction']

        pred = model(
            coarse_tokens, coarse_tidx,
            fine_tokens, fine_tidx,
            instr,
            yaw_hist=yaw_hist if use_angle_tvi else None,
            yaw_curr=yaw_curr if use_angle_tvi else None
        )  # absolute space (alpha applied inside model)

        # Save NPZ
        if cfg.infer_save_npz:
            try:
                with torch.no_grad():
                    pred_np = pred.detach().float().cpu().numpy()
                Bcur = pred_np.shape[0]
                for bi in range(Bcur):
                    fpath = os.path.join(npz_dir, f"b{bidx:06d}_i{bi:03d}.npz")
                    np.savez_compressed(
                        fpath,
                        pred=pred_np[bi],
                        instruction=instr[bi],
                        current_path=(batch.get('current_path', [''])[bi] if isinstance(batch.get('current_path', []), list) else '')
                    )
            except Exception:
                pass

        # Visualization in absolute space (consistent with training vis and deployment)
        if cfg.infer_vis:
            try:
                with torch.no_grad():
                    # Ensure predictions are in absolute units for visualization
                    pred_draw = pred.detach().float()
                    try:
                        model_inspect = model
                        alpha_vec = getattr(model_inspect, 'alpha_task', None)
                        if alpha_vec is not None and pred_draw.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                            max_xy = pred_draw[..., 0:2].abs().max().item()
                            if max_xy <= 1.5:
                                ax = alpha_vec[..., 0:2].clamp_min(1e-6).to(pred_draw.device, pred_draw.dtype)
                                pred_draw = pred_draw.clone()
                                pred_draw[..., 0:2] = pred_draw[..., 0:2] * ax
                    except Exception:
                        pass
                    pred_np = pred_draw.cpu().numpy()
                cur_paths = batch.get('current_path', [])
                Bcur = pred_np.shape[0]
                for bi in range(min(Bcur, 4)):
                    cur_path = cur_paths[bi] if isinstance(cur_paths, list) and bi < len(cur_paths) else None
                    if cur_path is None or (not os.path.exists(cur_path)):
                        continue
                    pil_img = Image.open(cur_path).convert('RGB')
                    draw = ImageDraw.Draw(pil_img)
                    w, h = pil_img.size
                    base_x = w // 2
                    base_y = int(h * 0.86)
                    def to_pxxy(traj):
                        pts = []
                        for i in range(min(traj.shape[0], 64)):
                            x, y = float(traj[i, 0]), float(traj[i, 1])
                            px = base_x - int(y * 120)
                            py = base_y - int(x * 120)
                            pts.append((px, py))
                        return pts
                    pts_pred = to_pxxy(pred_np[bi])
                    for i2 in range(1, len(pts_pred)):
                        draw.line([pts_pred[i2-1], pts_pred[i2]], fill=(0, 255, 200), width=6)
                    if pts_pred:
                        r0 = 6
                        sx, sy = pts_pred[0]
                        draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(0,255,0))
                    out_path = os.path.join(vis_dir, f"b{bidx:06d}_i{bi:03d}.jpg")
                    pil_img.save(out_path)
            except Exception:
                pass

        bdone += 1
        if batches_limit and bdone >= batches_limit:
            break

    print(f"[INFER] Done. Outputs under {cfg.infer_out}")


# ----------------------- CLI -----------------------

def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_json', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default='./ckpts')
    ap.add_argument('--n_waypoints', type=int, default=8)
    ap.add_argument('--history', type=int, default=31)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--mixed_precision', action='store_true')
    ap.add_argument('--vision_feat_dim', type=int, default=1536)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--cache_root', type=str, default=None, help='Root folder containing vision_cache; defaults to <train_json>/vision_cache')
    ap.add_argument('--distributed', action='store_true')
    ap.add_argument('--dist_backend', type=str, default='nccl')
    ap.add_argument('--use_angle_tvi', action='store_true')  # default False
    ap.add_argument('--alpha_xy', type=float, default=2.0, help='Scalar to scale only XY targets; yaw unscaled')
    ap.add_argument('--beta_nav',      type=float, default=10.0)
    # logging / saving
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--csv_logging', action='store_true')
    ap.add_argument('--save_trajectories', action='store_true')
    ap.add_argument('--traj_subdir', type=str, default='trajectories')
    # evaluation
    ap.add_argument('--eval_every', type=int, default=0, help='Evaluate every N steps (0 = disabled)')
    ap.add_argument('--eval_batches', type=int, default=8, help='Number of validation batches per eval')
    ap.add_argument('--final_wp_threshold', type=float, default=0.2, help='Hit threshold (units of target XY)')
    # modeling
    ap.add_argument('--no_tanh_actions', action=argparse.BooleanOptionalAction, default=True, help='Remove tanh cap in action head (unbounded outputs)')
    # checkpoint retention
    ap.add_argument('--max_ckpts', type=int, default=3, help='Keep at most this many checkpoints')
    # resume
    ap.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint in out_dir (or --resume_ckpt)')
    ap.add_argument('--resume_ckpt', type=str, default=None, help='Path to a specific checkpoint to resume from')
    # inference
    ap.add_argument('--infer_json', type=str, default=None, help='Run inference on this dataset (json/jsonl/dir)')
    ap.add_argument('--infer_ckpt', type=str, default=None, help='Checkpoint to load for inference (defaults to latest in out_dir)')
    ap.add_argument('--infer_out', type=str, default='./infer_out', help='Output directory for inference results')
    ap.add_argument('--infer_batches', type=int, default=0, help='Limit number of batches to run at inference (0 = all)')
    ap.add_argument('--infer_vis', action='store_true', help='Save visualization images during inference')
    ap.add_argument('--infer_save_npz', action='store_true', help='Save npz predictions during inference')
    # single-episode evaluation
    ap.add_argument('--episode_json', type=str, default=None, help='JSON/JSONL path for a single episode sequence to evaluate')
    ap.add_argument('--episode_eval_every', type=int, default=0, help='Evaluate the episode every N steps (0 = disabled)')
    ap.add_argument('--episode_threshold', type=float, default=0.2, help='Following threshold radius for episode eval')
    ap.add_argument('--episode_max_frames', type=int, default=256, help='Maximum frames to evaluate in the episode')
    ap.add_argument('--llm_name', type=str, default='Qwen/Qwen3-0.6B', help='HuggingFace model id for the LLM backbone')

    args = ap.parse_args()
    return TrainConfig(**vars(args))


if __name__ == '__main__':
    cfg = parse_args()
    # Inference-only mode: if infer_json is provided and epochs==0, run inference with a loaded model
    if cfg.infer_json and cfg.epochs == 0:
        _run_inference(cfg)
    else:
        train(cfg)
