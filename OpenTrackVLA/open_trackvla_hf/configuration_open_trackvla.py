#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

from transformers import PretrainedConfig


class OpenTrackVLAConfig(PretrainedConfig):
    """
    Minimal HuggingFace configuration wrapper for the OpenTrackVLA planner.
    This mirrors the fields consumed by `model.ModelConfig` so checkpoints
    converted via `convert_ckpt_to_hf.py` can be loaded with
    `OpenTrackVLAForWaypoint.from_pretrained(...)`.
    """

    model_type = "navfom"

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen3-0.6B",
        freeze_llm: bool = True,
        n_waypoints: int = 8,
        max_time: int = 4096,
        beta_nav: float = 10.0,
        use_angle_tvi: bool = False,
        use_tanh_actions: bool = True,
        alpha_xy: Optional[float] = None,
        vision_feat_dim: int = 1536,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_name = llm_name
        self.freeze_llm = freeze_llm
        self.n_waypoints = n_waypoints
        self.max_time = max_time
        self.beta_nav = beta_nav
        self.use_angle_tvi = use_angle_tvi
        self.use_tanh_actions = use_tanh_actions
        self.alpha_xy = alpha_xy
        self.vision_feat_dim = vision_feat_dim







