#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Optional

import torch
from transformers import PreTrainedModel

from model import ModelConfig, OpenTrackVLA
from .configuration_open_trackvla import OpenTrackVLAConfig


class OpenTrackVLAForWaypoint(PreTrainedModel):
    """
    HuggingFace-compatible wrapper around the native OpenTrackVLA planner.
    This module enables `from_pretrained` / `save_pretrained` semantics while
    delegating the actual forward pass to the existing `model.OpenTrackVLA`.
    """

    config_class = OpenTrackVLAConfig

    def __init__(self, config: OpenTrackVLAConfig):
        super().__init__(config)
        nav_cfg = ModelConfig(
            llm_name=config.llm_name,
            freeze_llm=config.freeze_llm,
            n_waypoints=config.n_waypoints,
            max_time=config.max_time,
            beta_nav=config.beta_nav,
            use_angle_tvi=config.use_angle_tvi,
            use_tanh_actions=config.use_tanh_actions,
            alpha_xy=config.alpha_xy,
        )
        self.model = OpenTrackVLA(nav_cfg, vision_feat_dim=config.vision_feat_dim)
        self._register_load_state_dict_pre_hook(self._maybe_prefix_state_dict)
        self.post_init()

    def forward(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        instructions: List[str],
        yaw_hist: Optional[torch.Tensor] = None,
        yaw_curr: Optional[torch.Tensor] = None,
        bbox_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            coarse_tokens,
            coarse_tidx,
            fine_tokens,
            fine_tidx,
            instructions,
            yaw_hist=yaw_hist,
            yaw_curr=yaw_curr,
            bbox_feat=bbox_feat,
        )

    @property
    def tokenizer(self):
        return getattr(self.model, "tokenizer", None)

    def _maybe_prefix_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Retrofit checkpoints saved before we added the `model.` prefix."""
        # If keys already have the correct prefix, nothing to do.
        target_prefix = f"{prefix}model."
        if any(k.startswith(target_prefix) for k in state_dict.keys()):
            return
        patched = {}
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            new_key = f"{target_prefix}{key[len(prefix):]}"
            patched[new_key] = state_dict.pop(key)
        state_dict.update(patched)

