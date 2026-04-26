#!/usr/bin/env python3
from typing import Optional

from transformers import AutoConfig, AutoModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING

from .configuration_open_trackvla import OpenTrackVLAConfig
from .modeling_open_trackvla import OpenTrackVLAForWaypoint

__all__ = ["OpenTrackVLAConfig", "OpenTrackVLAForWaypoint"]


def _register_with_auto():
    model_type = OpenTrackVLAConfig.model_type
    if model_type not in CONFIG_MAPPING:
        CONFIG_MAPPING.register(model_type, OpenTrackVLAConfig)
    try:
        AutoConfig.register(model_type, OpenTrackVLAConfig)
    except Exception:
        pass
    if OpenTrackVLAConfig not in MODEL_MAPPING:
        MODEL_MAPPING.register(OpenTrackVLAConfig, OpenTrackVLAForWaypoint)
    try:
        AutoModel.register(OpenTrackVLAConfig, OpenTrackVLAForWaypoint)
    except Exception:
        pass


_register_with_auto()

