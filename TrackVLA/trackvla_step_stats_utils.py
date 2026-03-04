
"""
TrackVLA/EVT-Bench step-level profiling + metrics logging utilities.

Design goals:
- Zero dependency beyond Python stdlib (+ optional torch for CUDA timing).
- Safe for multi-process eval: each process writes its own JSONL file.
- Records per-step wall time breakdown and per-episode SR/TR/CR-compatible fields.

Usage: used by agent_uninavid.evaluate_agent patched version.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class JSONLWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class WallTimer:
    """Simple wall-clock timer, returns milliseconds."""
    def __init__(self) -> None:
        self.t0 = _now_ms()

    def ms(self) -> float:
        return _now_ms() - self.t0


def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        return d.get(key, default)
    except Exception:
        return default


def hz_from_ms(total_ms: float) -> Optional[float]:
    if total_ms is None:
        return None
    if total_ms <= 0:
        return None
    return 1000.0 / total_ms


def percentile(sorted_list: List[float], q: float) -> Optional[float]:
    if not sorted_list:
        return None
    if q <= 0:
        return float(sorted_list[0])
    if q >= 100:
        return float(sorted_list[-1])
    k = (len(sorted_list) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_list) - 1)
    if f == c:
        return float(sorted_list[f])
    d0 = sorted_list[f] * (c - k)
    d1 = sorted_list[c] * (k - f)
    return float(d0 + d1)


def summarize_ms(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None}
    vals.sort()
    mean = sum(vals) / len(vals)
    return {
        "count": len(vals),
        "mean": mean,
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
        "min": float(vals[0]),
        "max": float(vals[-1]),
    }


def maybe_cuda_timer():
    """
    Returns (start, end, sync_fn, enabled) where start/end are callables that
    record CUDA events around a region. If torch/cuda is unavailable, it's disabled.
    """
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return None, None, None, False
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        def start():
            start_evt.record()

        def end():
            end_evt.record()

        def sync_and_elapsed_ms() -> float:
            torch.cuda.synchronize()
            return float(start_evt.elapsed_time(end_evt))

        return start, end, sync_and_elapsed_ms, True
    except Exception:
        return None, None, None, False
