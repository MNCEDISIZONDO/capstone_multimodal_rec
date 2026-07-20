"""Device management and CUDA OOM fallback for the capstone multimodal recommender.

Every memory-intensive operation in this project routes through run_with_fallback,
so a VRAM spike degrades the run to CPU instead of killing it.
"""

from __future__ import annotations

import gc
from typing import Any, Callable

import torch
from torch import nn


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the GPU if one is usable, otherwise the CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_gpu_memory() -> None:
    """Release cached GPU blocks. Call after every batch and before a retry."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_report() -> str:
    """One-line VRAM summary, useful in logs and when diagnosing an OOM."""
    if not torch.cuda.is_available():
        return "CUDA unavailable (CPU only)"
    free, total = torch.cuda.mem_get_info()
    to_gb = lambda b: b / 1024 ** 3
    return (
        f"VRAM total {to_gb(total):.2f} GB | free {to_gb(free):.2f} GB | "
        f"allocated {to_gb(torch.cuda.memory_allocated()):.2f} GB | "
        f"reserved {to_gb(torch.cuda.memory_reserved()):.2f} GB"
    )


def _to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors and modules inside nested containers."""
    if isinstance(obj, (torch.Tensor, nn.Module)):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def run_with_fallback(fn: Callable, *args, **kwargs) -> Any:
    """Run fn on the current device; on CUDA OOM, retry the whole thing on CPU.

    The caller does not need to know which device did the work.
    """
    try:
        return fn(*args, **kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
        if "out of memory" not in str(err).lower():
            raise  # a real error, not a memory problem — do not hide it
        print("[device_utils] CUDA OOM caught — falling back to CPU.")
        print(f"[device_utils] {vram_report()}")
        clear_gpu_memory()
        cpu = torch.device("cpu")
        result = fn(*_to_device(args, cpu), **_to_device(kwargs, cpu))
        print("[device_utils] CPU fallback completed.")
        return result