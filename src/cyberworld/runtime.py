"""Cross-platform device and reproducibility utilities."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    accelerator: str
    mixed_precision_supported: bool


def select_device(requested: str = "auto") -> DeviceInfo:
    """Resolve auto/cuda/mps/cpu and fail clearly for unavailable explicit devices."""
    requested = requested.lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device {requested!r}; use auto, cuda, mps, or cpu")

    cuda_available = torch.cuda.is_available()
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not mps_available:
        raise RuntimeError("Apple Metal (MPS) was requested but is not available")

    if requested == "cuda" or (requested == "auto" and cuda_available):
        return DeviceInfo(torch.device("cuda"), "cuda", True)
    if requested == "mps" or (requested == "auto" and mps_available):
        return DeviceInfo(torch.device("mps"), "mps", False)
    return DeviceInfo(torch.device("cpu"), "cpu", False)


def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.set_float32_matmul_precision("high")
