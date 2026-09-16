import pytest
import torch

from cyberworld.runtime import select_device


def test_cpu_device_is_always_available() -> None:
    info = select_device("cpu")
    assert info.device == torch.device("cpu")
    assert not info.mixed_precision_supported


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        select_device("quantum")


def test_unavailable_explicit_accelerator_fails() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            select_device("cuda")
