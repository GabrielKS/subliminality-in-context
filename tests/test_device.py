import torch

from subliminality import get_device
from subliminality.device import DEVICE_ENV_VAR


def test_get_device_returns_supported_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "mps", "cpu"}


def test_cpu_is_always_selectable():
    # CPU is the universal fallback and must always be honored when requested.
    assert get_device(prefer="cpu").type == "cpu"


def test_unavailable_preference_falls_through():
    # Requesting an unavailable device must not raise; it falls back to auto.
    device = get_device(prefer="definitely-not-a-device")
    assert device.type in {"cuda", "mps", "cpu"}


def test_env_var_override(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
    assert get_device().type == "cpu"
