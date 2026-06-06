"""Device selection for running on CUDA, Apple Silicon (MPS), or CPU.

This module is the single source of truth for choosing a torch device. Scripts
and notebooks should call :func:`get_device` rather than detecting the device
themselves, so device handling stays consistent and portable across machines.

Note on unsupported MPS ops: to let missing operations fall back to CPU on
Apple Silicon, set ``PYTORCH_ENABLE_MPS_FALLBACK=1`` in the environment *before*
the process starts (it cannot be set reliably from within Python after torch is
imported).
"""

import os

import torch

#: Environment variable that, if set, overrides automatic device selection.
DEVICE_ENV_VAR = "SUBLIMINALITY_DEVICE"


def _is_available(kind: str) -> bool:
    """Return whether a device *kind* (``"cuda"``/``"mps"``/``"cpu"``) is usable."""
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        return torch.backends.mps.is_available()
    if kind == "cpu":
        return True
    return False


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Resolution order:

    1. The ``prefer`` argument, if given and available.
    2. The ``SUBLIMINALITY_DEVICE`` environment variable, if set and available.
    3. Automatic preference: CUDA, then Apple Silicon MPS, then CPU.

    A requested device that is unavailable is skipped rather than raising, so the
    same code runs unchanged across CUDA, MPS, and CPU machines.

    Args:
        prefer: Optional device string such as ``"cuda"``, ``"mps"``, ``"cpu"``,
            or an indexed variant like ``"cuda:0"``. Ignored if unavailable.

    Returns:
        The selected :class:`torch.device`.
    """
    for candidate in (prefer, os.environ.get(DEVICE_ENV_VAR)):
        if candidate:
            kind = candidate.split(":", 1)[0]
            if _is_available(kind):
                return torch.device(candidate)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
