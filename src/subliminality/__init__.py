"""subliminality: research library for studying subliminal learning-like effects in context.

This package is the single source of truth for project logic. Scripts and
notebooks should import from here rather than re-implementing functionality.
"""

from subliminality.device import get_device, seed_everything
from subliminality.entanglement import compute_entanglements
from subliminality.tokens import (
    SENTINEL,
    build_input_ids,
    first_token,
    is_number,
    token_mask,
    top_bottom,
)

__version__ = "0.1.0"

__all__ = [
    "get_device",
    "seed_everything",
    "compute_entanglements",
    "SENTINEL",
    "build_input_ids",
    "first_token",
    "is_number",
    "token_mask",
    "top_bottom",
]
