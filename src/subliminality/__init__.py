"""subliminality: research library for studying subliminal learning-like effects in context.

This package is the single source of truth for project logic. Scripts and
notebooks should import from here rather than re-implementing functionality.
"""

from subliminality.core import greet
from subliminality.device import get_device

__version__ = "0.1.0"

__all__ = ["greet", "get_device"]
