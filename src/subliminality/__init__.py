"""subliminality: research library for studying subliminal learning-like effects in context.

This package is the single source of truth for project logic. Scripts and
notebooks should import from here rather than re-implementing functionality.
"""

from subliminality.device import get_device, seed_everything
from subliminality.entanglement import compute_entanglements
from subliminality.generation import batched_answer_probs, batched_generate_truncated
from subliminality.reasoning_generation import (
    DEFAULT_ANSWER_FORMAT,
    DEFAULT_ANSWER_SCAFFOLD,
    DEFAULT_THINK_BUDGET,
    AnswerScores,
    CoTRollout,
    answer_candidate_token,
    answer_scaffold_ids,
    answer_tokens_collide,
    batched_answer_scores,
    build_boxed_answer_instruction,
    build_reasoning_prompt,
    rollout_cot,
)
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
    "batched_answer_probs",
    "batched_generate_truncated",
    "build_reasoning_prompt",
    "build_boxed_answer_instruction",
    "rollout_cot",
    "CoTRollout",
    "batched_answer_scores",
    "AnswerScores",
    "answer_candidate_token",
    "answer_scaffold_ids",
    "answer_tokens_collide",
    "DEFAULT_THINK_BUDGET",
    "DEFAULT_ANSWER_SCAFFOLD",
    "DEFAULT_ANSWER_FORMAT",
    "SENTINEL",
    "build_input_ids",
    "first_token",
    "is_number",
    "token_mask",
    "top_bottom",
]
