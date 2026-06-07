"""Tokenizer-level primitives shared across experiments.

These are deliberately model-agnostic: pass any Hugging Face tokenizer.
"""

from collections.abc import Callable

import torch

#: Private-use codepoint used to mark an exact-token injection point. It never
#: appears in normal text, so we can render a prompt to a string and splice a
#: token id in at this marker without BPE re-tokenization mangling it.
SENTINEL = chr(0xE000)


def build_input_ids(messages: list[dict], target_id: int | None = None, *, tokenizer) -> list[int]:
    """Render chat ``messages`` to token ids, splicing the *exact* ``target_id``
    in place of :data:`SENTINEL`.

    Splicing the id directly (rather than decoding it into the string and
    re-encoding) guarantees the token survives, which a BPE round-trip does not.
    ``add_special_tokens=False`` because the rendered template already carries BOS.
    """
    text = tokenizer.apply_chat_template(
        messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    if target_id is None:
        assert SENTINEL not in text
        return tokenizer(text, add_special_tokens=False).input_ids
    assert text.count(SENTINEL) == 1, "exactly one injection point required"
    left, right = text.split(SENTINEL)
    return (tokenizer(left, add_special_tokens=False).input_ids
            + [int(target_id)]
            + tokenizer(right, add_special_tokens=False).input_ids)


def first_token(text: str, tokenizer) -> int:
    "First token id of ``text`` -- a single-token handle for a word or phrase."
    return tokenizer.encode(text, add_special_tokens=False)[0]


def is_number(s: str) -> bool:
    "True if ``s`` stripped of surrounding whitespace is a non-empty run of 0-9."
    s = s.strip()
    return s != "" and all(c in "0123456789" for c in s)


def token_mask(tokenizer, predicate: Callable[[str], bool]) -> torch.Tensor:
    "Boolean mask over the vocab: ``predicate`` applied to each decoded token."
    ids = torch.arange(len(tokenizer))
    return torch.tensor([predicate(s) for s in tokenizer.batch_decode(ids.unsqueeze(-1))])


def top_bottom(scores: torch.Tensor, tokenizer, topk: int = 5, bottomk: int = 5,
               mask: torch.Tensor | None = None):
    """Top-``topk`` and bottom-``bottomk`` tokens of a 1-D ``scores`` tensor.

    ``mask`` (a boolean vocab tensor) restricts eligibility. Returns
    ``(top, bottom)`` lists of ``(token_str, score)`` pairs; either side is
    ``None`` when its k is 0.
    """
    scores = scores.reshape(-1)
    keep = None if mask is None else mask.to(scores.device)

    def select(k: int, largest: bool):
        if k == 0:
            return None
        # push filtered-out tokens to the far end so they never enter the k we keep
        s = scores if keep is None else scores.masked_fill(~keep, float("-inf") if largest else float("inf"))
        vals, idx = s.topk(k, largest=largest)
        return [(tokenizer.convert_ids_to_tokens([int(i)])[0], v.item()) for v, i in zip(vals, idx)]

    return select(topk, largest=True), select(bottomk, largest=False)
