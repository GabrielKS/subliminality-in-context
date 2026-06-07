"""Naïve token-entanglement metrics.

Two ways to ask "which vocabulary tokens are entangled with a given token":

* ``"unembedding"`` — cosine similarity of unembedding-matrix rows (no forward pass).
* ``"output_distribution"`` — how the next-token distribution shifts when the token
  is injected as the model's favorite (one forward pass per query token).

Both return entanglement against the *whole* vocabulary, adding a trailing vocab
dimension to the input ``tokens``.
"""

import torch
import torch.nn.functional as F

from subliminality.tokens import SENTINEL, build_input_ids

# --- Prompt assumptions (edit here, or override per call via `prompt=`) -------
BASE_MESSAGES = [
    {"role": "user", "content": "What is your favorite token?"},
    {"role": "assistant", "content": "My favorite token is:\n"},
]
GUIDED_MESSAGES = [
    {"role": "system", "content": f"Your favorite token is{SENTINEL}."},
    *BASE_MESSAGES,
]


def _default_prompt(tokenizer, target_id: int | None = None) -> list[int]:
    "Chat-template prompt builder: the guided prompt if a token is given, else the base prompt."
    messages = GUIDED_MESSAGES if target_id is not None else BASE_MESSAGES
    return build_input_ids(messages, target_id, tokenizer=tokenizer)


def compute_entanglements(model, tokens, *, method: str, tokenizer=None,
                          prompt=_default_prompt, return_components: bool = False):
    """Entanglement of each token in ``tokens`` against the whole vocabulary.

    Args:
        model: a causal-LM whose ``get_output_embeddings().weight`` / ``logits`` we read.
        tokens: token ids of any shape ``[...]`` (int, list, or tensor).
        method: ``"unembedding"`` or ``"output_distribution"`` (required).
        tokenizer: required for ``"output_distribution"`` (to build prompts).
        prompt: builder ``(tokenizer, target_id=None) -> list[int]`` for the guided
            (token given) and base (token ``None``) prompts.
        return_components: ``"output_distribution"`` only; see below.

    Returns:
        ``"unembedding"``: a ``[..., vocab]`` tensor of cosine similarities.
        ``"output_distribution"``: by default the ``[..., vocab]`` entanglement ratio
        ``guided / base`` (same shape as ``"unembedding"``); with
        ``return_components=True``, the underlying ``(guided [..., vocab], base [vocab])``.
    """
    tokens = torch.as_tensor(tokens)
    if method == "unembedding":
        return _unembedding(model, tokens)
    if method == "output_distribution":
        if tokenizer is None:
            raise ValueError("method='output_distribution' requires a tokenizer")
        guided, base = _output_distribution(model, tokens, tokenizer, prompt)
        return (guided, base) if return_components else guided / base
    raise ValueError(f"unknown method {method!r}")


def _unembedding(model, tokens) -> torch.Tensor:
    W = model.get_output_embeddings().weight  # [vocab, hidden]
    with torch.no_grad():
        unit = F.normalize(W.float(), dim=-1)
        rows = F.normalize(W[tokens.to(W.device)].float(), dim=-1)  # [..., hidden]
        return rows @ unit.T  # [..., vocab]


def _output_distribution(model, tokens, tokenizer, prompt):
    flat = tokens.reshape(-1).tolist()
    guided_ids = torch.tensor([prompt(tokenizer, t) for t in flat], device=model.device)  # [N, L]
    base_ids = torch.tensor([prompt(tokenizer)], device=model.device)
    with torch.no_grad():
        # float32 softmax: the signal lives in tiny tail probabilities
        guided = model(guided_ids).logits[:, -1, :].float().softmax(dim=-1)  # [N, vocab]
        base = model(base_ids).logits[0, -1, :].float().softmax(dim=-1)  # [vocab]
    return guided.reshape(*tokens.shape, -1), base
