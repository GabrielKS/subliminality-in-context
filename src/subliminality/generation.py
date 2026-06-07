"""Batched generation/reading primitives for variable-length prompts.

These exist for measurement conditions where prompts differ in length and so
cannot be stacked into a single equal-length batch the way
:func:`subliminality.entanglement.compute_entanglements` does (its SENTINEL
splice keeps every prompt the same length). Here we **left-pad** instead, which
lets us:

* read ``P(target)`` at position ``-1`` for a whole batch in one forward
  (:func:`batched_answer_probs`), and
* run one batched ``generate`` for many prompts and truncate each output at a
  stop token (:func:`batched_generate_truncated`).

Both are domain-agnostic (no experiment-specific prompts) and device-agnostic
(they read ``model.device``; no hardcoded CUDA).

Left-padding correctness: we pass ``position_ids`` derived from the attention
mask (``cumsum - 1``) so the real tokens are numbered ``0, 1, 2, …`` regardless
of how much padding precedes them. This matters for models with *learned*
positional embeddings (e.g. GPT-2): without it the real tokens would pick up
shifted position embeddings and the padded result would diverge from the
unpadded one. RoPE models (Llama/DeepSeek) are insensitive to a constant shift,
but passing position_ids is correct for both.
"""

from collections.abc import Sequence

import torch

from subliminality.device import seed_everything


def _left_pad(rows: Sequence[Sequence[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad ragged token-id ``rows`` to a common length.

    Returns ``(input_ids, attention_mask)`` tensors on ``device``; the mask is 0
    over the left padding and 1 over the real tokens.
    """
    maxlen = max(len(r) for r in rows)
    input_rows, mask_rows = [], []
    for r in rows:
        pad = maxlen - len(r)
        input_rows.append([pad_id] * pad + list(r))
        mask_rows.append([0] * pad + [1] * len(r))
    return (torch.tensor(input_rows, device=device),
            torch.tensor(mask_rows, device=device))


def batched_answer_probs(model, prefixes, target: int, *, pad_id: int,
                         answer_ids: Sequence[int] = ()) -> list[float]:
    """``P(target)`` at the token right after each prefix (+ a shared answer).

    Each item in ``prefixes`` is a variable-length list of token ids. They are
    left-padded to a common length, the shared ``answer_ids`` (possibly empty) is
    appended to every row, and a single forward pass reads the next-token
    distribution at position ``-1`` (the real last token, thanks to left-padding).
    Softmax is taken in float32 — bf16 underflows the tiny tail probabilities we
    care about.

    Args:
        model: a causal LM.
        prefixes: list of token-id sequences (the context up to the read point).
        target: the vocabulary id whose probability we read.
        pad_id: padding token id (typically ``tokenizer.eos_token_id``).
        answer_ids: optional shared continuation appended to every prefix before
            the read (e.g. the teacher-forced answer prefix). May be empty.

    Returns:
        One ``P(target)`` per prefix, in order.
    """
    answer = list(answer_ids)
    input_ids, attn = _left_pad([list(p) + answer for p in prefixes], pad_id, model.device)
    position_ids = (attn.long().cumsum(-1) - 1).clamp(min=0)  # real tokens -> 0,1,2,...
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn,
                       position_ids=position_ids).logits[:, -1, :].float()
    return logits.softmax(dim=-1)[:, target].tolist()


def batched_generate_truncated(model, prompts, *, stop_id: int, pad_id: int,
                               gen_kwargs: dict, n_samples: int = 1,
                               seed: int = 0) -> list[list[int]]:
    """Left-padded batched ``generate``, each output truncated at ``stop_id``.

    All ``prompts`` are generated in one batched call (with
    ``num_return_sequences=n_samples``), then each output is cut to its real
    prompt tokens plus generated tokens up to and **including** the first
    ``stop_id`` (left padding and any post-stop padding dropped). If a sequence
    never emits ``stop_id`` (e.g. it hit real EOS or the length cap) its full
    generation is kept.

    Generation is seeded via :func:`subliminality.seed_everything` so the batch is
    reproducible; note the draws differ from generating each prompt separately.

    Args:
        model: a causal LM.
        prompts: list of token-id sequences to continue.
        stop_id: token at which to truncate each generated continuation.
        pad_id: padding token id (also passed as ``pad_token_id``).
        gen_kwargs: forwarded to ``model.generate`` (e.g. ``do_sample``,
            ``temperature``, ``top_p``, ``max_new_tokens``, ``eos_token_id``).
            Supply ``eos_token_id`` here if generation should also stop on model
            EOS — ``stop_id`` only controls truncation, not stopping.
        n_samples: traces per prompt (``num_return_sequences``).
        seed: RNG seed applied before the batched generate.

    Returns:
        ``len(prompts) * n_samples`` token-id lists, ordered prompt-major then
        sample (HF's ``num_return_sequences`` layout: prompt ``i`` occupies the
        slice ``[i*n_samples : (i+1)*n_samples]``).
    """
    input_ids, attn = _left_pad([list(p) for p in prompts], pad_id, model.device)
    maxlen = input_ids.shape[1]
    seed_everything(seed)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attn,
                             num_return_sequences=n_samples, pad_token_id=pad_id,
                             **gen_kwargs)
    results = []
    for i, seq in enumerate(out.tolist()):
        gen = seq[maxlen:]                       # generated tokens follow the padded prompt
        if stop_id in gen:
            gen = gen[: gen.index(stop_id) + 1]  # keep through the first stop
        results.append(list(prompts[i // n_samples]) + gen)
    return results
