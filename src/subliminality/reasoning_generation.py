"""Reasoning-model (think-block) generation + answer-reading primitives.

Where :mod:`subliminality.generation` holds the model-agnostic batched
``generate`` / read mechanics, this module holds the *reasoning-model* mechanics
the SCoT sprint needs: building a think-block prompt (optionally seeded with a
CoT **prefill**), rolling out a chain of thought with a token budget and a
**force-close** of the thinking block, and reading the logprobs that decide a
model's confidence between the **two candidate answers** of a comparison
question.

It is deliberately:

* **Domain-agnostic** — no questions/entities/animals; callers supply the user
  turn, the answer scaffold, and the candidate token ids.
* **Model-configurable** — the closing-think token id (``end_think_id``) is always
  a parameter (resolve it once per model via
  ``tokenizer.convert_tokens_to_ids("</think>")``), never hardcoded to DeepSeek's
  ``128014``.
* **Device-agnostic** — everything reads ``model.device``; no hardcoded CUDA.

The two SCoT use cases both reduce to *build prompt → rollout → read scores*:

1. **Baseline / caching** — ``build_reasoning_prompt(tok, question)`` (no prefill),
   ``rollout_cot(...)`` to generate + force-close the trace, then
   ``batched_answer_scores(...)`` to cache the two-answer logits/logprobs.
2. **Regenerate-from-prefill** — the same, but ``build_reasoning_prompt(..., prefill=...)``
   seeds the (truncated + injected) partial CoT, as a string or exact token ids.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from subliminality.generation import _left_pad, batched_generate_truncated

#: Default CoT token budget. SCOTSPRINT's "affordable floor" for R1-Distill — many
#: traces won't close on their own, so report results as "think for up to N tokens"
#: and keep the force-close rate visible (see :class:`CoTRollout`).
DEFAULT_THINK_BUDGET = 1024

#: DeepSeek R1-Distill's recommended sampling for *generation* (irrelevant to the
#: logit probes, which read the raw distribution). Merged-under by ``rollout_cot``.
DEFAULT_GEN_KWARGS = {"do_sample": True, "temperature": 0.6, "top_p": 0.95}

#: Recommended teacher-forced answer scaffold, appended after the closed ``</think>``
#: block before reading the answer (DeepSeek's "put your final answer within
#: ``\boxed{}``" convention). The trailing ``{`` tokenizes as its own token, giving a
#: hard *no-space* boundary: the very next token is the entity's first token with no
#: leading space and its natural casing. Empirically (R1-Distill-Llama-8B) the model
#: places ~0.999 of its mass on that single entity token here, with the competing
#: entity ~8 logits below — exactly the signal :func:`batched_answer_scores` reads.
#: Always derive the candidate token in this same context via
#: :func:`answer_candidate_token` (never assume a leading space / casing).
DEFAULT_ANSWER_SCAFFOLD = "\n\n\\boxed{"

def _chunks(seq, size):
    "Yield successive ``size``-length slices of ``seq``."
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _maybe_tqdm(iterable, progress: bool, desc: str):
    "Wrap ``iterable`` in a tqdm bar when ``progress``; tqdm is imported lazily."
    if not progress:
        return iterable
    from tqdm.auto import tqdm
    return tqdm(iterable, desc=desc)


#: ``str.format`` template rendering one answer option as it should appear in the
#: prompt instruction (``build_boxed_answer_instruction``). Kept in sync with
#: :data:`DEFAULT_ANSWER_SCAFFOLD`: stripped of leading whitespace the scaffold is a
#: prefix of every rendered option (both are the ``\boxed{…}`` convention), so the
#: token the model is *instructed* to write is exactly the one we later teacher-force
#: and read. Change them together (the instruction builder validates this).
DEFAULT_ANSWER_FORMAT = "\\boxed{{{}}}"


def build_reasoning_prompt(tokenizer, user_content: str, *, prefill=None,
                           system: str | None = None) -> list[int]:
    """Token ids for a reasoning prompt opened at the thinking block.

    Renders the chat template with ``add_generation_prompt=True`` (so it ends right
    after the model's opened ``<think>``) and encodes with
    ``add_special_tokens=False`` (the template already carries BOS — re-adding it
    would double it). An optional ``prefill`` — a partial chain of thought to
    continue from — is appended verbatim:

    * ``None`` — bare opened-think prompt (a baseline rollout generates the whole
      trace).
    * ``str`` — encoded with ``add_special_tokens=False`` and appended.
    * ``Sequence[int]`` — appended as **exact** token ids. Use this for
      injected-token prefills: decoding ids to text and re-encoding does not
      reliably preserve every token (the same hazard :func:`build_input_ids`
      avoids), so hand exact ids when the prefill carries spliced tokens.

    ``system`` defaults to ``None`` to honor reasoning models' "instructions go in
    the user turn, no system prompt" convention, but remains a parameter so the
    primitive isn't locked to that.
    """
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if prefill is None:
        return ids
    if isinstance(prefill, str):
        return ids + tokenizer(prefill, add_special_tokens=False).input_ids
    return ids + [int(t) for t in prefill]


def build_boxed_answer_instruction(question: str, options: Sequence[str], *,
                                   lead_in: str = "Please reason step by step before outputting either",
                                   answer_format: str = DEFAULT_ANSWER_FORMAT,
                                   scaffold: str = DEFAULT_ANSWER_SCAFFOLD) -> str:
    """A user-turn string that constrains the answer to one of ``options``.

    Returns ``"{question} {lead_in} <opt₀> or <opt₁> …."`` where each option is
    rendered via ``answer_format`` (default ``\\boxed{{…}}``) — e.g.::

        "Which is larger: France or Monaco? Please reason step by step before
         outputting either \\boxed{France} or \\boxed{Monaco}."

    This is the reasoning-model analogue of Arcuschin et al.'s "give a YES / NO
    answer": it makes the model emit one of the named options **verbatim** inside the
    box, so the boxed answer's first token matches the candidate read at the
    :data:`DEFAULT_ANSWER_SCAFFOLD` read point (it fixes long names being boxed as a
    short surface form). Pass ``options`` in the question's presentation order (not
    correct/incorrect) to avoid leaking a position cue; it only constrains the final
    token, leaving the chain of thought free.

    ``answer_format`` and ``scaffold`` are kept coupled: stripped of leading
    whitespace the scaffold must be a prefix of a rendered option, else the model
    would be told to write a delimiter different from the one teacher-forced at the
    read — a :class:`ValueError` is raised. This helper targets ``\\boxed{``-style
    scaffolds; a non-box scaffold needs a matching ``answer_format``.
    """
    if not answer_format.format("").startswith(scaffold.lstrip()):
        raise ValueError(
            f"answer_format {answer_format!r} is inconsistent with scaffold {scaffold!r}: "
            "a rendered option must start with the (whitespace-stripped) read scaffold")
    rendered = " or ".join(answer_format.format(o) for o in options)
    return f"{question} {lead_in} {rendered}."


@dataclass(frozen=True)
class CoTRollout:
    """One generated (and closed) chain of thought.

    ``think_ids`` is the generated continuation, **always** ending in
    ``end_think_id`` — either because the model emitted it naturally or because we
    appended it (``forced_close``). ``full_ids`` (``prompt_ids + think_ids``) is the
    closed-block prefix to hand to :func:`batched_answer_scores`.
    """
    prompt_ids: tuple[int, ...]
    think_ids: tuple[int, ...]
    forced_close: bool
    think_text: str

    @property
    def full_ids(self) -> list[int]:
        "Prompt followed by the closed think block; the answer-read prefix."
        return [*self.prompt_ids, *self.think_ids]


def rollout_cot(model, tokenizer, prompts, *, end_think_id: int,
                max_new_tokens: int = DEFAULT_THINK_BUDGET, n_samples: int = 1,
                seed: int = 0, gen_kwargs: dict | None = None,
                pad_id: int | None = None, batch_size: int | None = None,
                progress: bool = False) -> list[CoTRollout]:
    """Batched CoT rollout with a token budget and a force-closed think block.

    Wraps :func:`subliminality.generation.batched_generate_truncated` (one
    left-padded batched ``generate``, truncated at ``end_think_id``) and adds the
    **force-close** practice: any trace that hit the budget (or EOS) before emitting
    ``end_think_id`` gets it appended, so every trace is read in a *closed* block —
    which both fixes the otherwise-malformed open-block read and, empirically,
    strengthens the measured effect (see ``CLAUDE.md`` → "Reasoning models").

    ``gen_kwargs`` is merged over :data:`DEFAULT_GEN_KWARGS` plus
    ``max_new_tokens`` and ``eos_token_id=[end_think_id, eos]`` (so generation both
    stops and truncates at the closing think token); pass explicit keys to override.
    ``pad_id`` defaults to ``tokenizer.eos_token_id``.

    ``batch_size`` splits ``prompts`` into generation batches of that many prompts
    (each a separate ``generate`` call, so peak memory is bounded by the batch, not
    the whole set), concatenating the results in order; ``None`` runs one batch.
    Generation is **per-chunk seeded** at ``seed + chunk_index`` so the whole run is
    reproducible (batched seeding already differs from per-prompt — see
    :func:`batched_generate_truncated`). ``progress`` shows a ``tqdm`` bar over chunks.

    Returns ``len(prompts) * n_samples`` rollouts in HF's prompt-major-then-sample
    order (prompt ``i`` occupies ``[i*n_samples : (i+1)*n_samples]``). The
    force-close *rate* is left to the caller to log
    (``sum(r.forced_close for r in rollouts)``), matching the notebook's real-time +
    end-of-run reporting.
    """
    prompts = list(prompts)
    if batch_size is not None and len(prompts) > batch_size:
        out: list[CoTRollout] = []
        chunks = list(_chunks(prompts, batch_size))
        for i, chunk in enumerate(_maybe_tqdm(chunks, progress, "rollout_cot")):
            out.extend(rollout_cot(
                model, tokenizer, chunk, end_think_id=end_think_id,
                max_new_tokens=max_new_tokens, n_samples=n_samples,
                seed=seed + i, gen_kwargs=gen_kwargs, pad_id=pad_id))
        return out
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    kwargs = {
        **DEFAULT_GEN_KWARGS,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": [end_think_id, tokenizer.eos_token_id],
        **(gen_kwargs or {}),
    }
    seqs = batched_generate_truncated(
        model, prompts, stop_id=end_think_id, pad_id=pad_id,
        n_samples=n_samples, seed=seed, gen_kwargs=kwargs)
    rollouts = []
    for i, seq in enumerate(seqs):
        prompt = list(prompts[i // n_samples])
        gen = seq[len(prompt):]
        forced = end_think_id not in gen
        if forced:
            gen = gen + [end_think_id]
        rollouts.append(CoTRollout(
            prompt_ids=tuple(prompt),
            think_ids=tuple(gen),
            forced_close=forced,
            think_text=tokenizer.decode(gen, skip_special_tokens=False),
        ))
    return rollouts


@dataclass(frozen=True)
class AnswerScores:
    """The two-answer confidence read at a single answer position.

    ``logits``/``logprobs`` are aligned to ``candidate_ids`` (the caller's order).
    The ``*_diff`` properties follow ``candidate[0] - candidate[1]``; ordering the
    candidates as ``(correct, incorrect)`` makes ``logit_diff`` the SCOTSPRINT
    primary metric ``logit(correct) - logit(incorrect)`` — positive ⇒ the model
    favors the correct entity (a subliminal push toward the wrong entity lowers it).

    ``top_token_id`` is the **whole-vocabulary argmax** at the read point (with its
    ``top_logprob``), *independent of the candidates* — the token the model would
    actually emit if free. Use it for free-choice accuracy (does the model's top pick
    equal the correct answer, or even land on either candidate); ``top_token_id in
    candidate_ids`` need not hold. (Distinct from :attr:`argmax`, which is the index of
    the winning *candidate*.)
    """
    candidate_ids: tuple[int, ...]
    logits: tuple[float, ...]
    logprobs: tuple[float, ...]
    top_token_id: int
    top_logprob: float

    @property
    def logit_diff(self) -> float:
        "``logit(candidate[0]) - logit(candidate[1])`` (correct − incorrect, if so ordered)."
        return self.logits[0] - self.logits[1]

    @property
    def logprob_diff(self) -> float:
        "``logprob(candidate[0]) - logprob(candidate[1])``."
        return self.logprobs[0] - self.logprobs[1]

    @property
    def argmax(self) -> int:
        "Index of the highest-scoring *candidate* (the top-1 answer-flip handle)."
        return max(range(len(self.logprobs)), key=self.logprobs.__getitem__)


def batched_answer_scores(model, prefixes, candidates: Sequence[Sequence[int]], *,
                          pad_id: int, answer_ids: Sequence[int] = (),
                          batch_size: int | None = None,
                          progress: bool = False) -> list[AnswerScores]:
    """Raw logits + logprobs of each row's candidate tokens at the answer position.

    Mirrors :func:`subliminality.generation.batched_answer_probs` (left-pad +
    ``position_ids = cumsum-1`` + a float32 read at position ``-1``), but reads the
    **raw logits and log-softmax** of **per-row** candidate token ids rather than a
    single shared probability. The shared ``answer_ids`` scaffold
    (:func:`answer_scaffold_ids`, e.g. :data:`DEFAULT_ANSWER_SCAFFOLD`) is
    teacher-forced onto every row before the read; the read point is the next-token
    distribution right after it.

    ``candidates`` must be derived **in the same scaffold context** via
    :func:`answer_candidate_token` so each candidate is exactly the token the model
    would emit at the read point — do not pass a space-prefixed/guessed handle.

    Args:
        model: a causal LM.
        prefixes: per-row context token ids up to the read (e.g. a
            :attr:`CoTRollout.full_ids`).
        candidates: per-row candidate token ids, aligned with ``prefixes`` — e.g.
            ``[(answer_candidate_token(tok, correct), answer_candidate_token(tok, incorrect)), ...]``
            (correct first ⇒ ``logprob_diff`` = correct − incorrect) with collided rows
            (:func:`answer_tokens_collide`) already dropped.
        pad_id: padding token id (typically ``tokenizer.eos_token_id``).
        answer_ids: shared teacher-forced answer scaffold appended to every prefix
            before the read (e.g. ``answer_scaffold_ids(tokenizer)``; may be empty).
        batch_size: if set, read in forward batches of this many prefixes (bounding
            peak memory) and concatenate in order; ``None`` reads in one forward.
        progress: show a ``tqdm`` bar over batches.

    Returns:
        One :class:`AnswerScores` per prefix, in order.
    """
    prefixes = list(prefixes)
    candidates = list(candidates)
    if len(candidates) != len(prefixes):
        raise ValueError("candidates must be aligned 1:1 with prefixes")
    if batch_size is not None and len(prefixes) > batch_size:
        out: list[AnswerScores] = []
        pairs = list(zip(_chunks(prefixes, batch_size), _chunks(candidates, batch_size)))
        for pchunk, cchunk in _maybe_tqdm(pairs, progress, "answer_scores"):
            out.extend(batched_answer_scores(model, pchunk, cchunk, pad_id=pad_id,
                                             answer_ids=answer_ids))
        return out
    answer = list(answer_ids)
    input_ids, attn = _left_pad([list(p) + answer for p in prefixes], pad_id, model.device)
    position_ids = (attn.long().cumsum(-1) - 1).clamp(min=0)  # real tokens -> 0,1,2,...
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn,
                       position_ids=position_ids).logits[:, -1, :].float()
    logprobs = logits.log_softmax(dim=-1)
    top_ids = logits.argmax(dim=-1)  # whole-vocab argmax per row (free-choice token)
    out = []
    for row, cands in enumerate(candidates):
        ids = [int(c) for c in cands]
        top = int(top_ids[row])
        out.append(AnswerScores(
            candidate_ids=tuple(ids),
            logits=tuple(logits[row, ids].tolist()),
            logprobs=tuple(logprobs[row, ids].tolist()),
            top_token_id=top,
            top_logprob=float(logprobs[row, top]),
        ))
    return out


def answer_scaffold_ids(tokenizer, scaffold: str = DEFAULT_ANSWER_SCAFFOLD) -> list[int]:
    "Token ids of the teacher-forced answer scaffold (``add_special_tokens=False``)."
    return tokenizer(scaffold, add_special_tokens=False).input_ids


def answer_candidate_token(tokenizer, name: str, *,
                           scaffold: str = DEFAULT_ANSWER_SCAFFOLD) -> int | None:
    """The single token the model emits for ``name`` right after ``scaffold``.

    Derived **in context** — tokenize ``scaffold + name`` and take the first token
    past the scaffold — so we never guess whether the answer token carries a leading
    space or how it is cased: it is exactly the token that continues the forced
    scaffold, and the same token :func:`batched_answer_scores` reads ``P`` of. With
    the default ``\\boxed{`` scaffold the trailing ``{`` is its own token, so this is
    the no-space, naturally-cased first token of ``name`` (e.g. ``Paris`` / ``War``,
    *not* the space-prefixed ``ĠParis``).

    Returns ``None`` when the scaffold is **not** a clean token-prefix of
    ``scaffold + name`` — a cross-boundary BPE merge (e.g. ``{i`` for "iPhone") — or
    ``name`` contributes no tokens. Then the read point can't be pinned to one
    precomputed token, so the row should be dropped (see :func:`answer_tokens_collide`).
    """
    scaffold_ids = tokenizer(scaffold, add_special_tokens=False).input_ids
    full = tokenizer(scaffold + name, add_special_tokens=False).input_ids
    if full[: len(scaffold_ids)] != scaffold_ids or len(full) <= len(scaffold_ids):
        return None
    return full[len(scaffold_ids)]


def answer_tokens_collide(tokenizer, name_a: str, name_b: str, *,
                          scaffold: str = DEFAULT_ANSWER_SCAFFOLD) -> bool:
    """Whether two answers can't be cleanly distinguished at the scaffold read point.

    True when either name fails to align to a single post-scaffold token
    (:func:`answer_candidate_token` returns ``None``) **or** both map to the *same*
    token — in either case the first-token logit-difference metric is degenerate, so
    the pair should be filtered out before reading (SCOTSPRINT §5 first-token
    collision caveat). Uses the same in-context derivation as the read, so the
    collision test and the measured tokens always agree.
    """
    a = answer_candidate_token(tokenizer, name_a, scaffold=scaffold)
    b = answer_candidate_token(tokenizer, name_b, scaffold=scaffold)
    return a is None or b is None or a == b
