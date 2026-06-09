import pytest
import torch
from transformers import AutoTokenizer

from subliminality import (
    DEFAULT_ANSWER_FORMAT,
    DEFAULT_ANSWER_SCAFFOLD,
    AnswerScores,
    answer_candidate_token,
    answer_scaffold_ids,
    answer_tokens_collide,
    batched_answer_scores,
    build_boxed_answer_instruction,
    build_reasoning_prompt,
    rollout_cot,
)

# Greedy + neutral sampling params so distilgpt2 rollouts are deterministic and
# emit no "temperature set but do_sample=False" noise (DEFAULT_GEN_KWARGS supplies
# temperature/top_p, which we override here).
GREEDY = dict(do_sample=False, temperature=1.0, top_p=1.0)

# Minimal chat template: one line per message, and an opened think block when a
# generation prompt is requested (distilgpt2 ships no template of its own).
_CHAT_TEMPLATE = (
    "{%- for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}{%- if add_generation_prompt %}assistant: <think>\n{% endif %}"
)


@pytest.fixture(scope="module")
def chat_tokenizer():
    "A distilgpt2 tokenizer with a minimal chat template, for prompt-build tests."
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    tok.chat_template = _CHAT_TEMPLATE
    return tok


# --- build_reasoning_prompt ---------------------------------------------------

def test_build_reasoning_prompt_no_prefill(chat_tokenizer):
    ids = build_reasoning_prompt(chat_tokenizer, "What is bigger?")
    # Equals the rendered (generation-prompted) template, encoded without re-adding BOS.
    text = chat_tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is bigger?"}],
        add_generation_prompt=True, tokenize=False)
    assert ids == chat_tokenizer(text, add_special_tokens=False).input_ids


def test_build_reasoning_prompt_string_prefill_appends_encoded(chat_tokenizer):
    base = build_reasoning_prompt(chat_tokenizer, "Q")
    out = build_reasoning_prompt(chat_tokenizer, "Q", prefill="Let me think")
    assert out == base + chat_tokenizer("Let me think", add_special_tokens=False).input_ids


def test_build_reasoning_prompt_token_prefill_is_exact(chat_tokenizer):
    base = build_reasoning_prompt(chat_tokenizer, "Q")
    exact = [50, 51, 52]
    out = build_reasoning_prompt(chat_tokenizer, "Q", prefill=exact)
    assert out == base + exact            # exact ids appended verbatim, no round-trip


def test_build_reasoning_prompt_system_turn(chat_tokenizer):
    with_sys = build_reasoning_prompt(chat_tokenizer, "Q", system="be terse")
    without = build_reasoning_prompt(chat_tokenizer, "Q")
    assert len(with_sys) > len(without)   # the system turn adds tokens


# --- build_boxed_answer_instruction -------------------------------------------

def test_build_boxed_answer_instruction_renders_options():
    out = build_boxed_answer_instruction("Which is larger: France or Monaco?", ["France", "Monaco"])
    assert out == ("Which is larger: France or Monaco? Please reason step by step before "
                   "outputting either \\boxed{France} or \\boxed{Monaco}.")


def test_build_boxed_answer_instruction_default_couples_with_read(tokenizer):
    # The rendered box must start with the (whitespace-stripped) read scaffold, so the
    # token the model is told to write is exactly the one we teacher-force + read.
    assert DEFAULT_ANSWER_FORMAT.format("Paris").startswith(DEFAULT_ANSWER_SCAFFOLD.lstrip())
    sids = answer_scaffold_ids(tokenizer)
    assert answer_candidate_token(tokenizer, "Paris") == tokenizer(
        DEFAULT_ANSWER_SCAFFOLD + "Paris", add_special_tokens=False).input_ids[len(sids)]


def test_build_boxed_answer_instruction_rejects_mismatched_format():
    with pytest.raises(ValueError):
        build_boxed_answer_instruction("Q?", ["A", "B"], answer_format="{}")  # not a \boxed{} delimiter


# --- rollout_cot --------------------------------------------------------------

def test_rollout_cot_force_close(model, tokenizer):
    prompt = [464, 2068, 7586]
    eos = tokenizer.eos_token_id
    # Discover the greedy continuation so we can choose a stop the model won't emit.
    disc = rollout_cot(model, tokenizer, [prompt], end_think_id=eos,
                       max_new_tokens=4, gen_kwargs=GREEDY)[0]
    generated = set(disc.think_ids)
    stop = next(i for i in range(100, 60000) if i not in generated)

    r = rollout_cot(model, tokenizer, [prompt], end_think_id=stop,
                    max_new_tokens=4, gen_kwargs=GREEDY)[0]
    assert r.forced_close is True
    assert r.think_ids[-1] == stop                 # appended closing token
    assert r.full_ids == prompt + list(r.think_ids)
    assert isinstance(r.think_text, str)


def test_rollout_cot_natural_close(model, tokenizer):
    prompt = [464, 2068, 7586]
    eos = tokenizer.eos_token_id
    first_gen = rollout_cot(model, tokenizer, [prompt], end_think_id=eos,
                            max_new_tokens=4, gen_kwargs=GREEDY)[0].think_ids[0]
    # Use that exact token as the stop -> generation closes on it naturally.
    r = rollout_cot(model, tokenizer, [prompt], end_think_id=first_gen,
                    max_new_tokens=4, gen_kwargs=GREEDY)[0]
    assert r.forced_close is False
    assert r.think_ids == (first_gen,)             # stopped/truncated right at the close


def test_rollout_cot_sample_layout(model, tokenizer):
    prompts = [[5, 6], [10, 11, 12]]
    n_samples = 2
    rollouts = rollout_cot(model, tokenizer, prompts, end_think_id=tokenizer.eos_token_id,
                           max_new_tokens=3, n_samples=n_samples,
                           gen_kwargs=dict(do_sample=True, temperature=1.0))
    assert len(rollouts) == len(prompts) * n_samples
    for i, r in enumerate(rollouts):
        assert list(r.prompt_ids) == prompts[i // n_samples]   # prompt-major layout
        assert r.think_ids[-1] == tokenizer.eos_token_id       # always closed


def test_rollout_cot_batch_size_matches_manual_chunks(model, tokenizer):
    eos = tokenizer.eos_token_id
    prompts = [[5, 6], [10, 11, 12], [7], [1, 2, 3, 4], [8, 9]]
    greedy = dict(do_sample=False, temperature=1.0, top_p=1.0)
    chunked = rollout_cot(model, tokenizer, prompts, end_think_id=eos,
                          max_new_tokens=3, batch_size=2, gen_kwargs=greedy, seed=0)
    # Manual reference: same chunks, per-chunk seed offset (seed + chunk_index).
    manual = []
    for i in range(0, len(prompts), 2):
        manual += rollout_cot(model, tokenizer, prompts[i:i + 2], end_think_id=eos,
                              max_new_tokens=3, gen_kwargs=greedy, seed=0 + i // 2)
    assert [r.full_ids for r in chunked] == [r.full_ids for r in manual]


# --- batched_answer_scores ----------------------------------------------------

def _ref_scores(model, prefix, cand_ids, answer_ids=()):
    "Unpadded reference: raw logit/logprob of each candidate + the vocab argmax."
    ids = torch.tensor([list(prefix) + list(answer_ids)], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :].float()
    logprobs = logits.log_softmax(dim=-1)
    top = int(logits.argmax())
    return ([logits[c].item() for c in cand_ids],
            [logprobs[c].item() for c in cand_ids],
            top, logprobs[top].item())


def test_batched_answer_scores_matches_unpadded(model, tokenizer):
    pad_id = tokenizer.eos_token_id
    prefixes = [[5], [10, 20, 30], [7, 8], [1, 2, 3, 4, 5]]    # ragged -> exercises left-pad
    candidates = [[100, 7], [50, 51], [3, 9], [42, 1000]]
    out = batched_answer_scores(model, prefixes, candidates, pad_id=pad_id)
    for prefix, cands, scores in zip(prefixes, candidates, out):
        ref_logits, ref_logprobs, ref_top, ref_top_lp = _ref_scores(model, prefix, cands)
        assert scores.candidate_ids == tuple(cands)
        assert torch.allclose(torch.tensor(scores.logits), torch.tensor(ref_logits), atol=1e-4)
        assert torch.allclose(torch.tensor(scores.logprobs), torch.tensor(ref_logprobs), atol=1e-4)
        assert scores.top_token_id == ref_top                  # whole-vocab argmax
        assert abs(scores.top_logprob - ref_top_lp) < 1e-4


def test_batched_answer_scores_batch_size_matches_unbatched(model, tokenizer):
    pad_id = tokenizer.eos_token_id
    prefixes = [[5], [10, 20, 30], [7, 8], [1, 2, 3, 4, 5], [9, 9]]
    candidates = [[100, 7], [50, 51], [3, 9], [42, 1000], [6, 7]]
    whole = batched_answer_scores(model, prefixes, candidates, pad_id=pad_id)
    chunked = batched_answer_scores(model, prefixes, candidates, pad_id=pad_id, batch_size=2)
    assert len(chunked) == len(whole)
    for a, b in zip(whole, chunked):                           # deterministic -> identical
        assert a.candidate_ids == b.candidate_ids and a.top_token_id == b.top_token_id
        assert torch.allclose(torch.tensor(a.logits), torch.tensor(b.logits), atol=1e-4)


def test_batched_answer_scores_with_shared_answer_scaffold(model, tokenizer):
    pad_id = tokenizer.eos_token_id
    answer = [40, 41]
    prefixes = [[5, 6], [9], [3, 2, 1, 4]]
    candidates = [[7, 8], [11, 12], [13, 14]]
    out = batched_answer_scores(model, prefixes, candidates, pad_id=pad_id, answer_ids=answer)
    for prefix, cands, scores in zip(prefixes, candidates, out):
        ref_logits = _ref_scores(model, prefix, cands, answer)[0]
        assert torch.allclose(torch.tensor(scores.logits), torch.tensor(ref_logits), atol=1e-4)


def test_batched_answer_scores_rejects_misaligned(model, tokenizer):
    with pytest.raises(ValueError):
        batched_answer_scores(model, [[1, 2], [3]], [[5, 6]], pad_id=tokenizer.eos_token_id)


# --- AnswerScores properties (pure) -------------------------------------------

def test_answer_scores_diff_and_argmax():
    s = AnswerScores(candidate_ids=(11, 22), logits=(2.0, -1.0), logprobs=(-0.5, -3.0),
                     top_token_id=99, top_logprob=-0.1)
    assert s.logit_diff == 3.0          # candidate[0] - candidate[1]
    assert s.logprob_diff == 2.5
    assert s.argmax == 0                # candidate 0 wins (index into candidates)
    assert s.top_token_id == 99         # vocab argmax, need not be a candidate
    assert AnswerScores((1, 2), (0.0, 5.0), (-4.0, -0.1), top_token_id=2, top_logprob=-0.1).argmax == 1


# --- answer scaffold + in-context candidate derivation ------------------------

def test_answer_scaffold_ids_roundtrip(tokenizer):
    assert answer_scaffold_ids(tokenizer) == tokenizer(
        DEFAULT_ANSWER_SCAFFOLD, add_special_tokens=False).input_ids


def test_answer_candidate_token_is_in_context_no_space(tokenizer):
    # The candidate is the first token of `scaffold + name` past the scaffold itself,
    # i.e. the *no-space* token after `\boxed{` -- not the space-prefixed handle.
    sids = answer_scaffold_ids(tokenizer)
    for name in ["Paris", "London", "War", "France"]:
        cand = answer_candidate_token(tokenizer, name)
        full = tokenizer(DEFAULT_ANSWER_SCAFFOLD + name, add_special_tokens=False).input_ids
        assert cand == full[len(sids)]
        # distilgpt2: the `{` boundary makes this differ from the leading-space handle
        assert cand != tokenizer.encode(" " + name)[0]


def test_answer_candidate_token_none_on_boundary_merge(tokenizer):
    # A scaffold whose tail merges with the name across the boundary (here "Pa"+"ris"
    # -> the single token "Paris") is not a clean prefix -> None (read point can't be
    # pinned to one token).
    assert answer_candidate_token(tokenizer, "ris", scaffold="Pa") is None
    assert answer_candidate_token(tokenizer, "", scaffold="Pa") is None   # empty name


def test_answer_tokens_collide(tokenizer):
    assert answer_tokens_collide(tokenizer, "Paris", "Paris")              # identical -> same token
    assert not answer_tokens_collide(tokenizer, "Paris", "London")        # distinct tokens
    assert answer_tokens_collide(tokenizer, "def", "def", scaffold="abc")  # both None -> degenerate
    # Consistent with the underlying in-context derivation for any pair.
    a = answer_candidate_token(tokenizer, "France")
    b = answer_candidate_token(tokenizer, "Monaco")
    assert answer_tokens_collide(tokenizer, "France", "Monaco") == (a is None or b is None or a == b)
