import torch

from subliminality import batched_answer_probs, batched_generate_truncated


def _single_answer_prob(model, prefix, target, answer_ids=()):
    "Unpadded reference: one sequence, read P(target) at the last position."
    ids = torch.tensor([list(prefix) + list(answer_ids)], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :].float()
    return logits.softmax(dim=-1)[target].item()


# --- batched_answer_probs -----------------------------------------------------

def test_batched_answer_probs_matches_unpadded_loop(model, tokenizer):
    pad_id = tokenizer.eos_token_id
    target = 100
    # deliberately ragged prefixes so left-padding is exercised
    prefixes = [[5], [10, 20, 30], [7, 8], [1, 2, 3, 4, 5]]
    batched = batched_answer_probs(model, prefixes, target, pad_id=pad_id)
    expected = [_single_answer_prob(model, p, target) for p in prefixes]
    assert torch.allclose(torch.tensor(batched), torch.tensor(expected), atol=1e-5)


def test_batched_answer_probs_with_shared_answer(model, tokenizer):
    pad_id = tokenizer.eos_token_id
    target = 50
    answer = [40, 41]
    prefixes = [[5, 6], [9], [3, 2, 1, 4]]
    batched = batched_answer_probs(model, prefixes, target, pad_id=pad_id, answer_ids=answer)
    expected = [_single_answer_prob(model, p, target, answer) for p in prefixes]
    assert torch.allclose(torch.tensor(batched), torch.tensor(expected), atol=1e-5)


def test_batched_answer_probs_returns_probabilities(model, tokenizer):
    probs = batched_answer_probs(model, [[1, 2], [3]], target=7, pad_id=tokenizer.eos_token_id)
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)


# --- batched_generate_truncated -----------------------------------------------

def test_generate_truncated_shapes_and_prompt_prefix(model, tokenizer):
    pad_id = tokenizer.eos_token_id
    prompts = [[5, 6], [10, 11, 12]]
    n_samples = 2
    # sampling: greedy + num_return_sequences>1 is disallowed by HF, and the real
    # gen-think path samples anyway
    out = batched_generate_truncated(
        model, prompts, stop_id=pad_id, pad_id=pad_id, n_samples=n_samples,
        gen_kwargs=dict(do_sample=True, temperature=1.0, max_new_tokens=4))
    assert len(out) == len(prompts) * n_samples
    for i, seq in enumerate(out):
        prompt = prompts[i // n_samples]
        assert seq[: len(prompt)] == prompt              # prompt preserved, no padding
        assert len(seq) <= len(prompt) + 4               # bounded by max_new_tokens


def test_generate_truncated_cuts_at_first_stop(model, tokenizer):
    pad_id = tokenizer.eos_token_id
    prompt = [464, 2068, 7586]  # arbitrary tokens
    # discover the first token greedy decoding produces, then use it as the stop
    discovered = batched_generate_truncated(
        model, [prompt], stop_id=pad_id, pad_id=pad_id,
        gen_kwargs=dict(do_sample=False, max_new_tokens=5))[0]
    first_gen = discovered[len(prompt)]
    out = batched_generate_truncated(
        model, [prompt], stop_id=first_gen, pad_id=pad_id,
        gen_kwargs=dict(do_sample=False, max_new_tokens=5))[0]
    assert out == prompt + [first_gen]                   # truncated right after the stop
