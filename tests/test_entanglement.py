import pytest
import torch

from subliminality import compute_entanglements


def text_prompt(tokenizer, target_id=None):
    """Plain-text analog of the library's favorite-token prompt (distilgpt2 has no
    chat template). Like _default_prompt, the guided prompt states the favorite token
    up front and then reads a fresh "My favorite token is ___" prediction; the base
    prompt omits that statement. The token is spliced into the context, not predicted.
    """
    encode = lambda s: tokenizer(s, add_special_tokens=False).input_ids
    tail = "What is your favorite token? My favorite token is"
    if target_id is None:
        return encode(tail)
    return encode("My favorite token is") + [int(target_id)] + encode(". " + tail)


def vocab_size(model):
    return model.get_output_embeddings().weight.shape[0]


# --- unembedding --------------------------------------------------------------

def test_unembedding_shape_and_self_similarity(model):
    toks = torch.tensor([5, 100, 1000])
    ent = compute_entanglements(model, toks, method="unembedding")
    assert ent.shape == (3, vocab_size(model))
    # each token is its own nearest neighbour at cosine ~1
    assert torch.allclose(ent[torch.arange(3), toks], torch.ones(3), atol=1e-4)
    assert (ent.argmax(dim=-1) == toks).all()


def test_unembedding_broadcasts_over_shape(model):
    toks = torch.tensor([[5, 6], [7, 8]])
    ent = compute_entanglements(model, toks, method="unembedding")
    V = vocab_size(model)
    assert ent.shape == (2, 2, V)
    assert torch.allclose(ent.reshape(-1, V), compute_entanglements(model, toks.reshape(-1), method="unembedding"))


def test_unembedding_scalar_input(model):
    assert compute_entanglements(model, 42, method="unembedding").shape == (vocab_size(model),)


# --- output_distribution ------------------------------------------------------

def test_output_distribution_shapes_and_validity(model, tokenizer):
    guided, base = compute_entanglements(
        model, torch.tensor([5, 100, 1000]), method="output_distribution",
        tokenizer=tokenizer, prompt=text_prompt, return_components=True)
    V = vocab_size(model)
    assert guided.shape == (3, V) and base.shape == (V,)
    assert torch.allclose(guided.sum(-1), torch.ones(3), atol=1e-4)
    assert torch.allclose(base.sum(), torch.tensor(1.0), atol=1e-4)
    assert (guided >= 0).all() and (base >= 0).all()
    assert torch.isfinite(guided / base).all()


def test_output_distribution_defaults_to_ratio(model, tokenizer):
    kw = dict(method="output_distribution", tokenizer=tokenizer, prompt=text_prompt)
    ratio = compute_entanglements(model, torch.tensor([5, 100]), **kw)  # default: single tensor
    guided, base = compute_entanglements(model, torch.tensor([5, 100]), **kw, return_components=True)
    assert ratio.shape == (2, vocab_size(model))
    assert torch.allclose(ratio, guided / base)


def test_output_distribution_base_independent_and_batching(model, tokenizer):
    kw = dict(method="output_distribution", tokenizer=tokenizer, prompt=text_prompt, return_components=True)
    g5, b5 = compute_entanglements(model, torch.tensor([5]), **kw)
    g100, b100 = compute_entanglements(model, torch.tensor([100]), **kw)
    assert torch.allclose(b5, b100)  # base does not depend on the query token
    batched, _ = compute_entanglements(model, torch.tensor([5, 100]), **kw)
    assert torch.allclose(batched[0], g5[0], atol=1e-4)
    assert torch.allclose(batched[1], g100[0], atol=1e-4)


def test_output_distribution_requires_tokenizer(model):
    with pytest.raises(ValueError):
        compute_entanglements(model, torch.tensor([5]), method="output_distribution")


def test_method_is_required(model):
    with pytest.raises(TypeError):
        compute_entanglements(model, torch.tensor([5]))
