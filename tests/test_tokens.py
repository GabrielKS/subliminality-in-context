import torch

from subliminality import build_input_ids, first_token, is_number, token_mask, top_bottom
from subliminality.tokens import SENTINEL

# distilgpt2 has no chat template; give it a trivial one for the splice tests.
CHAT_TEMPLATE = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"


def test_is_number():
    assert is_number(" 087 ") and is_number("0")          # 87 and 0 are not blocklisted
    assert not is_number("") and not is_number(" ") and not is_number("12a")
    assert not is_number("13") and not is_number(" 666")  # excluded by the default blocklist
    assert is_number("13", blocklist=frozenset())          # ... unless the blocklist is empty


def test_first_token(tokenizer):
    assert first_token(" owl", tokenizer) == tokenizer.encode(" owl", add_special_tokens=False)[0]


def test_token_mask_selects_numbers(tokenizer):
    mask = token_mask(tokenizer, is_number)
    assert mask.dtype == torch.bool and mask.shape == (len(tokenizer),)
    assert mask[first_token("123", tokenizer)]          # a digit token passes
    assert not mask[first_token(" hello", tokenizer)]   # an alphabetic token does not


def test_build_input_ids_splices_exact_token(tokenizer):
    tokenizer.chat_template = CHAT_TEMPLATE
    messages = [{"role": "user", "content": f"num {SENTINEL} end"}]
    target = first_token(" 087", tokenizer)
    ids = build_input_ids(messages, target, tokenizer=tokenizer)
    text = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    left, right = text.split(SENTINEL)
    assert ids == (tokenizer(left, add_special_tokens=False).input_ids
                   + [target]
                   + tokenizer(right, add_special_tokens=False).input_ids)
    assert target in ids


def test_build_input_ids_no_target(tokenizer):
    tokenizer.chat_template = CHAT_TEMPLATE
    messages = [{"role": "user", "content": "no marker here"}]
    text = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    assert build_input_ids(messages, None, tokenizer=tokenizer) == tokenizer(text, add_special_tokens=False).input_ids


def test_top_bottom_ordering(tokenizer):
    scores = torch.zeros(len(tokenizer))
    scores[5], scores[7], scores[3], scores[9] = 10.0, 8.0, -4.0, -9.0
    top, bottom = top_bottom(scores, tokenizer, topk=2, bottomk=2)
    assert [t for t, _ in top] == [tokenizer.convert_ids_to_tokens([i])[0] for i in (5, 7)]
    assert [t for t, _ in bottom][0] == tokenizer.convert_ids_to_tokens([9])[0]


def test_top_bottom_zero_returns_none(tokenizer):
    top, bottom = top_bottom(torch.zeros(len(tokenizer)), tokenizer, topk=3, bottomk=0)
    assert bottom is None and len(top) == 3


def test_top_bottom_mask_restricts_eligibility(tokenizer):
    scores = torch.arange(len(tokenizer)).float()
    mask = torch.zeros(len(tokenizer), dtype=torch.bool)
    mask[10] = mask[20] = True
    top, bottom = top_bottom(scores, tokenizer, topk=1, bottomk=1, mask=mask)
    assert top[0][0] == tokenizer.convert_ids_to_tokens([20])[0]     # highest masked-in
    assert bottom[0][0] == tokenizer.convert_ids_to_tokens([10])[0]  # lowest masked-in
