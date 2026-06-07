"""Shared fixtures: a tiny, ungated, CPU model for cheap end-to-end tests."""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "distilgpt2"


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="session")
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL).eval()
