"""Rough token estimates for sizing batches under a tokens-per-minute limit.

There's no tokenizer dependency: characters / CHARS_PER_TOKEN overestimates English
and JSON on purpose. The limiter corrects each estimate with the usage the provider
reports, so the overestimate only makes batches a bit smaller than they could be.
"""

import json

CHARS_PER_TOKEN = 3.0
# One result object is ~120 tokens; the rest is headroom so replies aren't cut off.
OUTPUT_TOKENS_PER_ITEM = 250
# Headroom for reasoning models (gpt-oss), whose reasoning counts as output tokens.
REASONING_TOKENS = 1024
# What a result object actually takes (qwen: 110-135, gpt-oss without reasoning: ~80),
# for output-tokens-per-minute budgets; corrected by the reported usage.
EXPECTED_OUTPUT_TOKENS_PER_ITEM = 150


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def output_tokens(n_items: int) -> int:
    """max_tokens for a reply to `n_items` items."""
    return OUTPUT_TOKENS_PER_ITEM * max(n_items, 1) + REASONING_TOKENS


def expected_output(n_items: int) -> int:
    """Output tokens a reply to `n_items` items is expected to use."""
    return EXPECTED_OUTPUT_TOKENS_PER_ITEM * max(n_items, 1)


def count_items(user: str) -> int:
    """Number of items in a user prompt (a JSON array); 1 if it isn't one."""
    try:
        data = json.loads(user)
    except ValueError:
        return 1
    return len(data) if isinstance(data, list) and data else 1


def request_tokens(system: str, user: str) -> int:
    """Tokens a request may use in all: the prompt plus the reply allowance."""
    return estimate_tokens(system) + estimate_tokens(user) + output_tokens(count_items(user))
