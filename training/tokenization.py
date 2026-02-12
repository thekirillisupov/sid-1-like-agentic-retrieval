"""TI/TO-safe tokenization utilities.

The SID-1 paper's most critical stability finding: **never round-trip
tokens through message objects during training**.  vLLM generates raw
token IDs; the training step must receive those exact same IDs.
Retokenisation (tokens → text → messages → chat template → tokens) is
lossy (whitespace changes, special-token insertion) and creates extreme
log-probability spikes that destabilise GRPO.

This module provides helpers that:
1. Build the initial prompt as token IDs from the chat template.
2. Encode tool-result text directly into token IDs that are *appended*
   to the running sequence.
3. Never perform a full retokenisation of previously generated tokens.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase


def encode_initial_prompt(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    question: str,
) -> list[int]:
    """Encode the system + user turn into token IDs via the chat template.

    This is the *only* place where we call ``apply_chat_template``.
    All subsequent tokens are appended raw.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    token_ids: list[int] = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    return token_ids


def encode_tool_result(
    tokenizer: PreTrainedTokenizerBase,
    tool_result_text: str,
) -> list[int]:
    """Encode a tool-result string into token IDs for appending.

    We wrap the tool result in a lightweight delimiter so the model can
    distinguish tool output from its own generation, but we do NOT
    re-apply the full chat template.
    """
    wrapped = f"\n<tool_response>\n{tool_result_text}\n</tool_response>\n"
    return tokenizer.encode(wrapped, add_special_tokens=False)


def encode_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    """Encode arbitrary text without special tokens."""
    return tokenizer.encode(text, add_special_tokens=False)
