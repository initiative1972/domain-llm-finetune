"""Prompt construction and chat formatting for the grounded policy assistant.

Pure functions only (no heavy ML deps) so this module is import-safe in CI and
unit tests. The same formatting is used for both training (with an answer turn)
and inference (without), guaranteeing train/eval parity.
"""
from __future__ import annotations

from typing import Dict, List, Optional

SYSTEM_PROMPT = (
    "You are a policy and compliance assistant. Answer the user's question using "
    "ONLY the policy snippets provided in the context. Cite the policy you rely on "
    "using its identifier in square brackets, e.g. [POL-101]. If the context does "
    "not contain a policy that answers the question, do not guess: reply exactly "
    "with the refusal sentence and ask the user to escalate to the Compliance team."
)

REFUSAL_TEXT = (
    "I don't have a policy that covers this. Please escalate to the Compliance team."
)


def format_context(snippets: List[Dict[str, str]]) -> str:
    """Render policy snippets into a context block keyed by policy id."""
    if not snippets:
        return "(no policies provided)"
    return "\n".join(f"[{s['id']}] {s['text']}" for s in snippets)


def build_user_prompt(question: str, snippets: List[Dict[str, str]]) -> str:
    return f"Question:\n{question}\n\nPolicy context:\n{format_context(snippets)}"


def build_messages(
    question: str,
    snippets: List[Dict[str, str]],
    answer: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build a chat message list. Include the assistant turn only for training."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, snippets)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def is_refusal(text: str) -> bool:
    """Heuristic: did the model refuse / escalate rather than answer?"""
    t = text.lower()
    return (
        "don't have a policy" in t
        or "do not have a policy" in t
        or "escalate to the compliance" in t
    )
