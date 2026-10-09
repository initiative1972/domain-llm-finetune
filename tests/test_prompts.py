from src.prompts import (
    REFUSAL_TEXT,
    build_messages,
    build_user_prompt,
    format_context,
    is_refusal,
)

SNIPPETS = [
    {"id": "POL-101", "text": "20 days leave."},
    {"id": "POL-102", "text": "Receipts over $150."},
]


def test_format_context_lists_ids():
    ctx = format_context(SNIPPETS)
    assert "[POL-101]" in ctx and "[POL-102]" in ctx


def test_format_context_empty():
    assert format_context([]) == "(no policies provided)"


def test_build_user_prompt_contains_question_and_context():
    p = build_user_prompt("How much leave?", SNIPPETS)
    assert "How much leave?" in p and "[POL-101]" in p


def test_build_messages_inference_has_no_assistant_turn():
    msgs = build_messages("q", SNIPPETS)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_build_messages_training_includes_answer():
    msgs = build_messages("q", SNIPPETS, answer="A [POL-101].")
    assert msgs[-1] == {"role": "assistant", "content": "A [POL-101]."}


def test_is_refusal_detects_refusal_text():
    assert is_refusal(REFUSAL_TEXT)
    assert not is_refusal("Based on the policy, 20 days [POL-101].")
