from src.eval_harness import compare, evaluate, to_markdown
from src.prompts import REFUSAL_TEXT

RECORDS = [
    {
        "question": "How much leave?",
        "label": "answer",
        "gold_policy_id": "POL-101",
        "snippets": [{"id": "POL-101", "text": "20 days"}, {"id": "POL-102", "text": "y"}],
    },
    {
        "question": "Share price?",
        "label": "refuse",
        "gold_policy_id": None,
        "snippets": [{"id": "POL-103", "text": "z"}],
    },
]


def oracle_fn(messages):
    """A perfect model: answers answerable records, refuses the rest.

    It reads the gold behaviour from the user prompt text (test-only stub).
    """
    user = messages[-1]["content"]
    if "POL-101" in user and "How much leave" in user:
        return "Based on the policy, 20 days [POL-101]."
    return REFUSAL_TEXT


def always_refuse_fn(messages):
    return REFUSAL_TEXT


def test_evaluate_with_oracle_is_perfect():
    out = evaluate(oracle_fn, RECORDS)
    assert out["metrics"]["overall_pass_rate"] == 1.0
    assert len(out["predictions"]) == 2


def test_compare_shows_base_vs_tuned():
    report = compare(always_refuse_fn, oracle_fn, RECORDS)
    assert report["base"]["overall_pass_rate"] == 0.5
    assert report["tuned"]["overall_pass_rate"] == 1.0


def test_to_markdown_renders_table():
    report = compare(always_refuse_fn, oracle_fn, RECORDS)
    md = to_markdown(report)
    assert "| Metric | Base | Fine-tuned |" in md
    assert "Overall pass rate" in md
