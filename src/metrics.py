"""Deterministic evaluation metrics for the grounded policy assistant.

All pure functions -- no model, no network -- so they are fully unit-testable in
CI. The model's generated text is passed in as a string and scored against the
gold record produced by ``data_generator``.

Scored behaviours (the things that make the assistant *trustworthy*):
  * correct citation      -- answerable records cite the right policy id
  * grounded citation     -- every cited id is actually in the provided context
  * hallucinated citation -- any cited id NOT in context (a trust failure)
  * refusal accuracy       -- out-of-scope records are refused
  * false-refusal rate     -- answerable records wrongly refused
"""
from __future__ import annotations

import re
from typing import Dict, List

from .prompts import is_refusal

POLICY_ID_RE = re.compile(r"\[([A-Z]{3}-\d{3})\]")


def cited_ids(text: str) -> List[str]:
    return POLICY_ID_RE.findall(text)


def context_ids(record: Dict) -> List[str]:
    return [s["id"] for s in record.get("snippets", [])]


def is_grounded_citation(pred: str, record: Dict) -> bool:
    """True if the answer cites at least one id and every cited id is in context."""
    ids = cited_ids(pred)
    if not ids:
        return False
    ctx = set(context_ids(record))
    return all(i in ctx for i in ids)


def correct_citation(pred: str, record: Dict) -> bool:
    gold = record.get("gold_policy_id")
    if gold is None:
        return False
    return gold in cited_ids(pred)


def hallucinated_citation(pred: str, record: Dict) -> bool:
    ctx = set(context_ids(record))
    return any(i not in ctx for i in cited_ids(pred))


def example_passes(pred: str, record: Dict) -> bool:
    """End-to-end correctness for a single record."""
    if record.get("label") == "refuse":
        # Must refuse, and must not fabricate a citation while doing so.
        return is_refusal(pred) and not cited_ids(pred)
    # Answerable: must not refuse, must cite the correct policy, no hallucinated ids.
    return (
        not is_refusal(pred)
        and correct_citation(pred, record)
        and not hallucinated_citation(pred, record)
    )


def aggregate(preds: List[str], records: List[Dict]) -> Dict:
    """Aggregate metrics across a prediction/record set."""
    if len(preds) != len(records):
        raise ValueError("preds and records must be the same length")
    n = len(records)
    if n == 0:
        return {}

    pairs = list(zip(preds, records))
    ans = [(p, r) for p, r in pairs if r.get("label") == "answer"]
    ref = [(p, r) for p, r in pairs if r.get("label") == "refuse"]

    def frac(items, fn):
        return round(sum(1 for p, r in items if fn(p, r)) / len(items), 4) if items else None

    return {
        "n": n,
        "overall_pass_rate": round(sum(1 for p, r in pairs if example_passes(p, r)) / n, 4),
        "answerable_n": len(ans),
        "correct_citation_rate": frac(ans, correct_citation),
        "grounded_citation_rate": frac(ans, is_grounded_citation),
        "false_refusal_rate": frac(ans, lambda p, r: is_refusal(p)),
        "refusal_n": len(ref),
        "refusal_accuracy": frac(ref, lambda p, r: is_refusal(p)),
        "hallucinated_citation_rate": round(
            sum(1 for p, r in pairs if hallucinated_citation(p, r)) / n, 4
        ),
    }
