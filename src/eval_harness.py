"""Model-agnostic evaluation harness.

Takes a ``generate_fn`` that maps a chat message list to a response string, runs it
over the eval set, scores deterministically with ``metrics``, and emits a
before/after report (JSON + Markdown).

The harness never imports torch/unsloth: tests inject a stub ``generate_fn`` and
``scripts/run_eval.py`` injects a real model. This keeps all scoring logic green
in CI without a GPU.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List

from .metrics import aggregate
from .prompts import build_messages

GenerateFn = Callable[[List[Dict[str, str]]], str]


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_model(generate_fn: GenerateFn, records: List[Dict]) -> List[str]:
    preds: List[str] = []
    for r in records:
        messages = build_messages(r["question"], r["snippets"])  # no answer -> inference
        preds.append(generate_fn(messages))
    return preds


def evaluate(generate_fn: GenerateFn, records: List[Dict]) -> Dict:
    preds = run_model(generate_fn, records)
    return {"metrics": aggregate(preds, records), "predictions": preds}


def compare(base_fn: GenerateFn, tuned_fn: GenerateFn, records: List[Dict]) -> Dict:
    return {
        "base": evaluate(base_fn, records)["metrics"],
        "tuned": evaluate(tuned_fn, records)["metrics"],
    }


_REPORT_ROWS = [
    ("overall_pass_rate", "Overall pass rate"),
    ("correct_citation_rate", "Correct citation (answerable)"),
    ("grounded_citation_rate", "Grounded citation (answerable)"),
    ("false_refusal_rate", "False-refusal rate (answerable, lower=better)"),
    ("refusal_accuracy", "Refusal accuracy (out-of-scope)"),
    ("hallucinated_citation_rate", "Hallucinated-citation rate (lower=better)"),
]


def to_markdown(report: Dict) -> str:
    base = report.get("base", {})
    tuned = report.get("tuned", {})
    lines = ["| Metric | Base | Fine-tuned |", "|---|---|---|"]
    for key, label in _REPORT_ROWS:
        lines.append(f"| {label} | {base.get(key)} | {tuned.get(key)} |")
    return "\n".join(lines)


def write_report(report: Dict, json_path: str, md_path: str) -> None:
    for p in (json_path, md_path):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation report -- base vs fine-tuned\n\n")
        f.write(to_markdown(report) + "\n")
