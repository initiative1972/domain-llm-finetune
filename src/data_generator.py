"""Synthetic data generator for the grounded policy-assistant fine-tune.

Produces instruction-style records with policy context, questions (in-scope and
out-of-scope), and gold answers (grounded + cited, or a refusal). Synthetic data
only -- no real company policies -- so the repo is safe to publish and fully
reproducible from a seed.

Usage:
    python -m src.data_generator --out-dir data --n 240
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .prompts import REFUSAL_TEXT


@dataclass
class Policy:
    id: str
    topic: str
    text: str
    key_fact: str


POLICIES: List[Policy] = [
    Policy(
        "POL-101", "annual leave",
        "Employees accrue 20 days of paid annual leave per year, accruing monthly. "
        "Leave must be requested at least 2 weeks in advance via the HR portal.",
        "employees get 20 days of paid annual leave per year",
    ),
    Policy(
        "POL-102", "expense reimbursement",
        "Business expenses under $150 are reimbursed without receipts; expenses of "
        "$150 or more require an itemised receipt submitted within 30 days.",
        "a receipt is required for expenses of $150 or more",
    ),
    Policy(
        "POL-103", "data classification",
        "All customer data is classified as Confidential and must not be copied to "
        "personal devices or stored outside approved systems.",
        "customer data is Confidential and must not be copied to personal devices",
    ),
    Policy(
        "POL-104", "remote work",
        "Employees may work remotely up to 3 days per week with manager approval; "
        "fully remote arrangements require a formal agreement.",
        "employees may work remotely up to 3 days per week with manager approval",
    ),
    Policy(
        "POL-105", "security incident",
        "Suspected security incidents must be reported to the Security team within "
        "1 hour of discovery through the incident hotline.",
        "suspected security incidents must be reported within 1 hour",
    ),
    Policy(
        "POL-106", "password policy",
        "Passwords must be at least 14 characters, rotated every 180 days, and never "
        "reused across systems. Multi-factor authentication is mandatory.",
        "passwords must be at least 14 characters with mandatory MFA",
    ),
    Policy(
        "POL-107", "gifts and entertainment",
        "Gifts from third parties valued over $100 must be declared in the gifts "
        "register within 5 business days.",
        "gifts over $100 must be declared within 5 business days",
    ),
    Policy(
        "POL-108", "training",
        "All staff must complete annual compliance training by 30 June each year; "
        "non-completion is escalated to the staff member's manager.",
        "annual compliance training must be completed by 30 June",
    ),
]

# Questions that are out of scope for the catalogue above (the model should refuse).
OOS_QUESTIONS: List[str] = [
    "What is the current share price of the company?",
    "Can you book me a flight to Sydney next week?",
    "What's the Wi-Fi password in the Melbourne office kitchen?",
    "How many sick days did my colleague take last year?",
    "What will my annual bonus be this year?",
    "Can you write a resignation letter for me?",
]

QUESTION_TEMPLATES: Dict[str, str] = {
    "annual leave": "How many days of paid annual leave do I get?",
    "expense reimbursement": "Do I need a receipt to claim a $200 expense?",
    "data classification": "Can I copy customer data to my personal laptop?",
    "remote work": "How many days a week can I work from home?",
    "security incident": "How quickly must I report a suspected security incident?",
    "password policy": "What are the password requirements?",
    "gifts and entertainment": "I received a $250 gift from a vendor -- what must I do?",
    "training": "When is compliance training due?",
}


def _distractors(correct: Policy, k: int, rng: random.Random) -> List[Policy]:
    others = [p for p in POLICIES if p.id != correct.id]
    rng.shuffle(others)
    return others[:k]


def make_in_scope(correct: Policy, rng: random.Random, n_distractors: int = 2) -> Dict:
    question = QUESTION_TEMPLATES[correct.topic]
    snippets = [correct] + _distractors(correct, n_distractors, rng)
    rng.shuffle(snippets)
    answer = f"Based on the policy, {correct.key_fact} [{correct.id}]."
    return {
        "question": question,
        "snippets": [{"id": p.id, "text": p.text} for p in snippets],
        "answer": answer,
        "label": "answer",
        "gold_policy_id": correct.id,
    }


def make_out_of_scope(question: str, rng: random.Random, n_distractors: int = 3) -> Dict:
    snippets = _distractors(POLICIES[0], n_distractors, rng)
    rng.shuffle(snippets)
    return {
        "question": question,
        "snippets": [{"id": p.id, "text": p.text} for p in snippets],
        "answer": REFUSAL_TEXT,
        "label": "refuse",
        "gold_policy_id": None,
    }


def generate(n_examples: int = 240, oos_ratio: float = 0.25, seed: int = 7) -> List[Dict]:
    """Build a balanced dataset of answerable and out-of-scope records."""
    rng = random.Random(seed)
    records: List[Dict] = []
    n_oos = int(n_examples * oos_ratio)
    n_in = n_examples - n_oos
    answerable = [p for p in POLICIES if p.topic in QUESTION_TEMPLATES]
    for i in range(n_in):
        records.append(make_in_scope(answerable[i % len(answerable)], rng))
    for i in range(n_oos):
        records.append(make_out_of_scope(OOS_QUESTIONS[i % len(OOS_QUESTIONS)], rng))
    rng.shuffle(records)
    return records


def split(records: List[Dict], eval_frac: float = 0.2, seed: int = 7):
    rng = random.Random(seed)
    recs = list(records)
    rng.shuffle(recs)
    n_eval = max(1, int(len(recs) * eval_frac))
    return recs[n_eval:], recs[:n_eval]


def write_jsonl(records: List[Dict], path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic policy-assistant data.")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--oos-ratio", type=float, default=0.25)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    records = generate(args.n, args.oos_ratio, args.seed)
    train, ev = split(records, args.eval_frac, args.seed)
    write_jsonl(train, os.path.join(args.out_dir, "train.jsonl"))
    write_jsonl(ev, os.path.join(args.out_dir, "eval.jsonl"))
    print(f"Wrote {len(train)} train and {len(ev)} eval records to {args.out_dir}/")


if __name__ == "__main__":
    main()
