"""Preference-pair generation for DPO (Direct Preference Optimization).

Turns the synthetic records from ``data_generator`` into ``(chosen, rejected)`` pairs
that encode the exact trust failures the eval metrics punish, so DPO reinforces the
grounded-or-refuse behaviour:

  * answerable   -> chosen = grounded + correctly-cited answer
                    rejected = one of {hallucinated citation, ungrounded answer, false refusal}
  * out-of-scope -> chosen = refusal / escalate
                    rejected = confident fabricated answer with a made-up citation

Pure, seeded, no ML deps -- so it is fully unit-testable in CI.

Usage:
    python -m src.preferences --out-dir data --n 240
"""
from __future__ import annotations

import argparse
import os
import random
import re
from typing import Dict, List, Optional

from .data_generator import POLICIES, generate, split, write_jsonl
from .prompts import REFUSAL_TEXT

CITATION_RE = re.compile(r"\s*\[[A-Z]{3}-\d{3}\]")

ANSWERABLE_REJECTIONS = ("hallucinated", "ungrounded", "false_refusal")


def _strip_citation(text: str) -> str:
    return CITATION_RE.sub("", text).strip()


def _swap_citation(text: str, new_id: str) -> str:
    return CITATION_RE.sub(f" [{new_id}]", text, count=1)


def _fake_policy_id(context_ids, rng: random.Random) -> str:
    """Return a policy id that is NOT in the provided context (a hallucinated citation)."""
    exclude = set(context_ids)
    real_but_absent = [p.id for p in POLICIES if p.id not in exclude]
    invented = f"POL-{rng.randint(900, 999)}"
    return rng.choice(real_but_absent + [invented])


def make_pair(record: Dict, rng: random.Random, mode: Optional[str] = None) -> Dict:
    """Build a single preference pair from a data_generator record."""
    question = record["question"]
    snippets = record["snippets"]
    ctx_ids = [s["id"] for s in snippets]

    if record["label"] == "answer":
        chosen = record["answer"]
        mode = mode or rng.choice(ANSWERABLE_REJECTIONS)
        if mode == "hallucinated":
            rejected = _swap_citation(chosen, _fake_policy_id(ctx_ids, rng))
        elif mode == "ungrounded":
            rejected = _strip_citation(chosen)
        else:  # false_refusal
            rejected = REFUSAL_TEXT
    else:  # out-of-scope -> should refuse; rejected = confident fabrication
        chosen = REFUSAL_TEXT
        fake_id = _fake_policy_id(ctx_ids, rng)
        rejected = f"Yes, this is permitted under company policy [{fake_id}]."

    return {
        "question": question,
        "snippets": snippets,
        "chosen": chosen,
        "rejected": rejected,
    }


def make_pairs(records: List[Dict], seed: int = 7) -> List[Dict]:
    rng = random.Random(seed)
    return [make_pair(r, rng) for r in records]


def generate_preferences(
    n_examples: int = 240, oos_ratio: float = 0.25, seed: int = 7
) -> List[Dict]:
    records = generate(n_examples, oos_ratio, seed)
    return make_pairs(records, seed)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate DPO preference pairs.")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--oos-ratio", type=float, default=0.25)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    records = generate(args.n, args.oos_ratio, args.seed)
    train, ev = split(records, args.eval_frac, args.seed)
    train_pairs = make_pairs(train, args.seed)
    eval_pairs = make_pairs(ev, args.seed)
    write_jsonl(train_pairs, os.path.join(args.out_dir, "pref_train.jsonl"))
    write_jsonl(eval_pairs, os.path.join(args.out_dir, "pref_eval.jsonl"))
    print(f"Wrote {len(train_pairs)} train and {len(eval_pairs)} eval preference pairs "
          f"to {args.out_dir}/")


if __name__ == "__main__":
    main()
