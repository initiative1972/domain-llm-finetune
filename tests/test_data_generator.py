from src.data_generator import generate, make_in_scope, make_out_of_scope, split
from src.prompts import REFUSAL_TEXT
import random


def test_generate_is_deterministic_with_seed():
    a = generate(n_examples=50, seed=7)
    b = generate(n_examples=50, seed=7)
    assert a == b


def test_generate_count_and_oos_ratio():
    recs = generate(n_examples=100, oos_ratio=0.25, seed=7)
    assert len(recs) == 100
    refusals = [r for r in recs if r["label"] == "refuse"]
    assert len(refusals) == 25


def test_in_scope_record_has_gold_policy_in_context():
    rng = random.Random(1)
    from src.data_generator import POLICIES
    rec = make_in_scope(POLICIES[0], rng)
    ctx_ids = [s["id"] for s in rec["snippets"]]
    assert rec["gold_policy_id"] in ctx_ids
    assert rec["gold_policy_id"] in rec["answer"]
    assert rec["label"] == "answer"


def test_out_of_scope_record_is_refusal():
    rng = random.Random(1)
    rec = make_out_of_scope("What is the share price?", rng)
    assert rec["gold_policy_id"] is None
    assert rec["answer"] == REFUSAL_TEXT
    assert rec["label"] == "refuse"


def test_split_sizes():
    recs = generate(n_examples=100, seed=7)
    train, ev = split(recs, eval_frac=0.2, seed=7)
    assert len(ev) == 20
    assert len(train) == 80
    assert len(train) + len(ev) == len(recs)
