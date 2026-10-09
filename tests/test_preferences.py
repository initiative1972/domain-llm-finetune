import random

from src.data_generator import POLICIES, make_in_scope, make_out_of_scope
from src.metrics import example_passes, hallucinated_citation, cited_ids
from src.preferences import (
    ANSWERABLE_REJECTIONS,
    generate_preferences,
    make_pair,
    make_pairs,
)
from src.prompts import REFUSAL_TEXT, is_refusal


def test_generate_preferences_is_deterministic():
    first = generate_preferences(n_examples=40, seed=7)
    second = generate_preferences(n_examples=40, seed=7)
    assert first == second


def test_pair_count_matches_records():
    recs = [make_in_scope(POLICIES[0], random.Random(1)) for _ in range(5)]
    pairs = make_pairs(recs, seed=7)
    assert len(pairs) == 5
    for p in pairs:
        assert set(p) == {"question", "snippets", "chosen", "rejected"}


def test_answerable_chosen_passes_and_every_rejection_fails():
    record = make_in_scope(POLICIES[0], random.Random(1))
    for mode in ANSWERABLE_REJECTIONS:
        pair = make_pair(record, random.Random(3), mode=mode)
        # chosen is the grounded, correctly-cited answer -> passes
        assert example_passes(pair["chosen"], record)
        # rejected encodes a trust failure -> fails
        assert not example_passes(pair["rejected"], record)


def test_hallucinated_rejection_cites_absent_policy():
    record = make_in_scope(POLICIES[1], random.Random(2))
    pair = make_pair(record, random.Random(2), mode="hallucinated")
    assert hallucinated_citation(pair["rejected"], record)


def test_ungrounded_rejection_has_no_citation():
    record = make_in_scope(POLICIES[2], random.Random(2))
    pair = make_pair(record, random.Random(2), mode="ungrounded")
    assert cited_ids(pair["rejected"]) == []


def test_false_refusal_rejection_refuses_an_answerable():
    record = make_in_scope(POLICIES[3], random.Random(2))
    pair = make_pair(record, random.Random(2), mode="false_refusal")
    assert is_refusal(pair["rejected"])


def test_out_of_scope_chosen_refuses_rejected_fabricates():
    record = make_out_of_scope("What is the share price?", random.Random(5))
    pair = make_pair(record, random.Random(5))
    assert pair["chosen"] == REFUSAL_TEXT
    assert example_passes(pair["chosen"], record)
    # fabricated answer: does not refuse and hallucinates a citation -> fails
    assert not is_refusal(pair["rejected"])
    assert not example_passes(pair["rejected"], record)
