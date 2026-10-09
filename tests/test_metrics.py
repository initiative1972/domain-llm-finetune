from src.metrics import (
    aggregate,
    cited_ids,
    correct_citation,
    example_passes,
    hallucinated_citation,
    is_grounded_citation,
)
from src.prompts import REFUSAL_TEXT

ANSWER_REC = {
    "label": "answer",
    "gold_policy_id": "POL-101",
    "snippets": [{"id": "POL-101", "text": "x"}, {"id": "POL-102", "text": "y"}],
}
REFUSE_REC = {
    "label": "refuse",
    "gold_policy_id": None,
    "snippets": [{"id": "POL-103", "text": "z"}],
}


def test_cited_ids_extracts_policy_ids():
    assert cited_ids("See [POL-101] and [POL-108].") == ["POL-101", "POL-108"]


def test_correct_and_grounded_citation():
    pred = "Based on the policy, 20 days [POL-101]."
    assert correct_citation(pred, ANSWER_REC)
    assert is_grounded_citation(pred, ANSWER_REC)
    assert not hallucinated_citation(pred, ANSWER_REC)


def test_hallucinated_citation_flagged():
    pred = "You get leave [POL-999]."
    assert hallucinated_citation(pred, ANSWER_REC)
    assert not is_grounded_citation(pred, ANSWER_REC)


def test_example_passes_answerable():
    assert example_passes("Based on the policy, 20 days [POL-101].", ANSWER_REC)
    assert not example_passes("You get 20 days.", ANSWER_REC)  # no citation
    assert not example_passes(REFUSAL_TEXT, ANSWER_REC)  # wrong refusal


def test_example_passes_refusal():
    assert example_passes(REFUSAL_TEXT, REFUSE_REC)
    assert not example_passes("It is covered by [POL-103].", REFUSE_REC)


def test_aggregate_perfect_predictions():
    preds = ["Based on the policy, 20 days [POL-101].", REFUSAL_TEXT]
    recs = [ANSWER_REC, REFUSE_REC]
    m = aggregate(preds, recs)
    assert m["overall_pass_rate"] == 1.0
    assert m["correct_citation_rate"] == 1.0
    assert m["refusal_accuracy"] == 1.0
    assert m["false_refusal_rate"] == 0.0
    assert m["hallucinated_citation_rate"] == 0.0


def test_aggregate_all_refuse_predictions():
    preds = [REFUSAL_TEXT, REFUSAL_TEXT]
    recs = [ANSWER_REC, REFUSE_REC]
    m = aggregate(preds, recs)
    assert m["false_refusal_rate"] == 1.0
    assert m["refusal_accuracy"] == 1.0
    assert m["overall_pass_rate"] == 0.5
