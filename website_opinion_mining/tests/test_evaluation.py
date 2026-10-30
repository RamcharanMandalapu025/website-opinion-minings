import json

import numpy as np
import pandas as pd
import pytest

from opinion_mining import evaluation as ev
from opinion_mining.sentiment_vader import VaderSentimentScorer


# ------------------------------------------------------------- metric maths --
def test_prf_basic_and_zero_division():
    assert ev.prf(8, 2, 2) == pytest.approx((0.8, 0.8, 0.8))
    assert ev.prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_per_comment_counts_aspects_vs_pairs():
    gold = [{"delivery": "Negative", "pricing": "Positive"}]
    pred = [{"delivery": "Positive", "pricing": "Positive", "support": "Negative"}]
    assert ev.per_comment_counts(gold, pred, pairs=False).tolist() == [[2, 1, 0]]
    # wrong sentiment on a right aspect = 1 FP + 1 FN at pair level
    assert ev.per_comment_counts(gold, pred, pairs=True).tolist() == [[1, 2, 1]]


def test_matched_sentiment_counts_neutral_as_wrong():
    gold = [{"delivery": "Negative", "pricing": "Positive"}, {"support": "Negative"}]
    pred = [{"delivery": "Negative", "pricing": "Neutral"}, {}]
    m = ev.matched_sentiment(gold, pred)
    assert m["n"] == 2 and m["accuracy"] == 0.5 and m["neutral_rate"] == 0.5
    assert m["coverage"] == pytest.approx(2 / 3)           # one gold aspect never detected


def test_bootstrap_perfect_and_paired_identical():
    gold = [{"delivery": "Negative"}] * 30
    perfect = ev.per_comment_counts(gold, gold, pairs=True)
    ci, diff = ev.bootstrap({"a": perfect, "b": perfect.copy()}, n_boot=200)
    assert ci["a"] == (1.0, 1.0) and diff["b"] == (0.0, 0.0)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    counts = np.stack([rng.integers(0, 2, 100), rng.integers(0, 2, 100), rng.integers(0, 2, 100)], axis=1).astype(float)
    ci, _ = ev.bootstrap({"a": counts}, n_boot=500)
    point = ev.micro(counts)[2]
    assert ci["a"][0] <= point <= ci["a"][1]


# ------------------------------------------------------------ scorer layers --
def test_domain_lexicon_layer_changes_scores():
    clause = "Build feels very sturdy"                      # raw VADER has no opinion on "sturdy"
    assert ev.LexiconOnlyScorer(False).score_clause(clause).label == "Neutral"
    assert ev.LexiconOnlyScorer(True).score_clause(clause).label == "Positive"


def test_phrase_rules_fix_a_wrong_direction_error():
    # Raw VADER reads "support" as a positive word, so this complaint scores POSITIVE.
    clause = "support never replied"
    assert ev.LexiconOnlyScorer(True).score_clause(clause).label == "Positive"
    assert VaderSentimentScorer().score_clause(clause).label == "Negative"


def test_ablation_layers_are_monotonic_on_clause_set():
    tc = pd.read_csv(ev.TEST_CLAUSES_CSV)
    tc = tc[tc.aspect != "none"]
    items = list(zip(tc["clause"], tc["aspect"], tc["sentiment"]))
    acc = [ev.sentiment_on_clauses(s, items)["accuracy"] for s in list(ev.scorer_variants(False).values())[:3]]
    assert acc[0] < acc[1] < acc[2]                        # each layer helps


@pytest.mark.parametrize("clause,label", [
    ("Delivery was not on time", "Negative"),       # bug found by this evaluation
    ("It never arrived on time", "Negative"),
    ("Delivered right on time", "Positive"),        # un-negated rule still works
    ("It is not bad", "Positive"),
])
def test_negated_on_time_regression(clause, label):
    assert VaderSentimentScorer().score_clause(clause, "delivery").label == label


# ----------------------------------------------------------------- full run --
def test_full_run_writes_report(tmp_path):
    out = ev.run(use_transformer=False, out_dir=tmp_path, n_boot=50)
    saved = (tmp_path / "evaluation_report.md").read_text(encoding="utf-8")
    assert saved == out
    for heading in ["## A. Aspect detection", "## B-D.", "## E. Sentiment ablation",
                    "## F. Negation", "## G. Error analysis", "## Limitations", "CAVEAT"]:
        assert heading in out


def test_run_accepts_custom_csv(tmp_path):
    rows = [{"id": i, "comment": c, "aspects": json.dumps(a)} for i, (c, a) in enumerate([
        ("Delivery was late", {"delivery": "Negative"}),
        ("Great price", {"pricing": "Positive"}),
        ("Fake product and no refund", {"genuineness": "Negative", "refund": "Negative"}),
        ("Support never replied", {"support": "Negative"}),
    ] * 5)]
    csv = tmp_path / "mine.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    out = ev.run(csv_path=csv, use_transformer=False, out_dir=tmp_path, n_boot=50)
    assert "mine.csv" in out and "20 comments" in out
