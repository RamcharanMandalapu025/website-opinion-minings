import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opinion_mining import spam_dataset, spam_filter
from opinion_mining.aggregation import aggregate, records_from_analysis
from opinion_mining.config import SPAM_HARD_CSV, SPAM_TEST_CSV, SPAM_TRAIN_CSV
from opinion_mining.pipeline import PipelineV1
from opinion_mining.spam_filter import (SpamFilter, StyleFeatures, _is_gibberish_token,
                                        is_near_duplicate)


@pytest.fixture(scope="module")
def filt():
    spam_dataset.build()
    return SpamFilter(bundle=spam_filter.train(save=False, verbose=False))


# ------------------------------------------------------------- data ------
def test_spam_templates_do_not_leak_between_train_and_test():
    for pools in (spam_dataset.PROMO, spam_dataset.GUSH, spam_dataset.COMPETITOR):
        assert not set(pools["train"]) & set(pools["test"])


def test_dataset_shapes():
    spam_dataset.build()
    tr, te = pd.read_csv(SPAM_TRAIN_CSV), pd.read_csv(SPAM_TEST_CSV)
    assert (len(tr), tr.label.sum()) == (650, 150) and (len(te), te.label.sum()) == (310, 60)
    assert set(pd.read_csv(SPAM_HARD_CSV).label) == {0, 1}


# ---------------------------------------------------------- features -----
def test_gibberish_token_detector_is_vocabulary_free():
    for w in ["delivery", "refund", "support", "packaging", "customer", "quality", "genuine"]:
        assert not _is_gibberish_token(w), w
    for w in ["qwfpbj", "hhhhhhh", "zxcvbn", "bcdfgh"]:
        assert _is_gibberish_token(w), w


def test_style_features_detect_cues():
    names = spam_filter.STYLE_NAMES
    f = dict(zip(names, StyleFeatures()._row("WOW!!!!! visit www.deals.com call +91 9876543210")))
    assert f["link"] > 0 and f["phone"] == 1 and f["exclaim"] == 1
    shouted = dict(zip(names, StyleFeatures()._row("WORST SUPPORT EVER NEVER REPLIED")))
    assert shouted["caps"] > 0.9 and f["caps"] < 0.3        # ratio of capital letters
    clean = dict(zip(names, StyleFeatures()._row("Delivery took 9 days but the item is genuine.")))
    assert clean["link"] == 0 and clean["phone"] == 0 and clean["exclaim"] == 0 and clean["gibberish"] == 0


# ------------------------------------------------------------- model -----
def test_model_on_synthetic_test_set(filt):
    te = pd.read_csv(SPAM_TEST_CSV)
    pred = filt.predict_proba(te["text"].tolist()) >= filt.threshold
    tp = (pred & (te.label == 1)).sum(); fp = (pred & (te.label == 0)).sum()
    assert tp / max(tp + fp, 1) >= 0.9 and tp / (te.label == 1).sum() >= 0.9


@pytest.mark.parametrize("text", [
    "Visit www.dealz4u.com for 70% off on all items!!!",
    "asdkjh qwezxc lkjhgf",
    "Best best best amazing amazing amazing A+++ 10/10",
])
def test_obvious_spam_is_blocked(filt, text):
    v = filt.assess(text)
    assert v.is_spam and v.weight == 0.0


@pytest.mark.parametrize("text", [
    "Delivery took 9 days but the item is genuine and well packed.",
    "Bought 2 shirts for 499 each, colour faded after one wash, return was easy.",
    "Support never replied to my emails and the refund is still pending.",
])
def test_normal_reviews_pass(filt, text):
    assert filt.assess(text).label != "spam"


def test_reasons_are_explainable(filt):
    v = filt.assess("Click here www.win-free.xyz NOW!!!!! call +91 9876543210")
    assert "contains a link" in v.reasons and "contains a phone number" in v.reasons


# ---------------------------------------------------------- tiers & dup --
class _Stub:
    """Fixed probability, so we can test the tier logic exactly."""
    def __init__(self, p): self.p = p
    def predict_proba(self, X): return np.array([[1 - self.p, self.p] for _ in X])


@pytest.mark.parametrize("p,label,weight", [(0.9, "spam", 0.0), (0.2, "suspicious", 0.8), (0.05, "clean", 1.0)])
def test_tiers_and_weights(p, label, weight):
    f = SpamFilter(bundle={"model": _Stub(p), "threshold": 0.4})
    v = f.assess("some comment")
    assert (v.label, v.weight) == (label, weight)


def test_near_duplicates():
    base = ["Product was original but delivery took twelve days and support never replied"]
    assert is_near_duplicate("product was original but delivery took twelve days and support never replied!!", base)
    assert not is_near_duplicate("Fast delivery and great packaging overall", base)
    assert not is_near_duplicate("Good product", ["Good product"])      # too short to judge


def test_duplicate_flood_is_blocked(filt):
    existing = ["The seller delivered a genuine product within two days"]
    v = filt.assess("The seller delivered a genuine product within two days.", site_comments=existing)
    assert v.is_spam and "near-duplicate of an existing comment" in v.reasons


# ---------------------------------------------------- integration (Step 6) --
def test_spam_weight_removes_comment_from_ratings(filt):
    pipe = PipelineV1()
    comments = ["Great price great price great price!!! Visit www.deals.com for 90% off",
                "Delivery took 12 days"]
    records = []
    for i, c in enumerate(comments):
        w = filt.assess(c).weight
        records += records_from_analysis(pipe.analyze(c), weight=w, comment_id=i)
    rating = aggregate(records)
    assert rating.aspects["pricing"].stars is None          # spam ignored
    assert rating.aspects["delivery"].stars is not None     # real comment kept


# ---------------------------------------------- regression: pickle __main__ --
def test_saved_spam_model_loads_in_fresh_process():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    t = subprocess.run([sys.executable, "-m", "opinion_mining.spam_filter"],
                       cwd=root, env=env, capture_output=True, text=True)
    assert t.returncode == 0, t.stderr
    l = subprocess.run([sys.executable, "-c",
                        "from opinion_mining.spam_filter import SpamFilter;"
                        "print(SpamFilter().assess('Visit www.x.com now!!!').label)"],
                       cwd=root, env=env, capture_output=True, text=True)
    assert l.returncode == 0, l.stderr
    assert "spam" in l.stdout
