import json
import pandas as pd
import pytest

from opinion_mining import aspect_ml
from opinion_mining.config import TRAIN_CLAUSES_CSV, TEST_CLAUSES_CSV, EVAL_COMMENTS_CSV, ASPECTS
from opinion_mining.dataset_builder import build
from opinion_mining.pipeline import PipelineV1


@pytest.fixture(scope="module")
def data():
    build()                     # deterministic (fixed seeds)
    return (pd.read_csv(TRAIN_CLAUSES_CSV), pd.read_csv(TEST_CLAUSES_CSV),
            pd.read_csv(EVAL_COMMENTS_CSV))


@pytest.fixture(scope="module")
def detector(data):
    bundle = aspect_ml.train(save=False, verbose=False)
    return aspect_ml.MLAspectDetector(bundle=bundle)


def test_no_train_test_leakage(data):
    train, test, _ = data
    assert not set(train["clause"]) & set(test["clause"])


def test_eval_set_shape(data):
    _, _, ev = data
    assert len(ev) == 200
    for s in ev["aspects"]:
        labels = json.loads(s)
        assert labels and set(labels) <= set(ASPECTS)
        assert set(labels.values()) <= {"Positive", "Negative"}


def test_model_beats_sanity_floor_on_heldout_clauses(data, detector):
    _, test, _ = data
    Y = aspect_ml.to_matrix([[] if a == "none" else [a] for a in test["aspect"]])
    P = aspect_ml.to_matrix(detector.detect_aspects_batch(test["clause"].tolist()))
    f1 = aspect_ml._prf(Y, P)[0][2]
    assert f1 >= 0.75          # observed ~0.87; floor guards against regressions


def test_ml_detector_plugs_into_pipeline(detector):
    r = PipelineV1(detector=detector).analyze(
        "Product was original but delivery took 12 days and support never replied.")
    assert r.as_dict() == {"genuineness": "Positive", "delivery": "Negative", "support": "Negative"}


def test_saved_model_loads_in_a_fresh_process():
    """Regression: training via `python -m ...` once pickled classes as __main__.*"""
    import os, subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    train = subprocess.run([sys.executable, "-m", "opinion_mining.aspect_ml"],
                           cwd=root, env=env, capture_output=True, text=True)
    assert train.returncode == 0, train.stderr
    load = subprocess.run(
        [sys.executable, "-c",
         "from opinion_mining.aspect_ml import MLAspectDetector;"
         "print(MLAspectDetector().detect_aspects('delivery took 12 days'))"],
        cwd=root, env=env, capture_output=True, text=True)
    assert load.returncode == 0, load.stderr
    assert "delivery" in load.stdout
