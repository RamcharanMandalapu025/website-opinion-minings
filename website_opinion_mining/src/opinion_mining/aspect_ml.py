"""
Step 4b: Version-2 aspect detection = TRAINED multi-label classifier.

Model:  TF-IDF features  ->  One-vs-Rest Logistic Regression
  * TF-IDF (word 1-2 grams)  : which words/phrases matter
  * TF-IDF (char 3-5 grams)  : robust to typos/inflections ("delievry")
  * One-vs-Rest              : one binary classifier per aspect, so a clause
                               can have 0, 1 or several aspects (multi-label)
  * Decision threshold       : chosen by 5-fold cross-validation on TRAIN data
                               only (never on the test set -> no peeking).
"No aspect" is simply "all six probabilities below the threshold".
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from .aspect_keyword import KeywordAspectDetector
from .config import (ASPECT_MODEL_PATH, ASPECTS, EVAL_COMMENTS_CSV,
                     MODELS_DIR, TEST_CLAUSES_CSV, TRAIN_CLAUSES_CSV)
from .preprocessing import preprocess

_IDX = {a: i for i, a in enumerate(ASPECTS)}


def to_matrix(aspect_lists):
    """[[ 'delivery' ], [], ['pricing','support']] -> 0/1 matrix (n x 6)."""
    Y = np.zeros((len(aspect_lists), len(ASPECTS)), dtype=int)
    for i, lst in enumerate(aspect_lists):
        for a in lst:
            Y[i, _IDX[a]] = 1
    return Y


class KeywordFeatures(BaseEstimator, TransformerMixin):
    """
    Turns the Version-1 lexicon into 6 binary features ("did the delivery
    keywords fire?", ...). This is how the hybrid model starts from expert
    knowledge, then LEARNS where the lexicon is wrong (e.g. 'cheaply made'
    fires pricing but is really quality) and adds paraphrases it never saw.
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def fit(self, X, y=None):
        self.detector_ = KeywordAspectDetector()
        return self

    def transform(self, X):
        rows = [[1.0 if a in self.detector_.detect_aspects(c) else 0.0 for a in ASPECTS]
                for c in X]
        return csr_matrix(np.array(rows) * self.weight)


def build_model(C: float = 10.0, use_keywords: bool = True) -> Pipeline:
    parts = [
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ]
    if use_keywords:
        parts.append(("kw", KeywordFeatures()))
    clf = OneVsRestClassifier(
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
    return Pipeline([("features", FeatureUnion(parts)), ("clf", clf)])


def train(train_csv=TRAIN_CLAUSES_CSV, model_path=ASPECT_MODEL_PATH, seed: int = 42,
          use_keywords: bool = True, save: bool = True, verbose: bool = True):
    df = pd.read_csv(train_csv)
    X = df["clause"].tolist()
    Y = to_matrix([[] if a == "none" else [a] for a in df["aspect"]])

    # Pick the probability threshold using cross-validation on TRAIN data only.
    oof = np.zeros(Y.shape)
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        m = build_model(use_keywords=use_keywords).fit([X[i] for i in tr], Y[tr])
        oof[va] = m.predict_proba([X[i] for i in va])
    scores = [(t, f1_score(Y, oof >= t, average="micro", zero_division=0))
              for t in np.arange(0.2, 0.81, 0.05)]
    best_t, best_f1 = max(scores, key=lambda s: s[1])

    final = build_model(use_keywords=use_keywords).fit(X, Y)
    bundle = {"model": final, "threshold": float(best_t)}
    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, model_path)
    if verbose:
        tag = "hybrid (text + keyword features)" if use_keywords else "text-only"
        print(f"[{tag}] trained on {len(X)} clauses | CV micro-F1={best_f1:.3f} "
              f"at threshold={best_t:.2f}")
    return bundle


class MLAspectDetector:
    """Same interface as KeywordAspectDetector, so the pipeline can swap them."""

    def __init__(self, model_path=ASPECT_MODEL_PATH, threshold=None, bundle=None):
        bundle = bundle or joblib.load(model_path)
        self.model = bundle["model"]
        self.threshold = bundle["threshold"] if threshold is None else threshold

    def detect_aspects_batch(self, clauses):
        P = self.model.predict_proba(clauses)
        # highest-probability aspect first
        return [[ASPECTS[j] for j in np.argsort(-row) if row[j] >= self.threshold]
                for row in P]

    def detect_aspects(self, clause: str):
        return self.detect_aspects_batch([clause])[0]

    def detect_by_clause(self, raw_comment: str):
        clauses = preprocess(raw_comment)
        return list(zip(clauses, self.detect_aspects_batch(clauses))) if clauses else []


# ------------------------------------------------------------------ compare --
def _prf(Y, P):
    mi = precision_recall_fscore_support(Y, P, average="micro", zero_division=0)[:3]
    ma = precision_recall_fscore_support(Y, P, average="macro", zero_division=0)[:3]
    return mi, ma


def compare():
    kw = KeywordAspectDetector()
    ml_text = MLAspectDetector(bundle=train(use_keywords=False, save=False))
    ml = MLAspectDetector()                       # saved hybrid model
    models = [("V1 keyword", kw), ("V2 ML text-only", ml_text), ("V2 ML hybrid", ml)]

    # ---- A) clause level on HELD-OUT test clauses
    test = pd.read_csv(TEST_CLAUSES_CSV)
    clauses = test["clause"].tolist()
    Y = to_matrix([[] if a == "none" else [a] for a in test["aspect"]])
    P = {n: to_matrix([d.detect_aspects(c) for c in clauses]) if n == "V1 keyword"
         else to_matrix(d.detect_aspects_batch(clauses)) for n, d in models}

    # ---- B) comment level on the 200 evaluation comments
    ev = pd.read_csv(EVAL_COMMENTS_CSV)
    truth = [list(json.loads(s_).keys()) for s_ in ev["aspects"]]
    Yc = to_matrix(truth)

    def comment_pred(det):
        out = []
        for text in ev["comment"]:
            found = []
            for _, asp in det.detect_by_clause(text):
                found += [a for a in asp if a not in found]
            out.append(found)
        return to_matrix(out)
    Pc = {n: comment_pred(d) for n, d in models}

    print(f"\n{'level | model':<30}{'P':>7}{'R':>7}{'F1':>7}   (micro-averaged)")
    for level, y, preds in [("clause ", Y, P), ("comment", Yc, Pc)]:
        for n, _ in models:
            mp, mr, mf = _prf(y, preds[n])[0]
            print(f"{level} | {n:<19}{mp:>7.3f}{mr:>7.3f}{mf:>7.3f}")

    P_kw, P_ml = P["V1 keyword"], P["V2 ML hybrid"]
    print("\nPer-aspect F1 on held-out clauses (keyword -> hybrid ML):")
    for a, j in _IDX.items():
        f_kw = f1_score(Y[:, j], P_kw[:, j], zero_division=0)
        f_ml = f1_score(Y[:, j], P_ml[:, j], zero_division=0)
        print(f"  {a:<12}{f_kw:>6.2f}  ->{f_ml:>6.2f}")

    print("\nExamples the keyword detector gets wrong but ML gets right:")
    shown = 0
    for i, c in enumerate(clauses):
        if (P_kw[i] != Y[i]).any() and (P_ml[i] == Y[i]).all() and shown < 6:
            print(f"  '{c}'  truth={[a for a in ASPECTS if Y[i, _IDX[a]]] or '-'}"
                  f"  keyword={kw.detect_aspects(c) or '-'}")
            shown += 1
    print("\nExamples where ML is still wrong:")
    shown = 0
    for i, c in enumerate(clauses):
        if (P_ml[i] != Y[i]).any() and shown < 6:
            print(f"  '{c}'  truth={[a for a in ASPECTS if Y[i, _IDX[a]]] or '-'}"
                  f"  ml={ml.detect_aspects(c) or '-'}")
            shown += 1


if __name__ == "__main__":
    # Re-import under the real package name. If we trained while running as
    # "__main__", joblib would pickle KeywordFeatures as "__main__.KeywordFeatures"
    # and the saved model could never be loaded from the Streamlit app.
    from opinion_mining import aspect_ml as _m
    _m.train()
    _m.compare()
