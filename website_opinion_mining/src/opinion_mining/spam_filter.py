"""
Step 7b: spam / fake-review filter = TF-IDF + Logistic Regression.

Reads the RAW comment (NOT the cleaned one): links, "!!!!", ALL CAPS and
"soooo" are exactly the signals Step-2 cleaning throws away.

Features
  * TF-IDF word 1-2 grams   -> spammy phrases ("click here", "best best best")
  * TF-IDF char 2-5 grams   -> URL / phone / punctuation patterns, typos
  * 9 STYLE features (all in 0..1)  -> link, phone number, '!' density, ALL-CAPS ratio,
      digit ratio, length, repeated-word ratio, gibberish-token ratio (no vowels / long
      consonant clusters; vocabulary-FREE, so new genuine words don't trigger it),
      letter-runs ("hhhhh")
Model:   Logistic Regression (class_weight="balanced": spam is the minority class)

Decision threshold is tuned by 5-fold CV on TRAIN data maximizing F-beta with
beta=0.5, i.e. PRECISION counts more than recall. Reason: wrongly blocking a
genuine customer is worse than letting a little spam through, and the
aggregator already down-weights borderline comments.

Output per comment (SpamVerdict):
   prob, label (clean | suspicious | spam), weight for aggregation, reasons.
       spam        p >= threshold      -> weight 0.0   (excluded from ratings)
       suspicious  p >= threshold/2    -> weight 1 - p
       clean       otherwise           -> weight 1.0
Near-duplicate detection (copy-paste floods) is a separate rule: Jaccard >= 0.85
on word sets against the site's existing comments.
"""
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, fbeta_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import (MODELS_DIR, SPAM_HARD_CSV, SPAM_MODEL_PATH, SPAM_TEST_CSV,
                     SPAM_TRAIN_CSV)

_URL = re.compile(r"(https?://|www\.|\b[\w-]+\.(?:com|in|xyz|biz|net|org|co)\b|bit\.ly/|tinyurl\.)", re.I)
_PHONE = re.compile(r"(\+?\d[\d\s\-]{8,}\d)")
_WORD = re.compile(r"[a-z']+")
_RUN = re.compile(r"([A-Za-z])\1{3,}")

STYLE_NAMES = ["link", "phone", "exclaim", "caps", "digits", "length",
               "repeats", "gibberish", "letter_runs"]


_CONSONANT_RUN = re.compile(r"[^aeiouy\W\d_]{5,}")      # 5+ consonants in a row


def _is_gibberish_token(tok: str) -> bool:
    """Vocabulary-free check: keyboard mash has too few vowels or long consonant runs."""
    if len(tok) <= 2:
        return False
    vowels = sum(c in "aeiouy" for c in tok)
    return vowels == 0 or vowels / len(tok) < 0.2 or bool(_CONSONANT_RUN.search(tok))


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Hand-crafted style cues (stateless: nothing is learned here)."""

    def fit(self, X, y=None):
        return self

    def _row(self, text: str):
        toks = _WORD.findall(text.lower())
        letters = [c for c in text if c.isalpha()]
        n = len(toks)
        return [
            min(len(_URL.findall(text)), 2) / 2,
            1.0 if _PHONE.search(text) else 0.0,
            min(text.count("!"), 5) / 5,
            sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
            min(sum(c.isdigit() for c in text) / max(len(text), 1) * 5, 1.0),
            min(math.log1p(n) / math.log(60), 1.0),
            (1 - len(set(toks)) / n) if n > 1 else 0.0,
            (sum(_is_gibberish_token(t) for t in toks) / n) if n else 0.0,
            1.0 if _RUN.search(text) else 0.0,
        ]

    def transform(self, X):
        return csr_matrix(np.array([self._row(t) for t in X]))


def build_model(C: float = 10.0, use_style: bool = True) -> Pipeline:
    parts = [("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
             ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True))]
    if use_style:
        parts.append(("style", StyleFeatures()))
    return Pipeline([("features", FeatureUnion(parts)),
                     ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000))])


def train(train_csv=SPAM_TRAIN_CSV, model_path=SPAM_MODEL_PATH, seed: int = 42,
          use_style: bool = True, save: bool = True, verbose: bool = True) -> dict:
    df = pd.read_csv(train_csv)
    X, y = df["text"].tolist(), df["label"].to_numpy()
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = build_model(use_style=use_style).fit([X[i] for i in tr], y[tr])
        oof[va] = m.predict_proba([X[i] for i in va])[:, 1]
    cands = [(t, fbeta_score(y, oof >= t, beta=0.5, zero_division=0))
             for t in np.arange(0.2, 0.91, 0.05)]
    f = max(c[1] for c in cands)
    # CV can be flat (several thresholds equally good). Taking the FIRST one is
    # arbitrary and was far too aggressive; take the middle of the plateau.
    plateau = [t for t, sc in cands if sc >= f - 1e-9]
    thr = float(np.median(plateau))
    bundle = {"model": build_model(use_style=use_style).fit(X, y), "threshold": float(thr)}
    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, model_path)
    if verbose:
        print(f"[{'text+style' if use_style else 'text-only'}] trained on {len(X)} comments | "
              f"CV F0.5={f:.3f} at threshold={thr:.2f}")
    return bundle


# ------------------------------------------------------------------ verdict --
@dataclass
class SpamVerdict:
    prob: float
    label: str                      # clean | suspicious | spam
    weight: float                   # multiplier for aggregation (Step 6)
    reasons: List[str] = field(default_factory=list)

    @property
    def is_spam(self) -> bool:
        return self.label == "spam"


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def is_near_duplicate(text: str, existing: Iterable[str],
                      threshold: float = 0.85, min_tokens: int = 5) -> bool:
    a = _tokens(text)
    if len(a) < min_tokens:                  # "Good product" duplicates are normal
        return False
    for other in existing:
        b = _tokens(other)
        if b and len(a & b) / len(a | b) >= threshold:
            return True
    return False


class SpamFilter:
    def __init__(self, model_path=SPAM_MODEL_PATH, bundle: Optional[dict] = None):
        bundle = bundle or joblib.load(model_path)
        self.model, self.threshold = bundle["model"], bundle["threshold"]
        feats = getattr(self.model, "named_steps", {}).get("features")
        self._style = next((t for n, t in feats.transformer_list if n == "style"), None) if feats else None

    def _reasons(self, text: str) -> List[str]:
        if self._style is None:
            return []
        f = dict(zip(STYLE_NAMES, self._style._row(text)))
        out = []
        if f["link"] > 0: out.append("contains a link")
        if f["phone"] > 0: out.append("contains a phone number")
        if f["exclaim"] >= 0.6: out.append("excessive exclamation marks")
        if f["caps"] > 0.6: out.append("mostly UPPERCASE")
        if f["repeats"] >= 0.4: out.append("same words repeated")
        if f["gibberish"] >= 0.5: out.append("looks like keyboard mashing")
        if f["letter_runs"] > 0: out.append("stretched/repeated letters")
        return out

    def assess(self, text: str, site_comments: Optional[Iterable[str]] = None) -> SpamVerdict:
        p = float(self.model.predict_proba([text])[0, 1])
        reasons = self._reasons(text)
        if site_comments is not None and is_near_duplicate(text, site_comments):
            p = max(p, 0.95)
            reasons.append("near-duplicate of an existing comment")
        if p >= self.threshold:
            return SpamVerdict(round(p, 3), "spam", 0.0, reasons)
        if p >= self.threshold / 2:
            return SpamVerdict(round(p, 3), "suspicious", round(1 - p, 3), reasons)
        return SpamVerdict(round(p, 3), "clean", 1.0, reasons)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.model.predict_proba(texts)[:, 1]


# --------------------------------------------------------------- evaluation --
def _report(name, y, p, thr):
    pred = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"{name:<22}{pr:>7.3f}{rc:>7.3f}{f1:>7.3f}    TP={tp} FP={fp} FN={fn} TN={tn}")


def compare():
    test, hard = pd.read_csv(SPAM_TEST_CSV), pd.read_csv(SPAM_HARD_CSV)
    print(f"\n{'model / dataset (spam class)':<22}{'P':>7}{'R':>7}{'F1':>7}")
    final = None
    for use_style in (False, True):
        b = train(use_style=use_style, save=False, verbose=False)
        f = SpamFilter(bundle=b)
        tag = "text+style" if use_style else "text-only"
        _report(f"{tag} | synthetic", test["label"].to_numpy(), f.predict_proba(test["text"].tolist()), f.threshold)
        _report(f"{tag} | HARD set", hard["label"].to_numpy(), f.predict_proba(hard["text"].tolist()), f.threshold)
        final = f if use_style else final

    print("\nBy spam kind (text+style, synthetic test) - recall:")
    p = final.predict_proba(test["text"].tolist()) >= final.threshold
    for kind, g in test[test.label == 1].groupby("kind"):
        print(f"  {kind:<11}{p[g.index].mean():.2f}  (n={len(g)})")

    print("\nHARD-set mistakes (text+style):")
    ph = final.predict_proba(hard["text"].tolist())
    wrong = [(t, l, q) for t, l, q in zip(hard["text"], hard["label"], ph) if (q >= final.threshold) != bool(l)]
    for t, l, q in wrong:
        print(f"  truth={'SPAM ' if l else 'clean'} p={q:.2f}  {t}")
    if not wrong:
        print("  none")


if __name__ == "__main__":
    # Re-import under the package name so pickles don't reference __main__ (see Step 5 bug).
    from opinion_mining import spam_filter as _m
    from opinion_mining import spam_dataset
    spam_dataset.build()
    _m.train()
    _m.compare()
