"""
Step 10: evaluation.      python -m opinion_mining.evaluation [--csv my_labels.csv] [--no-transformer]

WHAT IS MEASURED (viva cheat-sheet)
 A. ASPECT DETECTION   per comment, is the SET of aspects right?  Precision / Recall / F1.
 B. SENTIMENT          given an aspect found in both gold and prediction, is the label right?
                       (a predicted "Neutral" counts as wrong: gold is never neutral)
 C. END-TO-END         (aspect, sentiment) PAIR must both be right - the strictest metric.
 D. MIXED comments     pair-F1 on comments having both positive and negative aspects.
 E. SENTIMENT ABLATION clause-level: what does each layer add? (raw VADER -> +domain lexicon
                       -> +phrase rules -> +sarcasm -> transformer)
 F. NEGATION / SARCASM hand-written mini-sets (include cases we EXPECT to fail).
 G. 95% BOOTSTRAP CIs  resample the 200 comments 1000x; paired differences between pipelines.

HONEST CAVEATS (also written into the report)
 * The default 200 comments are SYNTHETIC (held-out clauses, exact labels). Numbers are
   optimistic. For the final viva, label ~200 REAL comments by hand and pass --csv.
 * The mini-sets are small; the sarcasm rules were designed by the same author.

CSV format for --csv:  id, comment, aspects   (aspects = JSON like {"delivery": "Negative"})
"""
import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import (ASPECT_DISPLAY_NAMES, ASPECTS, EVAL_COMMENTS_CSV, REPORTS_DIR,
                     TEST_CLAUSES_CSV)
from .preprocessing import clean_text
from .sarcasm import SarcasmAwareScorer
from .sentiment_vader import DOMAIN_LEXICON, ClauseSentiment, VaderSentimentScorer, to_label

# ----------------------------------------------------------------- mini sets --
# (clause, gold label). Written BEFORE looking at results; some are expected to fail.
NEGATION_SET = [
    ("It is not bad", "Positive"), ("The delivery was not slow", "Positive"),
    ("Quality is not great", "Negative"), ("Not good at all", "Negative"),
    ("No complaints about the packaging", "Positive"), ("I am not disappointed", "Positive"),
    ("Support is not helpful", "Negative"), ("The price is not high", "Positive"),
    ("Never had any problem with refunds", "Positive"), ("Delivery was not on time", "Negative"),
    ("The product is not original", "Negative"), ("Not worth the money", "Negative"),
    ("It did not break", "Positive"), ("Customer care was not rude", "Positive"),
    ("Not a single complaint", "Positive"), ("The refund did not arrive", "Negative"),
]
SARCASM_SET = [  # sarcastic -> gold Negative
    ("Brilliant, my parcel is lost", "Negative"),
    ("Thanks a lot for the broken charger", "Negative"),
    ("Wow what a surprise, the screen cracked on day one", "Negative"),
    ("Great job guys, the seller ignored me again", "Negative"),
    ("Fantastic, only 3 weeks for a phone case", "Negative"),
    ("Love paying double for a fake", "Negative"),
    ("Perfect timing, it arrived after the festival", "Negative"),
    ("Oh wow, such amazing service, they hung up on me", "Negative"),
    ("Excellent, the zipper broke immediately", "Negative"),
    ("Nice, another damaged box", "Negative"),
    ("Yeah right, genuine product", "Negative"),
    ("Awesome, a faulty phone for full price", "Negative"),
    # genuine praise that LOOKS sarcastic -> gold Positive
    ("Great, no delays at all", "Positive"),
    ("Wonderful packaging, nothing was broken", "Positive"),
    ("Excellent, never had a late delivery", "Positive"),
    ("Amazing quality, not damaged at all", "Positive"),
    ("Perfect, it works without any problem", "Positive"),
    ("Thanks for the quick delivery", "Positive"),
    ("Wow, the delivery was fast", "Positive"),
    ("Love how fast the support replies", "Positive"),
    ("Great product, not slow at all", "Positive"),
    ("Awesome deal, the price is a bargain", "Positive"),
    ("Nice, the refund arrived within two days", "Positive"),
    ("Fantastic seller, they replaced the fake item immediately", "Positive"),
]


# ------------------------------------------------------------ scorer variants --
class LexiconOnlyScorer:
    """VADER with or without our domain lexicon, and NO phrase / delivery rules."""

    def __init__(self, domain_lexicon: bool):
        self.analyzer = SentimentIntensityAnalyzer()
        if domain_lexicon:
            self.analyzer.lexicon.update(DOMAIN_LEXICON)

    def score_clause(self, clause, aspect=None) -> ClauseSentiment:
        c = self.analyzer.polarity_scores(clause)["compound"]
        return ClauseSentiment(clause, round(c, 3), to_label(c), "vader")


def transformer_available() -> bool:
    return importlib.util.find_spec("transformers") is not None


def scorer_variants(use_transformer: bool) -> Dict[str, object]:
    v = {
        "1. VADER (original lexicon)": LexiconOnlyScorer(False),
        "2. + domain lexicon": LexiconOnlyScorer(True),
        "3. + phrase & delivery-days rules": VaderSentimentScorer(),
    }
    v["4. + sarcasm layer"] = SarcasmAwareScorer(v["3. + phrase & delivery-days rules"])
    if use_transformer:
        from .sentiment_transformer import TransformerSentimentScorer
        v["5a. Transformer only (no rules)"] = TransformerSentimentScorer(use_rules=False)
        v["5b. Transformer + rules + sarcasm"] = SarcasmAwareScorer(TransformerSentimentScorer())
    return v


# --------------------------------------------------------------------- metrics --
def prf(tp: float, fp: float, fn: float):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def per_comment_counts(gold: List[dict], pred: List[dict], pairs: bool) -> np.ndarray:
    """(n, 3) array of [TP, FP, FN] per comment; pairs=True compares (aspect, label)."""
    out = np.zeros((len(gold), 3))
    for i, (g, p) in enumerate(zip(gold, pred)):
        gs, ps = (set(g.items()), set(p.items())) if pairs else (set(g), set(p))
        out[i] = [len(gs & ps), len(ps - gs), len(gs - ps)]
    return out


def micro(counts: np.ndarray):
    return prf(*counts.sum(axis=0))


def per_aspect_prf(gold, pred) -> Dict[str, tuple]:
    res = {}
    for a in ASPECTS:
        tp = sum(a in g and a in p for g, p in zip(gold, pred))
        fp = sum(a not in g and a in p for g, p in zip(gold, pred))
        fn = sum(a in g and a not in p for g, p in zip(gold, pred))
        res[a] = prf(tp, fp, fn)
    return res


def matched_sentiment(gold, pred) -> dict:
    """Sentiment quality on aspects present in BOTH gold and prediction."""
    yt, yp, asp = [], [], []
    for g, p in zip(gold, pred):
        for a, lab in g.items():
            if a in p:
                yt.append(lab); yp.append(p[a]); asp.append(a)
    n_gold = sum(len(g) for g in gold)
    if not yt:
        return {"n": 0, "coverage": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "neutral_rate": 0.0}
    pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, labels=["Positive", "Negative"],
                                                    zero_division=0)
    return {"n": len(yt), "coverage": len(yt) / n_gold,
            "accuracy": float(np.mean([a == b for a, b in zip(yt, yp)])),
            "macro_f1": float(f1.mean()), "neutral_rate": float(np.mean([b == "Neutral" for b in yp])),
            "by_class": {"Positive": (pr[0], rc[0], f1[0]), "Negative": (pr[1], rc[1], f1[1])}}


def bootstrap(count_sets: Dict[str, np.ndarray], n_boot: int = 1000, seed: int = 0):
    """95% CI of micro-F1 per pipeline + PAIRED difference vs the first pipeline."""
    names = list(count_sets)
    n = len(next(iter(count_sets.values())))
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    f1s = {}
    for k, c in count_sets.items():
        s = c[idx].sum(axis=1)
        f1s[k] = 2 * s[:, 0] / np.maximum(2 * s[:, 0] + s[:, 1] + s[:, 2], 1e-9)
    ci = {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in f1s.items()}
    diff = {k: tuple(np.percentile(f1s[k] - f1s[names[0]], [2.5, 97.5])) for k in names[1:]}
    return ci, diff


def sentiment_on_clauses(scorer, items) -> dict:
    """items: [(clause, aspect_or_None, gold)] -> accuracy / macro-F1 / neutral rate."""
    preds = [scorer.score_clause(clean_text(c), a).label for c, a, _ in items]
    gold = [g for _, _, g in items]
    _, _, f1, _ = precision_recall_fscore_support(gold, preds, labels=["Positive", "Negative"],
                                                  zero_division=0)
    return {"accuracy": float(np.mean([p == g for p, g in zip(preds, gold)])),
            "macro_f1": float(f1.mean()),
            "neutral_rate": float(np.mean([p == "Neutral" for p in preds])), "preds": preds}


# ---------------------------------------------------------------------- report --
def md_table(headers: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt_ci(v, ci):
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def build_pipelines(use_transformer: bool) -> Dict[str, object]:
    from .aspect_keyword import KeywordAspectDetector  # noqa: F401
    from .aspect_ml import MLAspectDetector
    from .pipeline import PipelineV1, build_pipeline
    pipes = {
        "V1  keyword + VADER": PipelineV1(),
        "V2a ML aspects + VADER": PipelineV1(detector=MLAspectDetector(), scorer=VaderSentimentScorer()),
        "V2b ML aspects + VADER + sarcasm": build_pipeline("v2", transformer=False),
    }
    if use_transformer:
        pipes["V2c ML aspects + Transformer + sarcasm"] = build_pipeline("v2", transformer=True)
    return pipes


def run(csv_path=EVAL_COMMENTS_CSV, use_transformer: Optional[bool] = None,
        out_dir=REPORTS_DIR, n_boot: int = 1000) -> str:
    notes = []
    if use_transformer is None:
        use_transformer = transformer_available()
    if use_transformer:
        try:                                         # fail early & clearly (e.g. no internet)
            from .sentiment_transformer import TransformerSentimentScorer
            TransformerSentimentScorer().score_clause("warm up")
        except Exception as e:
            notes.append(f"Transformer skipped: could not load the model ({type(e).__name__}).")
            use_transformer = False
    if not use_transformer and not any("skipped" in n for n in notes):
        notes.append("Transformer rows skipped: `pip install transformers torch` and re-run to include them.")

    ev = pd.read_csv(csv_path)
    texts = ev["comment"].tolist()
    gold = [json.loads(s) for s in ev["aspects"]]
    mixed = np.array([("Positive" in g.values() and "Negative" in g.values()) for g in gold])

    pipes = build_pipelines(use_transformer)
    preds = {name: [p.analyze(t).as_dict() for t in texts] for name, p in pipes.items()}

    asp_counts = {n: per_comment_counts(gold, p, pairs=False) for n, p in preds.items()}
    pair_counts = {n: per_comment_counts(gold, p, pairs=True) for n, p in preds.items()}
    asp_ci, asp_diff = bootstrap(asp_counts, n_boot)
    pair_ci, pair_diff = bootstrap(pair_counts, n_boot)
    names = list(preds)

    L = ["# Evaluation report", "",
         f"Evaluation file: `{Path(csv_path).name}` - **{len(texts)} comments**, "
         f"{sum(len(g) for g in gold)} gold (aspect, sentiment) pairs, "
         f"{int(mixed.sum())} mixed-sentiment comments.", ""]
    L += [f"> NOTE: {n}" for n in notes] + [""] if notes else []
    L += ["> CAVEAT: the default evaluation set is synthetic (built from held-out clauses with exact "
          "labels), so scores are optimistic. Use `--csv` with hand-labelled REAL comments for final "
          "numbers.", ""]

    # A. aspect detection
    rows = []
    for n in names:
        p, r, f = micro(asp_counts[n])
        macro = np.mean([x[2] for x in per_aspect_prf(gold, preds[n]).values()])
        rows.append([n, f"{p:.3f}", f"{r:.3f}", fmt_ci(f, asp_ci[n]), f"{macro:.3f}"])
    L += ["## A. Aspect detection (comment level)", "",
          md_table(["Pipeline", "Precision", "Recall", "micro-F1 [95% CI]", "macro-F1"], rows), ""]
    L += ["Paired difference in micro-F1 vs V1 (95% CI; CI containing 0 = not significant):", ""]
    L += [f"- {n}: {micro(asp_counts[n])[2] - micro(asp_counts[names[0]])[2]:+.3f} "
          f"[{asp_diff[n][0]:+.3f}, {asp_diff[n][1]:+.3f}]" for n in names[1:]] + [""]

    # per-aspect F1
    pa = {n: per_aspect_prf(gold, preds[n]) for n in names}
    L += ["### Per-aspect F1", "",
          md_table(["Aspect"] + names, [[ASPECT_DISPLAY_NAMES[a]] + [f"{pa[n][a][2]:.2f}" for n in names]
                                        for a in ASPECTS]), ""]

    # B/C/D sentiment + end-to-end + mixed
    rows = []
    for n in names:
        ms = matched_sentiment(gold, preds[n])
        p, r, f = micro(pair_counts[n])
        fm = micro(pair_counts[n][mixed])[2] if mixed.any() else float("nan")
        fn_ = micro(pair_counts[n][~mixed])[2] if (~mixed).any() else float("nan")
        rows.append([n, f"{ms['accuracy']:.3f}", f"{ms['macro_f1']:.3f}", f"{ms['neutral_rate']:.2f}",
                     fmt_ci(f, pair_ci[n]), f"{fm:.3f}", f"{fn_:.3f}"])
    L += ["## B-D. Sentiment and end-to-end (aspect + sentiment)", "",
          "Sentiment accuracy / macro-F1 use only aspects found in both gold and prediction "
          "(neutral = wrong). 'Neutral rate' = share of those predicted Neutral.", "",
          md_table(["Pipeline", "Sentiment acc.", "Sentiment macro-F1", "Neutral rate",
                    "End-to-end pair F1 [95% CI]", "pair F1 mixed", "pair F1 single-polarity"], rows), ""]
    L += ["Paired difference in end-to-end pair F1 vs V1:", ""]
    L += [f"- {n}: {micro(pair_counts[n])[2] - micro(pair_counts[names[0]])[2]:+.3f} "
          f"[{pair_diff[n][0]:+.3f}, {pair_diff[n][1]:+.3f}]" for n in names[1:]] + [""]

    # E. sentiment ablation on held-out clauses
    tc = pd.read_csv(TEST_CLAUSES_CSV)
    tc = tc[tc.aspect != "none"]
    items = list(zip(tc["clause"], tc["aspect"], tc["sentiment"]))
    variants = scorer_variants(use_transformer)
    rows, res = [], {}
    for name, sc in variants.items():
        res[name] = sentiment_on_clauses(sc, items)
        rows.append([name, f"{res[name]['accuracy']:.3f}", f"{res[name]['macro_f1']:.3f}",
                     f"{res[name]['neutral_rate']:.2f}"])
    L += ["## E. Sentiment ablation (held-out aspect clauses, gold aspect given)", "",
          f"{len(items)} clauses. Each row adds one layer.", "",
          md_table(["Scorer", "Accuracy", "Macro-F1", "Neutral rate"], rows), ""]

    # F. negation / sarcasm
    rows, errs = [], []
    neg_items = [(c, None, g) for c, g in NEGATION_SET]
    sar_items = [(c, None, g) for c, g in SARCASM_SET]
    for name, sc in variants.items():
        rn, rs = sentiment_on_clauses(sc, neg_items), sentiment_on_clauses(sc, sar_items)
        rows.append([name, f"{rn['accuracy']:.2f}", f"{rs['accuracy']:.2f}"])
        if name.startswith("4."):
            errs = [(c, g, p) for (c, g), p in zip(SARCASM_SET, rs["preds"]) if p != g]
            neg_errs = [(c, g, p) for (c, g), p in zip(NEGATION_SET, rn["preds"]) if p != g]
    L += [f"## F. Negation ({len(NEGATION_SET)} clauses) and sarcasm ({len(SARCASM_SET)} clauses: "
          "12 sarcastic + 12 genuine look-alikes)", "",
          md_table(["Scorer", "Negation accuracy", "Sarcasm accuracy"], rows), ""]
    L += ["Remaining mistakes of the sarcasm-aware VADER scorer (row 4):", ""]
    L += [f"- `{c}` gold={g}, predicted={p}" for c, g, p in neg_errs + errs] + [""]

    # G. error analysis
    best = names[-1]
    wrong = [(t, g, p) for t, g, p in zip(texts, gold, preds[best]) if set(g.items()) != set(p.items())]
    L += [f"## G. Error analysis - {best}", "", f"{len(wrong)} of {len(texts)} comments differ from gold. "
          "First 12:", ""]
    for t, g, p in wrong[:12]:
        miss, extra = set(g.items()) - set(p.items()), set(p.items()) - set(g.items())
        L += [f"- “{t}”", f"  - missed/wrong: {sorted(miss) or '-'} · unexpected: {sorted(extra) or '-'}"]
    L += ["", "## Limitations", "",
          "- Synthetic default data (see caveat); mini-sets are small and written by the system's author.",
          "- Gold sentiment has no Neutral class, so any Neutral prediction counts as an error.",
          "- Only the six aspects with keyword/ML triggers are detected; implicit mentions are missed.",
          "- No inter-annotator agreement: labels come from one author."]

    report = "\n".join(L)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(EVAL_COMMENTS_CSV), help="labelled comments CSV")
    ap.add_argument("--no-transformer", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    from opinion_mining import evaluation as _m           # package-name import (pickle safety)
    print(_m.run(a.csv, False if a.no_transformer else None, n_boot=a.boot))
    print(f"\n(report saved to {REPORTS_DIR / 'evaluation_report.md'})")
