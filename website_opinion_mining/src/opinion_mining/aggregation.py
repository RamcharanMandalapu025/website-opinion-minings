"""
Step 6: turn many (aspect, sentiment) records into star ratings + trust score.

Pipeline of ideas (each is a viva question):
  1. score -> stars       linear map, saturating:  stars = 3 + 2*clip(score/0.6, -1, 1)
                          (saturation lets VADER and the transformer share one scale)
  2. Bayesian smoothing   stars = (k*prior + sum(w*stars)) / (k + sum(w))
                          One lucky 5-star comment must NOT give a site 5.0.
                          k = "pseudo-reviews" of prior opinion (default 5, prior 3.0).
  3. confidence           c = n / (n + k)   ->  Low (<0.5) / Medium / High (>=0.75)
                          n=1 -> 0.17 | n=5 -> 0.50 | n=15 -> 0.75 | n=50 -> 0.91
  4. trust score (0-100)  importance- AND confidence-weighted mean of aspect stars,
                          rescaled from 1-5 to 0-100, plus a genuineness veto.
  5. weights              each record has a weight in [0,1] (Step 7 passes
                          1 - spam_probability), so suspect comments count less.

Input is plain AspectRecord objects (not pipeline objects) so the same code works
on rows loaded from SQLite in Step 8.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .config import (ASPECT_DISPLAY_NAMES, ASPECT_IMPORTANCE, ASPECTS,
                     GENUINENESS_VETO_MIN_CONFIDENCE, GENUINENESS_VETO_STARS,
                     PRIOR_MEAN, PRIOR_STRENGTH, SCORE_SATURATION,
                     TRUST_CAP_ON_VETO)


# ----------------------------------------------------------------- records --
@dataclass(frozen=True)
class AspectRecord:
    aspect: str
    score: float                    # sentiment score in [-1, 1]
    label: str = ""                 # Positive / Neutral / Negative (derived if empty)
    weight: float = 1.0             # 0 = ignore, 1 = fully trusted
    comment_id: Optional[object] = None


def records_from_analysis(analysis, weight: float = 1.0, comment_id=None) -> List[AspectRecord]:
    """CommentAnalysis (from pipeline.py) -> list of AspectRecord."""
    return [AspectRecord(a.aspect, a.score, a.label, weight, comment_id)
            for a in analysis.aspects]


# --------------------------------------------------------------- the maths --
def score_to_stars(score: float, saturation: float = SCORE_SATURATION) -> float:
    s = max(-1.0, min(1.0, score / saturation))
    return 3.0 + 2.0 * s


def confidence_of(n_effective: float, k: float = PRIOR_STRENGTH) -> float:
    return n_effective / (n_effective + k) if n_effective > 0 else 0.0


def confidence_label(c: float) -> str:
    if c <= 0:
        return "No data"
    return "High" if c >= 0.75 else "Medium" if c >= 0.5 else "Low"


def render_stars(stars: Optional[float]) -> str:
    """4.3 -> '★★★★☆ 4.3'  (rounded to whole stars for display)."""
    if stars is None:
        return "no ratings yet"
    full = int(round(stars))
    return "★" * full + "☆" * (5 - full) + f" {stars:.1f}"


def _label_of(r: AspectRecord) -> str:
    if r.label:
        return r.label
    stars = score_to_stars(r.score)
    return "Positive" if stars >= 3.5 else "Negative" if stars <= 2.5 else "Neutral"


# ------------------------------------------------------------------ output --
@dataclass
class AspectRating:
    aspect: str
    stars: Optional[float]          # smoothed 1-5 (None = no data)
    raw_stars: Optional[float]      # plain weighted mean, shown for transparency
    n: float                        # effective number of mentions (sum of weights)
    n_pos: int = 0
    n_neu: int = 0
    n_neg: int = 0
    confidence: float = 0.0
    confidence_label: str = "No data"

    @property
    def display_name(self) -> str:
        return ASPECT_DISPLAY_NAMES[self.aspect]


@dataclass
class WebsiteRating:
    aspects: Dict[str, AspectRating]
    trust_score: Optional[float]    # 0-100, None if no data
    trust_stars: Optional[float]
    n_comments: int
    confidence: float
    confidence_label: str
    veto_applied: bool = False

    def to_rows(self) -> List[dict]:
        """Flat rows for Streamlit tables / Plotly charts."""
        return [{"aspect": a.display_name, "stars": a.stars, "mentions": round(a.n, 1),
                 "positive": a.n_pos, "neutral": a.n_neu, "negative": a.n_neg,
                 "confidence": a.confidence_label} for a in self.aspects.values()]


# -------------------------------------------------------------- aggregation --
def aggregate(records: Iterable[AspectRecord],
              prior_mean: float = PRIOR_MEAN, prior_strength: float = PRIOR_STRENGTH,
              importance: Optional[Dict[str, float]] = None,
              n_comments: Optional[int] = None) -> WebsiteRating:
    importance = importance or ASPECT_IMPORTANCE
    records = [r for r in records if r.weight > 0]
    by_aspect: Dict[str, List[AspectRecord]] = defaultdict(list)
    for r in records:
        by_aspect[r.aspect].append(r)

    ratings: Dict[str, AspectRating] = {}
    for aspect in ASPECTS:
        rs = by_aspect.get(aspect, [])
        w = sum(r.weight for r in rs)
        if w == 0:
            ratings[aspect] = AspectRating(aspect, None, None, 0.0)
            continue
        raw = sum(r.weight * score_to_stars(r.score) for r in rs) / w
        smoothed = (prior_strength * prior_mean + w * raw) / (prior_strength + w)
        labels = [_label_of(r) for r in rs]
        c = confidence_of(w, prior_strength)
        ratings[aspect] = AspectRating(
            aspect, round(smoothed, 2), round(raw, 2), round(w, 2),
            labels.count("Positive"), labels.count("Neutral"), labels.count("Negative"),
            round(c, 3), confidence_label(c))

    # ---- overall trust: importance x confidence weighted mean of aspect stars
    num = den = 0.0
    for a, r in ratings.items():
        if r.stars is None:
            continue
        wt = importance[a] * r.confidence
        num += wt * r.stars
        den += wt
    if den == 0:
        trust_stars = trust = None
    else:
        trust_stars = num / den
        trust = (trust_stars - 1) / 4 * 100

    veto = False
    g = ratings["genuineness"]
    if (trust is not None and g.stars is not None and g.stars <= GENUINENESS_VETO_STARS
            and g.confidence >= GENUINENESS_VETO_MIN_CONFIDENCE and trust > TRUST_CAP_ON_VETO):
        trust, veto = TRUST_CAP_ON_VETO, True

    if n_comments is None:
        ids = {r.comment_id for r in records if r.comment_id is not None}
        n_comments = len(ids) if ids else int(max((a.n for a in ratings.values()), default=0))
    oc = confidence_of(n_comments, prior_strength)
    return WebsiteRating(
        ratings, None if trust is None else round(trust, 1),
        None if trust_stars is None else round(trust_stars, 2),
        n_comments, round(oc, 3), confidence_label(oc), veto)


# ------------------------------------------------- top positive / negative --
def comment_polarity(analysis) -> Optional[float]:
    """Mean sentiment over the aspects of one comment (None if no aspects)."""
    if not analysis.aspects:
        return None
    return sum(a.score for a in analysis.aspects) / len(analysis.aspects)


def rank_comments(analyses: Iterable, top_n: int = 3) -> Tuple[List[dict], List[dict]]:
    """Return (top_positive, top_negative). Ties -> comment covering more aspects."""
    scored = []
    for an in analyses:
        p = comment_polarity(an)
        if p is not None:
            scored.append({"comment": an.comment, "polarity": round(p, 3),
                           "aspects": an.as_dict(), "n_aspects": len(an.aspects)})
    pos = sorted((s for s in scored if s["polarity"] > 0),
                 key=lambda s: (-s["polarity"], -s["n_aspects"]))[:top_n]
    neg = sorted((s for s in scored if s["polarity"] < 0),
                 key=lambda s: (s["polarity"], -s["n_aspects"]))[:top_n]
    return pos, neg


# ------------------------------------------------------------------- demo ---
if __name__ == "__main__":
    import json
    import pandas as pd
    from .config import EVAL_COMMENTS_CSV
    from .pipeline import build_pipeline

    pipe = build_pipeline("v2", transformer=False)       # VADER base: no download needed
    ev = pd.read_csv(EVAL_COMMENTS_CSV)
    # Fake 3 websites using the ground-truth sentiment mix of each comment.
    good, bad = [], []
    for text, truth in zip(ev["comment"], ev["aspects"]):
        vals = list(json.loads(truth).values())
        (good if vals.count("Positive") > vals.count("Negative") else bad).append(text)
    sites = {"GoodMart (many positive)": good, "BadBazaar (many negative)": bad,
             "NewShop (only 2 comments)": good[:2]}

    for name, comments in sites.items():
        analyses = [pipe.analyze(c) for c in comments]
        recs = [r for i, an in enumerate(analyses) for r in records_from_analysis(an, comment_id=i)]
        w = aggregate(recs)
        print(f"\n=== {name}: {len(comments)} comments | trust "
              f"{w.trust_score}/100 | evidence: {w.confidence_label}"
              f"{' | GENUINENESS VETO' if w.veto_applied else ''}")
        for a in w.aspects.values():
            print(f"  {a.display_name:<18}{render_stars(a.stars):<18} n={a.n:<5} "
                  f"(+{a.n_pos}/={a.n_neu}/-{a.n_neg})  confidence: {a.confidence_label}")
