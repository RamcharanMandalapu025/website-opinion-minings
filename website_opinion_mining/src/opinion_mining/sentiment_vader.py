"""
Step 3: Version-1 sentiment = VADER + domain tweaks.

VADER is a lexicon + rule based scorer: each word has a valence (-4..+4),
and rules adjust for negation ("not good"), intensifiers ("very"),
CAPS and "!". It outputs a `compound` score in [-1, +1].

Problem: VADER was built for social media, not e-commerce, so on its own it
scores "Product was original" or "support never replied" as neutral (0).
We fix this with three layers (applied in this order):

  1. PHRASE RULES   - regexes for multi-word patterns VADER cannot see
                      ("never replied", "not bad", "no refund").
  2. DOMAIN LEXICON - e-commerce words added to VADER's dictionary
                      ("original", "fake", "late", "sturdy" ...).
  3. DELIVERY-DAYS  - numeric rule: "took 12 days" is bad, "2 days" is good.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POS_THRESHOLD = 0.15      # compound >= +0.15 -> Positive
NEG_THRESHOLD = -0.15     # compound <= -0.15 -> Negative, else Neutral

# ---- Layer 2: words VADER doesn't know (valence scale -4 .. +4) ----------
DOMAIN_LEXICON = {
    # genuineness
    "original": 1.5, "genuine": 1.5, "authentic": 2.0, "legit": 1.5,
    "legitimate": 1.5, "fake": -2.5, "counterfeit": -3.0, "duplicate": -1.5,
    "replica": -1.5, "knockoff": -2.5, "forged": -2.5, "scam": -3.0,
    "fraud": -3.0, "imitation": -2.0,
    # delivery
    "late": -1.5, "delayed": -2.0, "delay": -1.5, "delays": -1.5,
    "slow": -1.5, "fast": 1.5, "quick": 1.5, "quickly": 1.2, "prompt": 1.5,
    "promptly": 1.5, "lost": -1.5, "undelivered": -2.5,
    # quality
    "sturdy": 1.5, "durable": 1.5, "flimsy": -2.0, "defective": -2.5,
    "faulty": -2.0, "damaged": -2.0, "broken": -2.0,
    # support
    "unresponsive": -2.0, "helpful": 2.0,
    # pricing
    "expensive": -1.5, "overpriced": -2.0, "costly": -1.5,
    "affordable": 1.5, "cheap": 0.8, "bargain": 1.5,
}

# ---- Layer 1: phrase rules (regex, score in [-1, 1]) ----------------------
PHRASE_RULES: List[Tuple[str, float]] = [
    # --- positive idioms that look negative to a word-level scorer
    (r"\bnot (?:too |that |so |really )?bad\b", 0.4),
    (r"\bno (?:complaints?|issues?|problems?)\b", 0.5),
    (r"\bon time\b", 0.5),
    # Negated version must be a LONGER match so it overrides "on time" (found in Step 10:
    # "Delivery was not on time" used to score Positive).
    (r"\b(?:not|never|hardly)\s+(?:\w+\s+){0,2}on time\b", -0.6),
    (r"\bvalue for money\b", 0.5),
    (r"\b(?:got|received) (?:my |the |a )?(?:full )?refund\b", 0.5),
    (r"\brefund(?:ed)? (?:was )?(?:processed|credited|quick\w*)\b", 0.4),
    # --- negative phrases: "never replied", "no refund" ...
    (r"\b(?:did not|does not|do not|will not|cannot|never|not)\s+"
     r"(?:repl\w*|respond\w*|answer\w*|refund\w*|resolv\w*|receiv\w*|"
     r"arriv\w*|deliver\w*|return\w*|replac\w*|help\w*)", -0.7),
    (r"\b(?:no|zero) (?:reply|response|support|help|refund)\b", -0.7),
    (r"\bnever got back\b", -0.7),
    (r"\bnever again\b", -0.8),
    (r"\bwaste of (?:money|time)\b", -0.8),
    (r"\bnot (?:original|genuine|authentic|legit|worth)\b", -0.7),
    (r"\b(?:no|not) value for money\b", -0.6),
    (r"\bstill waiting\b", -0.6),
    (r"\bnot as (?:described|advertised|shown|expected)\b", -0.6),
    (r"\b(?:cheap|cheaply)(?: quality| made| build| material)\b", -0.7),
]
_PHRASES = [(re.compile(p, re.I), s) for p, s in PHRASE_RULES]

# ---- Layer 3: delivery duration -------------------------------------------
_DAYS = re.compile(r"\b(\d{1,3})\s*(?:business\s+|working\s+)?days?\b", re.I)


def _delivery_days_score(clause: str) -> Optional[float]:
    m = _DAYS.search(clause)
    if not m:
        return None
    days = int(m.group(1))
    if days <= 3:
        return 0.5
    if days >= 10:
        return -0.7
    if days >= 7:
        return -0.5
    return None          # 4-6 days: leave neutral, it's a judgement call


def phrase_score(clause: str) -> Optional[float]:
    """Layer-1 phrase rules. Returns a score in [-1, 1] or None if no rule fires."""
    hits = [(m.start(), m.end(), s) for rx, s in _PHRASES for m in rx.finditer(clause)]
    # Drop a hit that sits INSIDE a longer hit: "no value for money"
    # (negative) must beat "value for money" (positive).
    hits = [h for h in hits
            if not any(o is not h and o[0] <= h[0] and h[1] <= o[1]
                       and (o[1] - o[0]) > (h[1] - h[0]) for o in hits)]
    if not hits:
        return None
    return sum(h[2] for h in hits) / len(hits)


delivery_days_score = _delivery_days_score      # public alias (used by Step 5)


def to_label(score: float) -> str:
    if score >= POS_THRESHOLD:
        return "Positive"
    if score <= NEG_THRESHOLD:
        return "Negative"
    return "Neutral"


@dataclass
class ClauseSentiment:
    clause: str
    score: float      # -1 .. +1
    label: str        # Positive / Neutral / Negative
    rule: str         # which layer decided: "phrase" | "delivery_days" | "vader"


class VaderSentimentScorer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.analyzer.lexicon.update(DOMAIN_LEXICON)      # Layer 2

    def _phrase_score(self, clause: str) -> Optional[float]:
        return phrase_score(clause)

    def score_clause(self, clause: str, aspect: Optional[str] = None) -> ClauseSentiment:
        # Layer 1: phrase rules take priority
        p = self._phrase_score(clause)
        if p is not None:
            return ClauseSentiment(clause, round(p, 3), to_label(p), "phrase")
        # Layer 2: VADER with the domain lexicon
        compound = self.analyzer.polarity_scores(clause)["compound"]
        # Layer 3: only when VADER found nothing and we know it's about delivery
        if aspect == "delivery" and abs(compound) < POS_THRESHOLD:
            d = _delivery_days_score(clause)
            if d is not None:
                return ClauseSentiment(clause, d, to_label(d), "delivery_days")
        return ClauseSentiment(clause, round(compound, 3), to_label(compound), "vader")


if __name__ == "__main__":
    scorer = VaderSentimentScorer()
    for c in ["Product was original", "support never replied", "It's not bad",
              "It is not good", "delivery took 12 days", "delivery took 2 days",
              "fake product", "Great price", "no value for money"]:
        r = scorer.score_clause(c, aspect="delivery" if "deliver" in c else None)
        print(f"{r.label:<9}{r.score:>7}  [{r.rule:<13}] {c}")
