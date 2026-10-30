"""
Step 5b: heuristic sarcasm handling.

WHY IT'S NEEDED: sentiment models read words, not intent.
    "Oh great, another late delivery"  ->  'great' makes a model say POSITIVE.
Sarcasm in reviews has a recognisable shape: a POSITIVE opener sitting next to
a NEGATIVE event. We detect that conflict and flip the sign.

Two cues (both only ever flip a POSITIVE result to NEGATIVE):
  1. HARD cue   : stock sarcastic phrases ("thanks for nothing", "yeah right",
                  an eye-roll emoji ...)  -> always negative.
  2. CONFLICT   : positive opener ("great", "wow", "love how" ...) AND a
                  negative event in the same clause ("late", "broke", "never
                  replies" ...).
A safe-list prevents false flips on genuine praise ("Great, no more delays").

HONEST LIMITS (mention in viva): clause-level only, English only, no real
understanding. "Great job. It took 3 weeks." (split over two sentences) or
"Fantastic, only 3 weeks!" are missed. A fine-tuned sarcasm model would be the
proper upgrade (listed in the improvements at the end).
"""
import re
from typing import Optional

from .sentiment_vader import ClauseSentiment, phrase_score, POS_THRESHOLD

_HARD = re.compile(
    r"\b(yeah,? right|thanks for nothing|so much for|what a joke|nice try|oh sure|"
    r"how (?:wonderful|lovely|nice)|eyeroll)\b", re.I)

_OPENER = re.compile(
    r"\b(great|wow|fantastic|wonderful|amazing|perfect|brilliant|lovely|awesome|"
    r"excellent|nice|thanks|thank you|love (?:how|that|it when)|just what i needed|"
    r"exactly what i wanted|well done|good job|bravo|congrats)\b", re.I)

# Negative EVENTS only. Bare negators ("never", "not") and words like "another"
# are NOT cues: "Great product, never had issues" / "Another great buy" are genuine.
_NEG_EVENT = re.compile(
    r"\b(late|delay\w*|slow|broke\w*|broken|damaged|fake|wrong item|lost|ignored|"
    r"useless|worst|waste|stopped working|scam|rude|defective|faulty)\b", re.I)

# Genuine praise that happens to contain a negative word.
_SAFE = re.compile(
    r"\b(no more|not (?:late|slow|broken|damaged|fake)|no (?:delay\w*|issues?|problems?)|"
    r"without (?:any )?(?:delay|issue|problem)\w*|never (?:late|slow|broke\w*|damaged))\b", re.I)


def sarcasm_cue(clause: str) -> Optional[str]:
    """Return 'hard', 'conflict' or None."""
    if _HARD.search(clause):
        return "hard"
    if _SAFE.search(clause):
        return None
    if _OPENER.search(clause):
        p = phrase_score(clause)
        if _NEG_EVENT.search(clause) or (p is not None and p < 0):
            return "conflict"
    return None


class SarcasmAwareScorer:
    """Wrap any scorer that has .score_clause(clause, aspect) -> ClauseSentiment."""

    def __init__(self, base, label_fn=None):
        self.base = base
        # label_fn lets the wrapper reuse the base scorer's own Neutral band.
        self.label_fn = label_fn or getattr(base, "label_for", None)

    def score_clause(self, clause: str, aspect: Optional[str] = None) -> ClauseSentiment:
        r = self.base.score_clause(clause, aspect)
        cue = sarcasm_cue(clause)
        if cue is None or r.score < 0:
            return r                                   # nothing to fix
        if cue == "hard" or r.score >= POS_THRESHOLD or r.label == "Positive":
            new = -max(abs(r.score), 0.5)              # flip sign, keep at least -0.5
            label = "Negative"
            return ClauseSentiment(clause, round(new, 3), label, f"sarcasm({cue})")
        return r
