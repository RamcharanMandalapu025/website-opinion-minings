"""
Version 1 aspect detector: keyword / lexicon matching.

Idea: for each aspect we have a list of trigger words. We compile them
into ONE regular expression per aspect and scan the text. Any hit means
"this aspect is mentioned".

Why it's a good baseline:
  + fast, no training data, fully explainable (you can show WHICH word fired)
Why we later replace it (Version 2):
  - misses paraphrases ("never got back to me" -> support)
  - ambiguous words ("service", "days", "product") cause false positives
"""
import re
from dataclasses import dataclass
from typing import Dict, List

from .config import ASPECT_KEYWORDS
from .preprocessing import preprocess

# Allow simple inflections after a keyword: deliver -> delivered/delivering/
# delivery, price -> prices/priced/pricey, late -> lately. We deliberately do
# NOT use "\w*", otherwise "late" would wrongly match "latest" or "later".
_SUFFIXES = r"(?:s|es|d|ed|ing|ly|y|ies)?"


@dataclass(frozen=True)
class AspectMatch:
    """One detected aspect mention, with evidence for explainability."""
    aspect: str      # e.g. "delivery"
    keyword: str     # the text that matched, e.g. "delivered"
    start: int       # character offset in the input text
    end: int


def _compile_patterns(lexicon: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    """Build one case-insensitive regex per aspect from its keyword list."""
    patterns = {}
    for aspect, words in lexicon.items():
        # Longest first so "customer service" wins over "service".
        words = sorted(set(words), key=len, reverse=True)
        parts = []
        for w in words:
            # Spaces/hyphens in a phrase match flexibly ("knock-off", "knock off").
            # Use placeholders so the two substitutions don't clobber each other.
            escaped = (re.escape(w.lower())
                       .replace(r"\-", "@@HY@@")
                       .replace(r"\ ", "@@SP@@")
                       .replace("@@HY@@", r"[\s\-]?")
                       .replace("@@SP@@", r"[\s\-]+"))
            parts.append(escaped)
        # \b = word boundary, so we match whole words only.
        patterns[aspect] = re.compile(
            r"\b(?:" + "|".join(parts) + r")" + _SUFFIXES + r"\b",
            flags=re.IGNORECASE,
        )
    return patterns


class KeywordAspectDetector:
    """Detects which of the six aspects are mentioned in a piece of text."""

    def __init__(self, lexicon: Dict[str, List[str]] = None):
        self.lexicon = lexicon or ASPECT_KEYWORDS
        self.patterns = _compile_patterns(self.lexicon)

    def detect(self, text: str) -> List[AspectMatch]:
        """Return every keyword hit (an aspect may appear several times)."""
        matches: List[AspectMatch] = []
        for aspect, pattern in self.patterns.items():
            for m in pattern.finditer(text):
                matches.append(AspectMatch(aspect, m.group(0), m.start(), m.end()))
        return sorted(matches, key=lambda x: x.start)

    def detect_aspects(self, text: str) -> List[str]:
        """Return just the distinct aspect names found (order preserved)."""
        seen = []
        for m in self.detect(text):
            if m.aspect not in seen:
                seen.append(m.aspect)
        return seen

    def detect_by_clause(self, raw_comment: str) -> List[tuple]:
        """
        Full Step-1 + Step-2 pipeline.
        Returns [(clause_text, [aspects in that clause]), ...] so each aspect
        can later receive its OWN sentiment score.
        """
        return [(c, self.detect_aspects(c)) for c in preprocess(raw_comment)]


if __name__ == "__main__":
    detector = KeywordAspectDetector()
    demo = [
        "Product was original but delivery took 12 days and support never replied.",
        "Great price, but the refund process was a nightmare.",
        "The latest model feels sturdy.",   # 'latest' must NOT trigger 'late'
    ]
    for sentence in demo:
        print(sentence)
        for clause, aspects in detector.detect_by_clause(sentence):
            print(f"   [{', '.join(aspects) or '-'}]  <-  '{clause}'")
