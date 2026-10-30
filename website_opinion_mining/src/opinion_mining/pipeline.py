"""
Version-1 pipeline: raw comment -> {aspect: sentiment}.

   raw text --preprocess--> clauses --KeywordAspectDetector--> aspects per clause
                                    --VaderSentimentScorer---> score per clause
   then: group clause scores by aspect (average if an aspect appears twice).

MIXED SENTIMENT is handled structurally: because every clause is scored
separately, "Great price but terrible support" yields Pricing=Positive AND
Support=Negative instead of one muddled overall score.
"""
from dataclasses import dataclass, field
from typing import Dict, List

from .aspect_keyword import KeywordAspectDetector
from .sentiment_vader import VaderSentimentScorer, to_label


@dataclass
class AspectSentiment:
    aspect: str
    score: float
    label: str
    evidence: List[str] = field(default_factory=list)   # clauses used


@dataclass
class CommentAnalysis:
    comment: str
    aspects: List[AspectSentiment]

    @property
    def is_mixed(self) -> bool:
        labels = {a.label for a in self.aspects}
        return "Positive" in labels and "Negative" in labels

    def as_dict(self) -> Dict[str, str]:
        return {a.aspect: a.label for a in self.aspects}


class PipelineV1:
    """Defaults = Version 1. Pass other detectors/scorers to upgrade parts
    (e.g. detector=MLAspectDetector() for Version 2 aspect detection)."""

    def __init__(self, detector=None, scorer=None):
        self.detector = detector or KeywordAspectDetector()
        self.scorer = scorer or VaderSentimentScorer()

    def analyze(self, raw_comment: str) -> CommentAnalysis:
        scores: Dict[str, List[float]] = {}
        evidence: Dict[str, List[str]] = {}
        for clause, aspects in self.detector.detect_by_clause(raw_comment):
            for aspect in aspects:
                s = self.scorer.score_clause(clause, aspect).score
                scores.setdefault(aspect, []).append(s)
                evidence.setdefault(aspect, []).append(clause)
        result = []
        for aspect, vals in scores.items():
            mean = sum(vals) / len(vals)
            result.append(AspectSentiment(aspect, round(mean, 3),
                                          to_label(mean), evidence[aspect]))
        return CommentAnalysis(raw_comment, result)


if __name__ == "__main__":
    p = PipelineV1()
    for text in [
        "Product was original but delivery took 12 days and support never replied.",
        "Great price but terrible customer service.",
        "Not bad at all, quick delivery!",
        "They sent a fake item and gave no refund.",
    ]:
        a = p.analyze(text)
        print(text)
        print("   ->", a.as_dict(), "| mixed" if a.is_mixed else "")


def build_pipeline(version: str = "v2", transformer: bool = True, sarcasm: bool = True):
    """
    version="v1": keyword aspects + VADER.
    version="v2": trained aspect classifier + (transformer | VADER) + sarcasm layer.
    transformer=False lets you run V2 aspects with VADER sentiment (no downloads).
    """
    if version == "v1":
        return PipelineV1()
    from .aspect_ml import MLAspectDetector
    from .sarcasm import SarcasmAwareScorer
    if transformer:
        from .sentiment_transformer import TransformerSentimentScorer
        base = TransformerSentimentScorer()
    else:
        base = VaderSentimentScorer()
    return PipelineV1(detector=MLAspectDetector(),
                      scorer=SarcasmAwareScorer(base) if sarcasm else base)
