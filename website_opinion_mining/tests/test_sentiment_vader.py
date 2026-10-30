import pytest
from opinion_mining.sentiment_vader import VaderSentimentScorer
from opinion_mining.pipeline import PipelineV1

scorer = VaderSentimentScorer()
pipe = PipelineV1()


@pytest.mark.parametrize("clause,expected", [
    ("Product was original", "Positive"),
    ("fake product", "Negative"),
    ("support never replied", "Negative"),
    ("It's not bad", "Positive"),            # negation of a negative
    ("It is not good", "Negative"),          # negation of a positive
    ("no value for money", "Negative"),      # must beat "value for money"
    ("value for money", "Positive"),
    ("Great price", "Positive"),
])
def test_clause_labels(clause, expected):
    assert scorer.score_clause(clause).label == expected


def test_delivery_days_rule():
    assert scorer.score_clause("delivery took 12 days", "delivery").label == "Negative"
    assert scorer.score_clause("delivery took 2 days", "delivery").label == "Positive"
    # the rule must NOT fire for other aspects ("30 days return policy")
    assert scorer.score_clause("30 days return policy", "refund").label == "Neutral"


def test_brief_example():
    r = pipe.analyze("Product was original but delivery took 12 days and support never replied.")
    assert r.as_dict() == {"genuineness": "Positive", "delivery": "Negative", "support": "Negative"}
    assert r.is_mixed


def test_mixed_sentiment():
    r = pipe.analyze("Great price but terrible customer service.")
    assert r.as_dict() == {"pricing": "Positive", "support": "Negative"}
