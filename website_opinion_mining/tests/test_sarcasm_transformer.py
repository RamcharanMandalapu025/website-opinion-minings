import pytest
from opinion_mining.pipeline import PipelineV1
from opinion_mining.preprocessing import preprocess
from opinion_mining.sarcasm import SarcasmAwareScorer, sarcasm_cue
from opinion_mining.sentiment_transformer import TransformerSentimentScorer, probs_to_score
from opinion_mining.sentiment_vader import VaderSentimentScorer

vader = VaderSentimentScorer()
sarc = SarcasmAwareScorer(vader)


# ------------------------------------------------------------- sarcasm ----
@pytest.mark.parametrize("clause", [
    "Oh great, another late delivery",
    "Wow thanks for nothing",
    "Fantastic, it broke on day one",
    "Yeah right, original",
    "Love how support never replies",
    "Perfect, my parcel is lost",
])
def test_sarcastic_clauses_become_negative(clause):
    assert sarc.score_clause(clause).label == "Negative"


@pytest.mark.parametrize("clause", [
    "Great product, never had any issues",
    "Another great purchase from this site",
    "Excellent quality",
    "Great, no more delays",
    "Thanks for the quick delivery",
    "Fast delivery and wonderful packaging",
])
def test_genuine_praise_is_not_flipped(clause):
    assert sarc.score_clause(clause).label == "Positive"
    assert sarcasm_cue(clause) is None


def test_sarcasm_wrapper_actually_changes_the_baseline():
    # Without the wrapper VADER is fooled by "great"; with it, it is not.
    clause = "Oh great, another late delivery"
    assert vader.score_clause(clause, "delivery").label != "Negative"
    assert sarc.score_clause(clause, "delivery").label == "Negative"


def test_interjection_not_split_from_its_clause():
    assert preprocess("Oh great, another late delivery") == ["Oh great, another late delivery"]
    # normal comma splitting still works
    assert preprocess("Great price, fast delivery") == ["Great price", "fast delivery"]


def test_eyeroll_emoji_is_a_sarcasm_marker():
    r = PipelineV1(scorer=sarc).analyze("Great delivery 🙄")
    assert r.as_dict()["delivery"] == "Negative"


def test_pipeline_end_to_end_with_sarcasm():
    r = PipelineV1(scorer=sarc).analyze("Oh great, another late delivery.")
    assert r.as_dict() == {"delivery": "Negative"}


# --------------------------------------------- transformer (fake model) ----
class FakeClassifier:
    """Stands in for the HF pipeline: returns canned probabilities, counts calls."""
    def __init__(self, p_pos):
        self.p_pos, self.calls = p_pos, []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[{"label": "POSITIVE", "score": self.p_pos},
                 {"label": "NEGATIVE", "score": 1 - self.p_pos}] for _ in texts]


def test_probs_to_score_binary_and_three_class():
    assert probs_to_score([{"label": "POSITIVE", "score": 0.9}, {"label": "NEGATIVE", "score": 0.1}]) == pytest.approx(0.8)
    assert probs_to_score([{"label": "negative", "score": 0.7}, {"label": "neutral", "score": 0.2},
                           {"label": "positive", "score": 0.1}]) == pytest.approx(-0.6)
    assert probs_to_score([{"label": "NEGATIVE", "score": 0.9}]) == pytest.approx(-0.8)  # top-1 only
    with pytest.raises(ValueError):
        probs_to_score([{"label": "LABEL_0", "score": 1.0}])


@pytest.mark.parametrize("p_pos,label", [(0.99, "Positive"), (0.01, "Negative"), (0.6, "Neutral")])
def test_label_mapping_and_neutral_band(p_pos, label):
    s = TransformerSentimentScorer(classifier=FakeClassifier(p_pos))
    assert s.score_clause("some clause").label == label


def test_rules_run_before_the_model():
    fake = FakeClassifier(0.99)                        # model would say Positive...
    s = TransformerSentimentScorer(classifier=fake)
    r = s.score_clause("support never replied")        # ...but our phrase rule knows better
    assert r.label == "Negative" and r.rule == "phrase"
    assert fake.calls == []                            # model was never even called


def test_batching_sends_one_call_for_all_non_rule_clauses():
    fake = FakeClassifier(0.9)
    s = TransformerSentimentScorer(classifier=fake)
    out = s.score_clauses(["nice colour", "support never replied", "solid build"])
    assert len(fake.calls) == 1 and fake.calls[0] == ["nice colour", "solid build"]
    assert [o.rule for o in out] == ["transformer", "phrase", "transformer"]


def test_delivery_days_rule_used_when_model_is_unsure():
    s = TransformerSentimentScorer(classifier=FakeClassifier(0.55))   # unsure -> Neutral
    assert s.score_clause("it took 14 days", "delivery").label == "Negative"
    assert s.score_clause("it took 14 days", "refund").label == "Neutral"


def test_sarcasm_wrapper_works_on_transformer_base():
    base = TransformerSentimentScorer(classifier=FakeClassifier(0.97))  # fooled by "great"
    r = SarcasmAwareScorer(base).score_clause("Oh great, another late delivery", "delivery")
    assert r.label == "Negative" and r.rule.startswith("sarcasm")
