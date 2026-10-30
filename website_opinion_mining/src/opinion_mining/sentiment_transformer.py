"""
Step 5a: Version-2 sentiment = Hugging Face transformer (+ our domain rules).

Default model: distilbert-base-uncased-finetuned-sst-2-english (~250 MB)
  * DistilBERT = BERT compressed to ~60% size / ~2x faster, keeps ~97% accuracy.
  * Fine-tuned on SST-2 (movie-review sentences) -> binary POSITIVE/NEGATIVE.
  * Unlike VADER it reads WORD ORDER and CONTEXT ("not bad", "bad? not at all").

Converting probabilities to our scale:
      score = P(positive) - P(negative)        in [-1, +1]
A binary model never says "neutral" and is over-confident, so we call a clause
Neutral when |score| < NEUTRAL_BAND (i.e. the model is < 75% sure).
(For a native 3-class model use e.g. cardiffnlp/twitter-roberta-base-sentiment-latest;
 labels named positive/neutral/negative are handled automatically.)

Domain rules from Step 3 still run FIRST: phrase rules ("never replied") and the
delivery-days rule are things we KNOW, so we don't ask a movie-review model.

The classifier is injectable, which keeps this class testable without
downloading anything:  TransformerSentimentScorer(classifier=my_function)
"""
from typing import Callable, List, Optional

from .sentiment_vader import (ClauseSentiment, delivery_days_score, phrase_score,
                              to_label)

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
NEUTRAL_BAND = 0.5


def probs_to_score(preds: List[dict]) -> float:
    """HF output [{'label':'POSITIVE','score':0.98}, ...] -> P(pos) - P(neg)."""
    p_pos = p_neg = 0.0
    for d in preds:
        label = d["label"].lower()
        if label.startswith("pos"):
            p_pos = d["score"]
        elif label.startswith("neg"):
            p_neg = d["score"]
        elif label.startswith("neu"):
            pass
        else:
            raise ValueError(f"Unrecognised label '{d['label']}'. Use a model whose labels "
                             "are named positive/negative(/neutral).")
    # Binary models may return only the top label: infer the other side.
    if len(preds) == 1:
        p_pos, p_neg = (preds[0]["score"], 1 - preds[0]["score"]) if p_pos else \
                       (1 - preds[0]["score"], preds[0]["score"])
    return p_pos - p_neg


class TransformerSentimentScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 classifier: Optional[Callable] = None, use_rules: bool = True,
                 neutral_band: float = NEUTRAL_BAND, device: int = -1):
        self.model_name, self.use_rules = model_name, use_rules
        self.neutral_band, self.device = neutral_band, device
        self._clf = classifier            # lazy: loaded on first use if None

    # -- model loading (lazy so importing this module is instant) -----------
    def _classifier(self):
        if self._clf is None:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise ImportError("Transformer sentiment needs:  pip install transformers torch") from e
            # top_k=None -> return probabilities for ALL labels, not just the best
            self._clf = pipeline("sentiment-analysis", model=self.model_name,
                                 top_k=None, truncation=True, device=self.device)
        return self._clf

    def label_for(self, score: float) -> str:
        if score >= self.neutral_band:
            return "Positive"
        if score <= -self.neutral_band:
            return "Negative"
        return "Neutral"

    # -- scoring -------------------------------------------------------------
    def score_clauses(self, clauses: List[str], aspects: Optional[List[Optional[str]]] = None
                      ) -> List[ClauseSentiment]:
        """Batch scoring: rules first, then ONE batched model call for the rest."""
        aspects = aspects or [None] * len(clauses)
        out: List[Optional[ClauseSentiment]] = [None] * len(clauses)
        todo = []
        for i, c in enumerate(clauses):
            p = phrase_score(c) if self.use_rules else None
            if p is not None:
                out[i] = ClauseSentiment(c, round(p, 3), to_label(p), "phrase")
            else:
                todo.append(i)
        if todo:
            raw = self._classifier()([clauses[i] for i in todo])
            if raw and isinstance(raw[0], dict):      # some versions flatten a single input
                raw = [raw]
            for i, preds in zip(todo, raw):
                score = probs_to_score(preds)
                label = self.label_for(score)
                if self.use_rules and aspects[i] == "delivery" and label == "Neutral":
                    d = delivery_days_score(clauses[i])
                    if d is not None:
                        out[i] = ClauseSentiment(clauses[i], d, to_label(d), "delivery_days")
                        continue
                out[i] = ClauseSentiment(clauses[i], round(score, 3), label, "transformer")
        return out

    def score_clause(self, clause: str, aspect: Optional[str] = None) -> ClauseSentiment:
        return self.score_clauses([clause], [aspect])[0]


if __name__ == "__main__":
    # SMOKE TEST: first run downloads ~250 MB to ~/.cache/huggingface
    from .sarcasm import SarcasmAwareScorer
    scorer = SarcasmAwareScorer(TransformerSentimentScorer())
    for c in ["Product was original", "support never replied", "It's not bad",
              "delivery took 12 days", "Oh great, another late delivery",
              "Customer service was wonderful", "The refund process was a nightmare"]:
        r = scorer.score_clause(c, "delivery" if "deliver" in c else None)
        print(f"{r.label:<9}{r.score:>7}  [{r.rule:<16}] {c}")
