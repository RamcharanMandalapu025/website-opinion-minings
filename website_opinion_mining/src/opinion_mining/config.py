"""
Central configuration: paths + the aspect keyword lexicon.

Keeping the lexicon here (not buried inside the detector) means you can
tune it without touching any logic -- handy to explain in a viva.
"""
from pathlib import Path

# ---- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DB_PATH = ROOT_DIR / "db" / "opinions.db"

# ---- Aspect lexicon ----------------------------------------------------
# Keys are the six aspects. Values are keywords/phrases that signal them.
# Matching is word-based (see aspect_keyword.py), so "late" will NOT match
# "latest". Simple suffixes (s, ed, ing, ly ...) are handled automatically.
ASPECT_KEYWORDS = {
    "genuineness": [
        "genuine", "authentic", "original", "fake", "counterfeit",
        "duplicate", "replica", "knockoff", "knock-off", "first copy",
        "imitation", "forged", "scam", "fraud", "authenticity",
        "not original", "branded", "legit", "legitimate",
    ],
    "delivery": [
        "delivery", "deliver", "delivered", "shipping", "shipment", "ship",
        "shipped", "dispatch", "dispatched", "courier", "arrive", "arrived",
        "late", "delay", "delayed", "on time", "packaging",
        "packed", "package", "parcel", "tracking", "transit",
    ],
    "quality": [
        "quality", "durable", "durability", "sturdy", "flimsy", "defective",
        "damaged", "broken", "broke", "stopped working", "build",
        "material", "cheaply made", "well made", "works great", "faulty",
        "performance",
    ],
    "support": [
        "support", "customer service", "customer care", "helpline",
        "service", "agent", "representative", "replied", "reply",
        "response", "respond", "responded", "complaint", "ticket",
        "helpful", "rude", "chat", "call",
    ],
    "refund": [
        "refund", "return", "returned", "replacement", "replace",
        "exchange", "money back", "cancel", "cancellation", "reimburse",
        "reimbursement", "return policy", "refunded", "warranty",
    ],
    "pricing": [
        "price", "pricing", "priced", "cost", "expensive", "cheap",
        "affordable", "discount", "offer", "deal", "value for money",
        "overpriced", "worth", "bargain", "costly", "coupon", "sale",
    ],
}

# Human-readable names for the UI
ASPECT_DISPLAY_NAMES = {
    "genuineness": "Genuineness",
    "delivery": "Delivery Time",
    "quality": "Product Quality",
    "support": "Customer Support",
    "refund": "Refund / Return",
    "pricing": "Pricing",
}

# ---- Step 4: labeled data + trained model paths ------------------------
ASPECTS = list(ASPECT_KEYWORDS.keys())
LABELED_DIR = DATA_DIR / "labeled"
TRAIN_CLAUSES_CSV = LABELED_DIR / "train_clauses.csv"   # used to TRAIN the classifier
TEST_CLAUSES_CSV = LABELED_DIR / "test_clauses.csv"     # held out, never trained on
EVAL_COMMENTS_CSV = LABELED_DIR / "eval_comments.csv"   # 200 full comments for evaluation
ASPECT_MODEL_PATH = MODELS_DIR / "aspect_clf.joblib"

# ---- Step 6: aggregation settings ----------------------------------------
SCORE_SATURATION = 0.6      # a sentiment score of +/-0.6 already counts as 5 / 1 star
PRIOR_MEAN = 3.0            # a site with no evidence is "average"
PRIOR_STRENGTH = 5.0        # = pseudo-reviews of prior; higher -> more cautious
# How much each aspect matters for the overall TRUST score (sums to 1.0).
ASPECT_IMPORTANCE = {
    "genuineness": 0.30, "delivery": 0.20, "quality": 0.20,
    "support": 0.10, "refund": 0.10, "pricing": 0.10,
}
# Genuineness veto: a site that sells fakes can't be "trusted" however cheap it is.
GENUINENESS_VETO_STARS = 2.0
GENUINENESS_VETO_MIN_CONFIDENCE = 0.5
TRUST_CAP_ON_VETO = 40.0

# ---- Step 7: spam filter ----------------------------------------------------
SPAM_TRAIN_CSV = LABELED_DIR / "spam_train.csv"
SPAM_TEST_CSV = LABELED_DIR / "spam_test.csv"
SPAM_HARD_CSV = LABELED_DIR / "spam_hard.csv"     # hand-written tricky cases
SPAM_MODEL_PATH = MODELS_DIR / "spam_clf.joblib"

# ---- Step 10: evaluation ----------------------------------------------------
REPORTS_DIR = ROOT_DIR / "reports"
