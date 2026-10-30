"""
Step 4a: build a labeled starter dataset.

HOW IT WORKS (explain this honestly in your viva):
  * BANK = hand-written clauses, each labeled (aspect, sentiment).
    14 per (aspect x sentiment) cell; ~half contain obvious keywords, the rest
    are paraphrases ("Nobody got back to me") so keyword matching can fail.
  * The bank is split BEFORE anything else: 9 clauses/cell -> TRAIN, 5 -> TEST.
    No clause appears in both, so the test set is truly unseen.
  * The 200 evaluation comments are assembled ONLY from TEST clauses by joining
    1-3 clauses with connectors ("but", "and", "."), so labels are exact.

CAVEAT: this is a synthetic starter set. Replace/extend it with real comments
(see "I have data" workflow) before reporting final numbers.
Clauses deliberately avoid commas / "and" / "but" so they stay single clauses.
"""
import json
import random

import pandas as pd

from .config import (ASPECTS, EVAL_COMMENTS_CSV, LABELED_DIR,
                     TEST_CLAUSES_CSV, TRAIN_CLAUSES_CSV)

BANK = {
    "genuineness": {
        "Positive": [
            "The product was 100% original", "Got a genuine item with the authenticity card inside",
            "Authentic item with a sealed box", "The brand verified it as legit",
            "It is exactly the real thing", "Serial number matched the brand website",
            "No doubt about the originality of this watch", "Received a genuine branded product",
            "The hologram checked out fine", "Seller sells only authentic goods",
            "Compared with the store model it is identical", "Feels like the real deal",
            "Not a single sign of a fake", "Totally trustworthy authentic stuff",
        ],
        "Negative": [
            "The item is a cheap counterfeit", "Definitely a fake product",
            "It was a duplicate sold as original", "The logo looks like a knockoff",
            "Serial number did not match the brand records", "Not original at all",
            "They sell first copy items as branded", "The brand confirmed it is not genuine",
            "Total scam with imitation goods", "The logo clearly shows it is a replica",
            "Packaging looked fake", "I doubt the authenticity of this item",
            "Received a forged warranty card", "Nothing about it feels real",
        ],
    },
    "delivery": {
        "Positive": [
            "Delivery was super fast", "Arrived two days before the promised date",
            "Reached me within 2 days", "The courier was prompt",
            "Package came well packed", "Got it the very next morning",
            "Shipping was quick", "Tracking updates were accurate",
            "Dispatched the same day", "Delivered right on time",
            "Fastest shipping I have seen", "Parcel showed up earlier than expected",
            "The order came in just 3 days", "Neat packaging with speedy shipping",
        ],
        "Negative": [
            "Delivery took 15 days", "The package arrived two weeks late",
            "Order kept getting delayed", "Shipping is painfully slow",
            "Still waiting for my parcel after a month", "The courier never showed up",
            "Tracking has not updated for ten days", "Package was lost in transit",
            "Dispatch took forever", "Arrived with a torn box",
            "Promised 3 day shipping that took 3 weeks", "Got it way later than promised",
            "The delivery guy was unreachable", "Waited 3 weeks for it to arrive",
        ],
    },
    "quality": {
        "Positive": [
            "The quality is excellent", "Build feels very sturdy",
            "Works perfectly since day one", "Material feels premium",
            "Really durable stuff", "No defects at all",
            "Stitching is neat", "Performance has been flawless",
            "Looks even better than the photos", "The finish is top notch",
            "Battery lasts all day", "Very well made product",
            "It has held up great for months", "Sound quality blew me away",
        ],
        "Negative": [
            "Quality is terrible", "It stopped working after a week",
            "The product was broken", "Feels flimsy",
            "Screen had a crack out of the box", "Faulty piece from day one",
            "Material is cheaply made", "The zipper tore on the first use",
            "Battery drains in two hours", "Colour faded after one wash",
            "Poor build quality", "Item was damaged",
            "The buttons fell off within days", "Performance is laggy",
        ],
    },
    "support": {
        "Positive": [
            "Customer support was very helpful", "They replied within an hour",
            "The agent solved my issue quickly", "Got a call back the same day",
            "Support team was polite", "Chat support resolved everything",
            "Staff went out of their way to help me", "Responsive team",
            "They answered all my questions patiently", "Customer care was excellent",
            "Service team followed up twice", "The helpline picked up immediately",
            "My complaint was handled professionally", "Great after sales service",
        ],
        "Negative": [
            "Support never replied", "Customer care is useless",
            "Nobody answered my emails", "The agent was rude",
            "Helpline keeps cutting my calls", "Raised a ticket with zero response",
            "No one bothered to get back to me", "Chat support just gave canned replies",
            "They ignored my complaint", "Service team is unresponsive",
            "Got bounced between agents", "Support is a joke",
            "Customer service hung up on me", "Messages left on read for days",
        ],
    },
    "refund": {
        "Positive": [
            "Refund was processed in 3 days", "Return was hassle free",
            "Got my money back quickly", "They replaced the item without questions",
            "Easy exchange process", "Return pickup was arranged next day",
            "The refund reached my account promptly", "Cancellation was instant",
            "Return policy is very fair", "Replacement came without any fuss",
            "Full refund with no arguments", "Exchange was smooth",
            "They honored the warranty", "Money was credited back right away",
        ],
        "Negative": [
            "Still no refund after two months", "They refuse to take returns",
            "Return policy is a trap", "Cancelled my order but never refunded",
            "Replacement request was rejected", "Refund keeps getting delayed",
            "Money back requests get excuses", "No exchange allowed for any reason",
            "They deducted charges from my refund", "Return window is too short",
            "Refund process is a nightmare", "Warranty claim was denied",
            "Return pickup never happened", "They keep saying refund is in process",
        ],
    },
    "pricing": {
        "Positive": [
            "Great price for what you get", "Much cheaper than other sites",
            "Amazing discounts during the sale", "Totally worth the money",
            "Very affordable", "Best deal I found online",
            "Prices are unbeatable", "Got it at half the market rate",
            "Fair pricing", "Coupons made it even cheaper",
            "Excellent value for money", "Offers here beat every competitor",
            "The bargain of the year", "Cost me way less than the store",
        ],
        "Negative": [
            "Way too expensive", "Overpriced for the quality",
            "Cheaper in the local shop", "Hidden charges at checkout",
            "Prices were raised right before the sale", "The discount is based on an inflated MRP",
            "Not worth the price", "Extra fees made it costly",
            "Pricing is all over the place", "Too pricey",
            "Costs more than the competition", "Not a bargain at all",
            "I paid double the actual price", "Zero value for money",
        ],
    },
}

# Neutral, no-aspect clauses: the classifier must learn to output NOTHING for these.
FILLER = [
    "I ordered a phone last week", "Bought this for my brother", "Placed the order on Sunday",
    "I have been shopping here for a year", "This was my first purchase",
    "I ordered it for my daughter", "Used this site for a gift", "Ordered a pair of shoes",
    "Got a laptop from here", "Needed a charger urgently", "Bought two of them",
    "I read the reviews before ordering", "Ordered during the festival",
    "I use the mobile app",
]

N_TRAIN_PER_CELL, N_TRAIN_FILLER = 9, 10


def split_bank(seed: int = 42):
    """Return (train, test) lists of (clause, aspect, sentiment)."""
    rng = random.Random(seed)
    train, test = [], []
    for aspect, cells in BANK.items():
        for sentiment, clauses in cells.items():
            items = clauses[:]
            rng.shuffle(items)
            train += [(c, aspect, sentiment) for c in items[:N_TRAIN_PER_CELL]]
            test += [(c, aspect, sentiment) for c in items[N_TRAIN_PER_CELL:]]
    f = FILLER[:]
    rng.shuffle(f)
    train += [(c, "none", "Neutral") for c in f[:N_TRAIN_FILLER]]
    test += [(c, "none", "Neutral") for c in f[N_TRAIN_FILLER:]]
    return train, test


def _lower_first(c: str) -> str:
    return c if c.startswith(("I ", "I'")) else c[0].lower() + c[1:]


def make_eval_comments(test, n: int = 200, seed: int = 7):
    """Assemble n labeled comments from held-out TEST clauses only."""
    rng = random.Random(seed)
    by_aspect = {a: [(c, s) for c, a2, s in test if a2 == a] for a in ASPECTS}
    fillers = [c for c, a, _ in test if a == "none"]
    rows = []
    for i in range(n):
        k = rng.choices([1, 2, 3], weights=[0.3, 0.45, 0.25])[0]
        parts = [(*rng.choice(by_aspect[a]), a) for a in rng.sample(ASPECTS, k)]   # (clause, sent, aspect)
        if rng.random() < 0.2:                                                     # optional neutral opener
            parts.insert(0, (rng.choice(fillers), None, None))
        text, labels = parts[0][0], {}
        if parts[0][2]:
            labels[parts[0][2]] = parts[0][1]
        for prev, cur in zip(parts, parts[1:]):
            if prev[1] is None:
                text += ". " + cur[0]                         # after the opener: new sentence
            else:
                if prev[1] != cur[1]:
                    sep = ", but "                            # opposite sentiment -> contrast
                else:
                    sep = rng.choice([" and ", ", and ", ". "])
                text += sep + (cur[0] if sep == ". " else _lower_first(cur[0]))
            labels[cur[2]] = cur[1]
        rows.append({"id": i + 1, "comment": text + ".", "aspects": json.dumps(labels)})
    return rows


def build():
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    train, test = split_bank()
    cols = ["clause", "aspect", "sentiment"]
    pd.DataFrame(train, columns=cols).to_csv(TRAIN_CLAUSES_CSV, index=False)
    pd.DataFrame(test, columns=cols).to_csv(TEST_CLAUSES_CSV, index=False)
    pd.DataFrame(make_eval_comments(test)).to_csv(EVAL_COMMENTS_CSV, index=False)
    print(f"train clauses: {len(train)} | test clauses: {len(test)} | eval comments: 200")


if __name__ == "__main__":
    build()
