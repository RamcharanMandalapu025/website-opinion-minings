"""
Step 7a: labeled spam / genuine data (label 1 = spam, 0 = genuine).

* GENUINE ("ham") comments: assembled from the Step-4 clause bank (so they read
  like real reviews: specific, mixed sentiment). Train ham uses TRAIN clauses,
  test ham uses held-out TEST clauses.
* SPAM comments come in 4 kinds, built from templates with random slots:
    promo       links / phone numbers / "earn money" / discount codes
    gush        generic fake praise ("best best best A+++ 10/10")
    competitor  paid bashing ("worst site, go to X instead")
    gibberish   keyboard mashing
  Each kind has a TRAIN template pool and a different TEST template pool.
* Realism tweaks: 25% of promo spam hides behind a genuine-sounding first
  sentence; 5% of ALL comments are SHOUTED; some genuine comments end in "!!".
  So the model can't just learn "caps = spam" or "!! = spam".
* A separate HARD set is hand-written (see HARD_CASES).

CAVEAT: synthetic -> scores on spam_test.csv are optimistic. The hard set and,
above all, real labeled spam (e.g. a public fake-review dataset) are the honest tests.
"""
import random
import string

import pandas as pd

from .config import (LABELED_DIR, SPAM_HARD_CSV, SPAM_TEST_CSV, SPAM_TRAIN_CSV)
from .dataset_builder import _lower_first, split_bank

URLS = ["www.dealz4u.com", "bit.ly/3xYz9a", "shop-now.xyz", "http://cheapstuff.biz/win",
        "buyfast.in", "www.megasale-online.net", "tinyurl.com/k9d2x"]

PROMO = {
    "train": [
        "Visit {url} for {pct}% off on all items!!! Limited offer",
        "Click here {url} to win a free iPhone now",
        "WhatsApp {phone} for wholesale deals buy now",
        "Earn money from home with no investment {url}",
        "Best deals only on {url} contact {phone}",
        "Cheap replica watches available at {url}",
        "Get {pct}% cashback use code SAVE{pct} at {url}",
        "Make $500 daily online guaranteed check {url}",
    ],
    "test": [
        "Hot offer!!! Visit {url} and save {pct}% today only",
        "Contact {phone} for cheap bulk orders",
        "Win free vouchers just click {url}",
        "Work from home job apply now {url} earn daily",
        "Follow us on {url} for daily giveaways",
    ],
}
GUSH = {
    "train": [
        "Best website ever!!! Highly recommended 5 stars",
        "Amazing amazing amazing love it so much best ever",
        "A+++ seller fast shipping great product will buy again 10/10",
        "Superb excellent wonderful fantastic service everyone must buy",
        "Awesome awesome awesome five star five star",
        "Perfect perfect perfect best best best",
        "This is the best site in the world buy now you will not regret",
        "Great great great product great seller great website",
    ],
    "test": [
        "Wow wow wow just wow best purchase of my life!!!",
        "10/10 would recommend to everyone A+++++",
        "Incredible amazing fantastic outstanding must buy",
        "Best best best site never disappointed 5 stars 5 stars",
        "Everything is perfect perfect perfect",
    ],
}
COMPETITOR = {
    "train": [
        "Worst site ever do not buy anything here go to {url} instead",
        "Scam scam scam stay away use {url} only",
        "Total fraud avoid avoid avoid shop at {url}",
        "Never ever buy from here terrible terrible terrible 1 star",
        "This site is garbage better to order from {url}",
        "Horrible horrible horrible worst worst worst",
        "Do not trust them all fake all lies try {url}",
        "Zero stars complete waste go elsewhere",
    ],
    "test": [
        "Worst worst worst never again visit {url} instead",
        "Fraud fraud fraud do not order anything",
        "Pathetic site buy from {url} much better",
        "Stay away stay away stay away scammers",
        "Terrible terrible site all fake go to {url}",
    ],
}
KEYMASH = ["asdfgh", "qwertyui", "zxcvbn", "hhhhhhh", "lololol", "jkjkjk", "aaaaaaa"]

# Hand-written tricky cases: (text, label). NOT used for training.
HARD_CASES = [
    # --- genuine but "spammy-looking"
    ("Excellent quality!!! Fast delivery!!! Totally worth it!!!", 0),
    ("WORST SUPPORT EVER. NEVER REPLIED TO MY 3 EMAILS.", 0),
    ("Great product. Used it for 2 months and the battery still lasts all day.", 0),
    ("Delivery took 9 days but the item is genuine and well packed.", 0),
    ("Good value for money", 0),
    ("Refund took 3 weeks but they finally paid back to my card ending 4421", 0),
    ("Cheap price but the strap broke in a week, disappointed.", 0),
    ("Bought 2 shirts for 499 each, colour faded after one wash, return was easy.", 0),
    ("Was skeptical about the site but item is 100% original, checked the serial number.", 0),
    # --- spam but "genuine-looking"
    ("Great product. Visit bit.ly/2abc for more deals", 1),
    ("Nice nice nice nice nice nice", 1),
    ("Best prices guaranteed contact 9876543210 for bulk", 1),
    ("good good good good good", 1),
    ("Amazing store!! Everyone should buy from here!! Five stars!!", 1),
    ("I made $2000 last week working from home, DM me", 1),
    ("Awesome product fast delivery great seller A+", 1),
    ("qwertyuiop asdfghjkl", 1),
]


def _fill(tpl: str, rng: random.Random) -> str:
    return tpl.format(url=rng.choice(URLS),
                      phone="+91 " + "".join(rng.choices("0123456789", k=10)),
                      pct=rng.choice([30, 40, 50, 60, 70, 80, 90]))


def _gibberish(rng: random.Random) -> str:
    toks = []
    for _ in range(rng.randint(2, 6)):
        if rng.random() < 0.3:
            toks.append(rng.choice(KEYMASH))
        else:
            toks.append("".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))))
    return " ".join(toks)


def _join_ham(parts, rng: random.Random) -> str:
    text = parts[0]
    for p in parts[1:]:
        sep = rng.choice([". ", ", and ", " and ", ", but ", ". "])
        text += sep + (p if sep == ". " else _lower_first(p))
    text += rng.choices([".", "!", "!!", ""], weights=[0.7, 0.15, 0.05, 0.1])[0]
    return text.upper() if rng.random() < 0.05 else text


def make_split(pool: str, clauses, n_ham: int, n_spam: int, seed: int) -> pd.DataFrame:
    """pool = 'train' or 'test' selects which template pool the spam comes from."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_ham):
        k = rng.choices([1, 2, 3], weights=[0.35, 0.4, 0.25])[0]
        rows.append((_join_ham(rng.sample(clauses, k), rng), 0, "ham"))
    kinds = rng.choices(["promo", "gush", "competitor", "gibberish"],
                        weights=[0.35, 0.25, 0.2, 0.2], k=n_spam)
    pools = {"promo": PROMO[pool], "gush": GUSH[pool], "competitor": COMPETITOR[pool]}
    for kind in kinds:
        text = _gibberish(rng) if kind == "gibberish" else _fill(rng.choice(pools[kind]), rng)
        if kind == "promo" and rng.random() < 0.25:          # hide behind a normal sentence
            text = rng.choice(clauses) + ". " + text
        if rng.random() < 0.25:
            text = text.upper()
        rows.append((text, 1, kind))
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label", "kind"])


def build():
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    train, test = split_bank()
    train_clauses = [c for c, _, _ in train]
    test_clauses = [c for c, _, _ in test]
    make_split("train", train_clauses, 500, 150, seed=11).to_csv(SPAM_TRAIN_CSV, index=False)
    make_split("test", test_clauses, 250, 60, seed=12).to_csv(SPAM_TEST_CSV, index=False)
    pd.DataFrame(HARD_CASES, columns=["text", "label"]).to_csv(SPAM_HARD_CSV, index=False)
    print("spam_train: 500 genuine + 150 spam | spam_test: 250 + 60 | spam_hard:", len(HARD_CASES))


if __name__ == "__main__":
    build()
