"""
Fill db/opinions.db with demo data so the Streamlit app (Step 9) has something to show.

    python -m opinion_mining.seed_demo           # create if missing
    python -m opinion_mining.seed_demo --reset   # delete and recreate

Uses the REAL pipeline (trained aspect model + VADER + sarcasm layer + spam filter),
so it also doubles as an end-to-end test of Steps 1-8.
Needs models from Step 4 and 7 (run those modules first if models/ is empty).
"""
import json
import sys

import pandas as pd

from .config import DB_PATH, EVAL_COMMENTS_CSV
from .database import Database
from .pipeline import build_pipeline
from .service import OpinionService
from .spam_filter import SpamFilter

DEMO_SPAM = [
    "Visit www.dealz4u.com for 70% off on all items!!!",
    "Best best best site ever amazing amazing 10/10 A+++",
    "asdkjh qwezxc lkjhgf",
    "Click here bit.ly/3xYz9a to win a free iPhone",
]


def seed(db_path=DB_PATH, reset: bool = False) -> None:
    if reset and DB_PATH.exists() and str(db_path) == str(DB_PATH):
        for suffix in ("", "-wal", "-shm"):
            p = type(DB_PATH)(str(DB_PATH) + suffix)
            if p.exists():
                p.unlink()
    db = Database(db_path)
    if db.list_websites():
        print("Database already has data (use --reset to recreate).")
        return
    svc = OpinionService(db, build_pipeline("v2", transformer=False), SpamFilter(),
                         pipeline_version="v2-vader")

    ev = pd.read_csv(EVAL_COMMENTS_CSV)
    good, bad = [], []
    for text, truth in zip(ev["comment"], ev["aspects"]):
        vals = list(json.loads(truth).values())
        (good if vals.count("Positive") > vals.count("Negative") else bad).append(text)

    plan = {"GoodMart": (good, "https://goodmart.example"),
            "BadBazaar": (bad, "https://badbazaar.example"),
            "NewShop": (good[:2], "https://newshop.example")}
    for name, (comments, url) in plan.items():
        wid = db.add_website(name, url)
        # Demo comments are synthetic and repeat clauses BY CONSTRUCTION, so the
        # duplicate check would wrongly flag them; real submissions keep it on.
        for text in comments:
            svc.submit_comment(wid, text, check_duplicates=False)
        if name != "NewShop":
            for text in DEMO_SPAM:
                svc.submit_comment(wid, text, check_duplicates=False)
    for w in db.list_websites():
        r = db.website_rating(w["id"])
        print(f"{w['name']:<10} accepted={w['n_comments']:<4} spam_blocked={w['n_spam']:<3} "
              f"trust={r.trust_score}")


if __name__ == "__main__":
    seed(reset="--reset" in sys.argv)
