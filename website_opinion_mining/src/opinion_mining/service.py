"""
Step 8b: the SERVICE layer = the one place that wires the whole system together.

    raw comment -> validate -> spam filter -> (if not spam) pipeline -> database

The Streamlit app (Step 9) only calls this class; it never touches models or SQL.
That separation is what lets us swap V1/V2 pipelines, or replace Streamlit with an
API, without rewriting anything else.
"""
import re
from dataclasses import dataclass
from typing import Optional

from .database import Database

MIN_LEN, MAX_LEN = 3, 2000


@dataclass
class SubmissionResult:
    comment_id: int
    status: str            # "accepted" | "accepted_low_weight" | "blocked_spam"
    message: str           # human-readable text for the UI
    verdict: object        # SpamVerdict
    analysis: Optional[object] = None   # CommentAnalysis (None if blocked)

    @property
    def accepted(self) -> bool:
        return self.status != "blocked_spam"


class OpinionService:
    def __init__(self, db: Database, pipeline, spam_filter, pipeline_version: str = "v2"):
        self.db, self.pipeline = db, pipeline
        self.spam_filter, self.pipeline_version = spam_filter, pipeline_version

    @staticmethod
    def validate(text: str) -> str:
        text = (text or "").strip()
        if len(text) < MIN_LEN:
            raise ValueError("Comment is too short - please write at least a few words.")
        if len(text) > MAX_LEN:
            raise ValueError(f"Comment is too long (max {MAX_LEN} characters).")
        if not re.search(r"[A-Za-z]", text):
            raise ValueError("Comment must contain some text.")
        return text

    def submit_comment(self, website_id: int, text: str,
                       check_duplicates: bool = True) -> SubmissionResult:
        if self.db.get_website(website_id) is None:
            raise ValueError("Unknown website.")
        text = self.validate(text)
        existing = self.db.comment_texts(website_id) if check_duplicates else None
        verdict = self.spam_filter.assess(text, site_comments=existing)

        if verdict.is_spam:       # stored for audit, but weight 0 -> never affects ratings
            cid = self.db.add_comment(website_id, text, None, verdict, self.pipeline_version)
            why = "; ".join(verdict.reasons) or "it looks like spam"
            return SubmissionResult(cid, "blocked_spam",
                                    f"Your comment was held back ({why}).", verdict)

        analysis = self.pipeline.analyze(text)
        cid = self.db.add_comment(website_id, text, analysis, verdict, self.pipeline_version)
        if not analysis.aspects:
            msg = ("Thanks! We couldn't find any of the rated topics (delivery, quality, "
                   "support, refund, pricing, genuineness) in your comment, so it won't "
                   "change the ratings.")
        elif verdict.label == "suspicious":
            msg = "Thanks! Your comment was saved but counts with reduced weight."
        else:
            msg = "Thanks! Your comment was added."
        status = "accepted_low_weight" if verdict.label == "suspicious" else "accepted"
        return SubmissionResult(cid, status, msg, verdict, analysis)

    def get_report(self, website_id: int, top_n: int = 3) -> dict:
        site = self.db.get_website(website_id)
        if site is None:
            raise ValueError("Unknown website.")
        pos, neg = self.db.top_comments(website_id, top_n)
        summary = next(w for w in self.db.list_websites() if w["id"] == website_id)
        return {"website": site, "rating": self.db.website_rating(website_id),
                "top_positive": pos, "top_negative": neg,
                "n_comments": summary["n_comments"], "n_spam_blocked": summary["n_spam"]}
