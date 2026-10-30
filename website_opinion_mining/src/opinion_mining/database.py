"""
Step 8a: SQLite data-access layer. All SQL lives here; nothing else touches it.

* Every query is PARAMETERISED (?, ?) -> user text can never inject SQL.
* A fresh connection per operation (file DBs): Streamlit runs scripts in several
  threads and sqlite3 connections must not be shared across threads.
  ":memory:" databases keep one connection (otherwise each call would see an empty DB).
* Each public method is one transaction: either everything is saved or nothing is.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .aggregation import AspectRecord, WebsiteRating, aggregate
from .config import DB_PATH

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path=DB_PATH):
        self.path = str(path)
        self._mem = None
        if self.path == ":memory:":
            self._mem = sqlite3.connect(":memory:")
            self._mem.row_factory = sqlite3.Row
            self._mem.execute("PRAGMA foreign_keys = ON")
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    # ------------------------------------------------------------ plumbing --
    @contextmanager
    def _conn(self):
        if self._mem is not None:
            conn, owns = self._mem, False
        else:
            conn, owns = sqlite3.connect(self.path, timeout=10), True
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")      # OFF by default in SQLite!
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if owns:
                conn.close()

    def init_schema(self):
        with self._conn() as c:
            c.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            if self._mem is None:
                c.execute("PRAGMA journal_mode = WAL")    # readers don't block the writer

    # ------------------------------------------------------------ websites --
    def add_website(self, name: str, url: Optional[str] = None) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("Website name cannot be empty.")
        if len(name) > 100:
            raise ValueError("Website name is too long (max 100 characters).")
        try:
            with self._conn() as c:
                cur = c.execute("INSERT INTO websites (name, url, created_at) VALUES (?, ?, ?)",
                                (name, (url or "").strip() or None, _now()))
                return cur.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"A website named '{name}' already exists.") from None

    def get_website(self, website_id: int) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM websites WHERE id = ?", (website_id,)).fetchone()
        return dict(row) if row else None

    def find_website(self, name: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM websites WHERE name = ?", (name.strip(),)).fetchone()
        return dict(row) if row else None

    def list_websites(self) -> List[dict]:
        """Websites with counts of accepted (weight>0) and blocked (spam) comments."""
        with self._conn() as c:
            rows = c.execute("""
                SELECT w.id, w.name, w.url, w.created_at,
                       COALESCE(SUM(c.spam_weight > 0), 0)        AS n_comments,
                       COALESCE(SUM(c.spam_label = 'spam'), 0)    AS n_spam
                FROM websites w LEFT JOIN comments c ON c.website_id = w.id
                GROUP BY w.id ORDER BY w.name COLLATE NOCASE""").fetchall()
        return [dict(r) for r in rows]

    def delete_website(self, website_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM websites WHERE id = ?", (website_id,))   # cascades

    # ------------------------------------------------------------ comments --
    def add_comment(self, website_id: int, text: str, analysis=None, verdict=None,
                    pipeline_version: Optional[str] = None,
                    created_at: Optional[str] = None) -> int:
        """Insert a comment + its aspect rows in ONE transaction."""
        aspects = list(analysis.aspects) if analysis is not None else []
        polarity = sum(a.score for a in aspects) / len(aspects) if aspects else None
        p, label, weight, reasons = (0.0, "clean", 1.0, []) if verdict is None else \
            (verdict.prob, verdict.label, verdict.weight, verdict.reasons)
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO comments (website_id, text, created_at, spam_prob, spam_label,
                       spam_weight, spam_reasons, polarity, pipeline_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (website_id, text, created_at or _now(), p, label, weight,
                 json.dumps(reasons), polarity, pipeline_version))
            cid = cur.lastrowid
            c.executemany(
                "INSERT INTO comment_aspects (comment_id, aspect, score, label, evidence) "
                "VALUES (?, ?, ?, ?, ?)",
                [(cid, a.aspect, a.score, a.label, json.dumps(a.evidence)) for a in aspects])
        return cid

    def comment_texts(self, website_id: int, limit: int = 500) -> List[str]:
        """Most recent comment texts (used for near-duplicate detection)."""
        with self._conn() as c:
            rows = c.execute("SELECT text FROM comments WHERE website_id = ? "
                             "ORDER BY id DESC LIMIT ?", (website_id, limit)).fetchall()
        return [r["text"] for r in rows]

    def _attach_aspects(self, conn, comments: List[dict]) -> List[dict]:
        for cm in comments:
            rows = conn.execute("SELECT aspect, label, score FROM comment_aspects "
                                "WHERE comment_id = ? ORDER BY id", (cm["id"],)).fetchall()
            cm["aspects"] = {r["aspect"]: r["label"] for r in rows}
            cm["spam_reasons"] = json.loads(cm["spam_reasons"]) if "spam_reasons" in cm else []
        return comments

    def recent_comments(self, website_id: int, limit: int = 20,
                        include_spam: bool = False) -> List[dict]:
        sql = ("SELECT id, text, created_at, polarity, spam_label, spam_prob, spam_reasons "
               "FROM comments WHERE website_id = ? "
               + ("" if include_spam else "AND spam_label != 'spam' ")
               + "ORDER BY id DESC LIMIT ?")
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(sql, (website_id, limit)).fetchall()]
            return self._attach_aspects(c, rows)

    def top_comments(self, website_id: int, n: int = 3) -> Tuple[List[dict], List[dict]]:
        """(most positive, most negative) non-spam comments. Ties -> more aspects first."""
        base = """SELECT c.id, c.text, c.created_at, c.polarity, c.spam_reasons,
                         COUNT(a.id) AS n_aspects
                  FROM comments c JOIN comment_aspects a ON a.comment_id = c.id
                  WHERE c.website_id = ? AND c.spam_weight > 0 AND c.polarity {cmp} 0
                  GROUP BY c.id ORDER BY c.polarity {order}, n_aspects DESC LIMIT ?"""
        with self._conn() as c:
            pos = [dict(r) for r in c.execute(base.format(cmp=">", order="DESC"), (website_id, n))]
            neg = [dict(r) for r in c.execute(base.format(cmp="<", order="ASC"), (website_id, n))]
            return self._attach_aspects(c, pos), self._attach_aspects(c, neg)

    # --------------------------------------------------------- aggregation --
    def fetch_records(self, website_id: int) -> List[AspectRecord]:
        """Aspect rows -> AspectRecord objects (weight = the comment's spam weight)."""
        with self._conn() as c:
            rows = c.execute("""SELECT a.aspect, a.score, a.label, c.spam_weight, c.id
                                FROM comment_aspects a JOIN comments c ON c.id = a.comment_id
                                WHERE c.website_id = ?""", (website_id,)).fetchall()
        return [AspectRecord(r["aspect"], r["score"], r["label"], r["spam_weight"], r["id"])
                for r in rows]

    def website_rating(self, website_id: int) -> WebsiteRating:
        return aggregate(self.fetch_records(website_id))
