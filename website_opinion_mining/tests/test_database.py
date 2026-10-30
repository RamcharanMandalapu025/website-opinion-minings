import sqlite3
from types import SimpleNamespace

import pytest

from opinion_mining import spam_dataset, spam_filter
from opinion_mining.database import Database
from opinion_mining.pipeline import AspectSentiment, CommentAnalysis, PipelineV1
from opinion_mining.service import OpinionService
from opinion_mining.spam_filter import SpamFilter


def analysis(text, **aspects):
    """Build a CommentAnalysis by hand: analysis('x', delivery=-0.7, pricing=0.6)."""
    lab = lambda s: "Positive" if s >= 0.15 else "Negative" if s <= -0.15 else "Neutral"
    return CommentAnalysis(text, [AspectSentiment(a, s, lab(s), [text]) for a, s in aspects.items()])


def verdict(label="clean", weight=1.0, prob=0.0, reasons=()):
    return SimpleNamespace(label=label, weight=weight, prob=prob, reasons=list(reasons))


@pytest.fixture
def db():
    return Database(":memory:")


@pytest.fixture(scope="module")
def spam():
    spam_dataset.build()
    return SpamFilter(bundle=spam_filter.train(save=False, verbose=False))


@pytest.fixture
def svc(db, spam):
    return OpinionService(db, PipelineV1(), spam, pipeline_version="test")


# ------------------------------------------------------------- schema ------
def test_schema_is_idempotent(db):
    db.init_schema(); db.init_schema()
    assert db.list_websites() == []


def test_unique_website_names_case_insensitive(db):
    db.add_website("GoodMart")
    with pytest.raises(ValueError, match="already exists"):
        db.add_website("goodmart")


@pytest.mark.parametrize("bad", ["", "   ", "x" * 101])
def test_bad_website_names(db, bad):
    with pytest.raises(ValueError):
        db.add_website(bad)


def test_foreign_key_enforced(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_comment(999, "orphan comment", None, None)


def test_check_constraints_reject_impossible_values(db):
    wid = db.add_website("A")
    bad = analysis("x", delivery=0.5)
    bad.aspects[0].aspect = "shipping_speed"                    # not one of the six
    with pytest.raises(sqlite3.IntegrityError):
        db.add_comment(wid, "valid text here", bad)
    bad2 = analysis("x", delivery=0.5)
    bad2.aspects[0].score = 7.0                                 # outside [-1, 1]
    with pytest.raises(sqlite3.IntegrityError):
        db.add_comment(wid, "valid text here", bad2)


def test_failed_insert_rolls_back_everything(db):
    wid = db.add_website("A")
    bad = analysis("x", delivery=0.5, pricing=0.5)
    bad.aspects[1].aspect = "nonsense"                          # 2nd aspect row fails
    with pytest.raises(sqlite3.IntegrityError):
        db.add_comment(wid, "a valid comment", bad)
    assert db.comment_texts(wid) == []                          # comment row rolled back too


def test_cascade_delete(db):
    wid = db.add_website("A")
    db.add_comment(wid, "delivery was late", analysis("delivery was late", delivery=-0.7))
    db.delete_website(wid)
    with db._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM comments").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM comment_aspects").fetchone()[0] == 0


def test_sql_injection_text_is_stored_literally(db):
    wid = db.add_website("A")
    evil = "'); DROP TABLE comments; --"
    db.add_comment(wid, evil, None)
    assert db.comment_texts(wid) == [evil]


def test_data_persists_in_file_database(tmp_path):
    path = tmp_path / "t.db"
    d1 = Database(path); wid = d1.add_website("Persist")
    d1.add_comment(wid, "great price here", analysis("great price here", pricing=0.7))
    d2 = Database(path)                                         # new object, same file
    assert d2.find_website("Persist")["id"] == wid
    assert len(d2.fetch_records(wid)) == 1


# ------------------------------------------------------ queries / ratings --
def test_records_carry_spam_weight_into_ratings(db):
    wid = db.add_website("A")
    db.add_comment(wid, "delivery was slow", analysis("delivery was slow", delivery=-0.8), verdict())
    db.add_comment(wid, "spam about price", analysis("spam about price", pricing=0.9),
                   verdict("spam", 0.0, 0.99, ["contains a link"]))
    r = db.website_rating(wid)
    assert r.aspects["delivery"].stars is not None
    assert r.aspects["pricing"].stars is None                   # weight 0 -> ignored


def test_top_comments_ordering_and_spam_exclusion(db):
    wid = db.add_website("A")
    db.add_comment(wid, "ok price", analysis("ok price", pricing=0.3))
    db.add_comment(wid, "superb price", analysis("superb price", pricing=0.9))
    db.add_comment(wid, "awful delivery", analysis("awful delivery", delivery=-0.9))
    db.add_comment(wid, "spam gush", analysis("spam gush", pricing=1.0), verdict("spam", 0.0, 0.9))
    pos, neg = db.top_comments(wid, n=2)
    assert [p["text"] for p in pos] == ["superb price", "ok price"]   # spam (1.0) excluded
    assert [n["text"] for n in neg] == ["awful delivery"]
    assert pos[0]["aspects"] == {"pricing": "Positive"}


def test_list_websites_counts(db):
    wid = db.add_website("A")
    db.add_comment(wid, "good comment", None, verdict())
    db.add_comment(wid, "spam comment", None, verdict("spam", 0.0, 0.9))
    w = db.list_websites()[0]
    assert (w["n_comments"], w["n_spam"]) == (1, 1)


def test_recent_comments_hides_spam_by_default(db):
    wid = db.add_website("A")
    db.add_comment(wid, "good comment", None, verdict())
    db.add_comment(wid, "spam comment", None, verdict("spam", 0.0, 0.9, ["contains a link"]))
    assert [c["text"] for c in db.recent_comments(wid)] == ["good comment"]
    assert len(db.recent_comments(wid, include_spam=True)) == 2


# ----------------------------------------------------------- service -------
def test_submit_normal_comment(svc, db):
    wid = db.add_website("A")
    r = svc.submit_comment(wid, "Product was original but delivery took 12 days and support never replied.")
    assert r.status == "accepted" and r.accepted
    assert r.analysis.as_dict() == {"genuineness": "Positive", "delivery": "Negative", "support": "Negative"}
    assert len(db.fetch_records(wid)) == 3


def test_submit_spam_is_stored_but_has_no_effect(svc, db):
    wid = db.add_website("A")
    r = svc.submit_comment(wid, "Visit www.dealz4u.com for 70% off on all price items!!!")
    assert r.status == "blocked_spam" and "held back" in r.message
    assert db.fetch_records(wid) == []
    assert svc.get_report(wid)["n_spam_blocked"] == 1


def test_duplicate_submission_is_blocked(svc, db):
    wid = db.add_website("A")
    text = "The seller delivered a genuine product within two days"
    assert svc.submit_comment(wid, text).accepted
    assert svc.submit_comment(wid, text + ".").status == "blocked_spam"


def test_comment_without_aspects(svc, db):
    wid = db.add_website("A")
    r = svc.submit_comment(wid, "I ordered a phone last week")
    assert r.accepted and "won't change the ratings" in r.message


@pytest.mark.parametrize("bad", ["", "  ", "ab", "12345!!!", "x" * 2001])
def test_validation_errors(svc, db, bad):
    wid = db.add_website("A")
    with pytest.raises(ValueError):
        svc.submit_comment(wid, bad)


def test_unknown_website(svc):
    with pytest.raises(ValueError, match="Unknown website"):
        svc.submit_comment(12345, "delivery was fast")


def test_full_report(svc, db):
    wid = db.add_website("A")
    for t in ["Fast delivery and great price", "Fake product and no refund", "Excellent quality"]:
        svc.submit_comment(wid, t)
    rep = svc.get_report(wid)
    assert rep["n_comments"] == 3 and rep["rating"].trust_score is not None
    assert rep["top_positive"] and rep["top_negative"]
    assert rep["rating"].aspects["delivery"].stars > 3 and rep["rating"].aspects["genuineness"].stars < 3
