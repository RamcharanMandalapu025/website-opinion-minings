"""Drive the real Streamlit script headlessly with streamlit.testing.AppTest."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from opinion_mining.config import ASPECT_MODEL_PATH, SPAM_MODEL_PATH
from opinion_mining.database import Database

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
needs_models = pytest.mark.skipif(not (ASPECT_MODEL_PATH.exists() and SPAM_MODEL_PATH.exists()),
                                  reason="train models first (Steps 4 and 7)")


def run_app(tmp_path, monkeypatch, sites=("TestShop",)):
    path = tmp_path / "app.db"
    db = Database(path)
    for s in sites:
        db.add_website(s)
    monkeypatch.setenv("OPINION_DB", str(path))
    return AppTest.from_file(APP, default_timeout=60).run(), db


def texts(elements):
    return " ".join(str(e.value) for e in elements)


def test_empty_database_shows_onboarding(tmp_path, monkeypatch):
    monkeypatch.setenv("OPINION_DB", str(tmp_path / "empty.db"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "No websites yet" in texts(at.info)


@needs_models
def test_page_renders_without_comments(tmp_path, monkeypatch):
    at, _ = run_app(tmp_path, monkeypatch)
    assert not at.exception
    assert at.title[0].value.startswith("Website Evaluation")
    assert "No aspect ratings yet" in texts(at.info)
    assert [t.label for t in at.tabs] == ["📊 Ratings", "✍️ Add a comment", "🗂 Recent comments"]


@needs_models
def test_submit_comment_updates_ratings(tmp_path, monkeypatch):
    at, db = run_app(tmp_path, monkeypatch)
    at.text_area(key="comment_text").set_value(
        "Product was original but delivery took 12 days and support never replied.")
    at.button(key="submit_btn").click().run()
    assert not at.exception
    assert "Your comment was added" in texts(at.success)
    body = texts(at.markdown)
    assert "Genuineness" in body and "Delivery Time" in body and "Customer Support" in body
    assert len(db.fetch_records(1)) == 3                    # really stored
    assert at.session_state["comment_text"] == ""           # box cleared after success
    # the Ratings tab was refreshed in the SAME run (callbacks run before rendering)
    assert "No aspect ratings yet" not in texts(at.info)


@needs_models
def test_spam_is_blocked_and_explained(tmp_path, monkeypatch):
    at, db = run_app(tmp_path, monkeypatch)
    at.button(key="ex_Spam").click().run()                  # fills the box with a spam example
    at.button(key="submit_btn").click().run()
    assert "held back" in texts(at.error)
    assert db.fetch_records(1) == []                        # no effect on ratings
    assert at.session_state["comment_text"] != ""           # kept so the user can edit it


@needs_models
def test_validation_error_is_shown(tmp_path, monkeypatch):
    at, _ = run_app(tmp_path, monkeypatch)
    at.text_area(key="comment_text").set_value("ab")
    at.button(key="submit_btn").click().run()
    assert "too short" in texts(at.error)


@needs_models
def test_add_website_from_sidebar(tmp_path, monkeypatch):
    at, db = run_app(tmp_path, monkeypatch)
    at.text_input(key="new_site_name").set_value("Brand New Store")
    at.button(key="add_site_btn").click().run()
    assert db.find_website("Brand New Store") is not None
    assert at.session_state["site_id"] == db.find_website("Brand New Store")["id"]
    at.text_input(key="new_site_name").set_value("brand new store")    # duplicate (case-insensitive)
    at.button(key="add_site_btn").click().run()
    assert "already exists" in texts(at.error)


@needs_models
def test_switching_websites_isolates_data(tmp_path, monkeypatch):
    at, db = run_app(tmp_path, monkeypatch, sites=("ShopA", "ShopB"))
    at.text_area(key="comment_text").set_value("Delivery was super fast and the price is great")
    at.button(key="submit_btn").click().run()
    assert len(db.fetch_records(1)) == 2 and db.fetch_records(2) == []
    at.selectbox(key="site_id").set_value(2).run()
    assert "No aspect ratings yet" in texts(at.info)        # ShopB is still empty


@needs_models
def test_transformer_option_hidden_when_not_installed(tmp_path, monkeypatch):
    import importlib.util
    at, _ = run_app(tmp_path, monkeypatch)
    expected = ["VADER", "Transformer"] if importlib.util.find_spec("transformers") else ["VADER"]
    assert list(at.radio(key="engine").options) == expected
