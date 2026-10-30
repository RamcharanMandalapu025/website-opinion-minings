"""
Step 9b: Streamlit web app.            Run:  streamlit run app/streamlit_app.py

The app contains NO machine-learning or SQL code: it only calls
OpinionService (service.py) and Database (database.py) and draws the results.
"""
import importlib.util
import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))          # so `streamlit run` works without PYTHONPATH

import streamlit as st  # noqa: E402

from opinion_mining.aggregation import render_stars  # noqa: E402
from opinion_mining.config import ASPECT_DISPLAY_NAMES, ASPECTS, DB_PATH  # noqa: E402
from opinion_mining.database import Database  # noqa: E402
from opinion_mining.ui_helpers import (CONF_EMOJI, LABEL_EMOJI, chips, sentiment_mix_figure,  # noqa: E402
                                       stars_figure, trust_gauge, trust_label)

st.set_page_config(page_title="Website Opinion Mining", page_icon="🛍️", layout="wide")
DB_FILE = os.environ.get("OPINION_DB", str(DB_PATH))
TF_AVAILABLE = importlib.util.find_spec("transformers") is not None
EXAMPLES = {
    "Mixed": "Product was original but delivery took 12 days and support never replied.",
    "Sarcasm": "Oh great, another late delivery. Love how support never replies 🙄",
    "Negation": "The price is not bad, but the refund process was a nightmare.",
    "Spam": "Visit www.dealz4u.com for 70% off on all items!!! Click here now",
}


# ------------------------------------------------------------------ resources --
@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path)


@st.cache_resource(show_spinner=False)
def get_service(engine: str, path: str):
    """Heavy objects (models) are built ONCE per engine and reused across reruns."""
    from opinion_mining.pipeline import build_pipeline
    from opinion_mining.service import OpinionService
    from opinion_mining.spam_filter import SpamFilter
    use_tf = engine == "Transformer"
    pipeline = build_pipeline("v2", transformer=use_tf)
    if use_tf:
        pipeline.scorer.base.score_clause("warm up")      # download/load now, not at first submit
    return OpinionService(get_db(path), pipeline, SpamFilter(), pipeline_version=f"v2-{engine.lower()}")


def plot(fig):
    """Streamlit renamed use_container_width -> width='stretch'; support both."""
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ callbacks --
def cb_add_website():
    try:
        wid = get_db(DB_FILE).add_website(st.session_state.get("new_site_name", ""),
                                          st.session_state.get("new_site_url", ""))
        st.session_state["site_id"] = wid
        st.session_state["flash"] = ("success", "Website added.")
        st.session_state["new_site_name"] = st.session_state["new_site_url"] = ""
    except ValueError as e:
        st.session_state["flash"] = ("error", str(e))


def cb_submit():
    wid, text = st.session_state["site_id"], st.session_state.get("comment_text", "")
    try:
        res = get_service(st.session_state.get("engine", "VADER"), DB_FILE).submit_comment(wid, text)
        st.session_state["last_result"] = {"website_id": wid, "result": res}
        if res.accepted:
            st.session_state["comment_text"] = ""
    except ValueError as e:
        st.session_state["last_result"] = {"website_id": wid, "error": str(e)}


def cb_example(key: str):
    st.session_state["comment_text"] = EXAMPLES[key]


# ------------------------------------------------------------------- sidebar --
db = get_db(DB_FILE)
with st.sidebar:
    st.header("🛍️ Opinion Mining")
    sites = db.list_websites()
    if sites:
        ids = [s["id"] for s in sites]
        if st.session_state.get("site_id") not in ids:
            st.session_state["site_id"] = ids[0]
        info = {s["id"]: s for s in sites}
        st.selectbox("Website", ids, key="site_id",
                     format_func=lambda i: f"{info[i]['name']}  ({info[i]['n_comments']} comments)")

    with st.expander("➕ Add a website", expanded=not sites):
        st.text_input("Name", key="new_site_name", max_chars=100)
        st.text_input("URL (optional)", key="new_site_url")
        st.button("Add website", on_click=cb_add_website, key="add_site_btn")
    if "flash" in st.session_state:
        kind, msg = st.session_state.pop("flash")
        getattr(st, kind)(msg)

    st.divider()
    options = ["VADER", "Transformer"] if TF_AVAILABLE else ["VADER"]
    st.radio("Sentiment engine", options, key="engine",
             help="VADER = fast lexicon rules. Transformer = DistilBERT (needs `pip install "
                  "transformers torch`; downloads ~250 MB on first use).")
    if not TF_AVAILABLE:
        st.caption("Transformer option hidden: `transformers`/`torch` not installed.")
    st.caption("Aspects: " + ", ".join(ASPECT_DISPLAY_NAMES[a] for a in ASPECTS))

# ----------------------------------------------------------------------- main --
st.title("Website Evaluation Using Opinion Mining")
st.caption("Aspect-based sentiment analysis of customer comments - ratings per aspect, "
           "an overall trust score, and spam filtering.")

if not sites:
    st.info("No websites yet. Add one in the sidebar, or run "
            "`python -m opinion_mining.seed_demo` to load demo data.")
    st.stop()

site = info[st.session_state["site_id"]]
try:
    with st.spinner("Loading models..."):
        service = get_service(st.session_state.get("engine", "VADER"), DB_FILE)
except FileNotFoundError:
    st.error("Trained models not found. Run `python -m opinion_mining.aspect_ml` and "
             "`python -m opinion_mining.spam_filter` first.")
    st.stop()
except Exception as e:                       # e.g. no internet for the transformer download
    st.session_state["engine"] = "VADER"
    st.error(f"Could not load the transformer ({e}). Switched back to VADER - reload the page.")
    st.stop()

st.subheader(site["name"])
if site["url"]:
    st.caption(site["url"])
tab_rate, tab_add, tab_recent = st.tabs(["📊 Ratings", "✍️ Add a comment", "🗂 Recent comments"])


def chip_line(aspects):
    return chips(aspects, ASPECT_DISPLAY_NAMES)


# ---------------------------------------------------------------- ratings tab --
with tab_rate:
    report = service.get_report(site["id"])
    rating = report["rating"]
    label, emoji = trust_label(rating.trust_score)
    left, right = st.columns([1, 1.4])
    with left:
        plot(trust_gauge(rating.trust_score))
    with right:
        st.metric("Overall trust score", "-" if rating.trust_score is None else f"{rating.trust_score:.0f} / 100")
        st.markdown(f"### {emoji} {label}")
        st.markdown(f"Based on **{report['n_comments']}** accepted comments · evidence: "
                    f"**{CONF_EMOJI.get(rating.confidence_label, '⚪')} {rating.confidence_label}**"
                    + (f" · {report['n_spam_blocked']} spam comment(s) excluded" if report["n_spam_blocked"] else ""))
        if rating.veto_applied:
            st.warning("⚠️ Trust score capped: customers repeatedly report non-genuine products.")
        if rating.confidence_label == "Low":
            st.info("Few comments so far - treat these ratings as provisional.")

    st.markdown("#### Ratings by aspect")
    c1, c2 = st.columns(2)
    fig_stars, fig_mix = stars_figure(rating), sentiment_mix_figure(rating)
    if fig_stars is None:
        st.info("No aspect ratings yet - add a comment!")
    else:
        with c1:
            plot(fig_stars)
        with c2:
            plot(fig_mix)
    for a in rating.aspects.values():
        cols = st.columns([2, 2, 3, 2])
        cols[0].markdown(f"**{a.display_name}**")
        cols[1].markdown(render_stars(a.stars))
        cols[2].caption("no mentions yet" if a.stars is None else
                        f"👍 {a.n_pos} · 😐 {a.n_neu} · 👎 {a.n_neg}  (raw avg {a.raw_stars:.1f}★)")
        cols[3].caption(f"{CONF_EMOJI[a.confidence_label]} {a.confidence_label} confidence")

    st.markdown("#### Top comments")
    p_col, n_col = st.columns(2)
    for col, title, items in [(p_col, "👍 Most positive", report["top_positive"]),
                              (n_col, "👎 Most negative", report["top_negative"])]:
        with col:
            st.markdown(f"**{title}**")
            if not items:
                st.caption("None yet.")
            for c in items:
                with st.container(border=True):
                    st.write(f"“{c['text']}”")
                    st.caption(f"{chip_line(c['aspects'])}  ·  score {c['polarity']:+.2f}")

    with st.expander("ℹ️ How are these numbers computed?"):
        st.markdown(
            "- **Stars**: each mention's sentiment score is mapped to 1-5 stars, then **smoothed** "
            "toward 3.0 as if the site had 5 neutral reviews already - so one lucky comment can't "
            "give 5.0 stars.\n"
            "- **Confidence**: `n / (n + 5)` - Low below 5 mentions, High from 15.\n"
            "- **Trust score**: importance- and confidence-weighted average of the aspect stars "
            "(genuineness counts most), rescaled to 0-100, capped at 40 if the site shows clear "
            "evidence of selling fakes.\n"
            "- **Spam**: flagged comments are kept for audit but have zero weight.")

# ----------------------------------------------------------------- submit tab --
with tab_add:
    st.markdown("Write a comment about this website. Try an example:")
    ex_cols = st.columns(len(EXAMPLES))
    for col, key in zip(ex_cols, EXAMPLES):
        col.button(key, key=f"ex_{key}", on_click=cb_example, args=(key,))
    with st.form("comment_form", clear_on_submit=False):
        st.text_area("Your comment", key="comment_text", max_chars=2000, height=130,
                     placeholder="e.g. Product was original but delivery took 12 days...")
        st.form_submit_button("Submit comment", on_click=cb_submit, key="submit_btn", type="primary")

    last = st.session_state.get("last_result")
    if last and last["website_id"] == site["id"]:
        if "error" in last:
            st.error(last["error"])
        else:
            res = last["result"]
            (st.success if res.status == "accepted" else st.warning if res.status == "accepted_low_weight"
             else st.error)(res.message)
            if res.analysis and res.analysis.aspects:
                st.markdown("**What the system understood:**")
                for a in res.analysis.aspects:
                    st.markdown(f"{LABEL_EMOJI[a.label]} **{ASPECT_DISPLAY_NAMES[a.aspect]}** - "
                                f"{a.label} ({a.score:+.2f})  ·  _“{'; '.join(a.evidence)}”_")
                if res.analysis.is_mixed:
                    st.caption("Mixed comment: it contains both positive and negative aspects.")
            with st.expander("Spam check details"):
                st.write(f"Spam probability: **{res.verdict.prob:.2f}** → *{res.verdict.label}* "
                         f"(weight {res.verdict.weight})")
                st.write("Reasons: " + (", ".join(res.verdict.reasons) or "none"))

# ---------------------------------------------------------------- recent tab --
with tab_recent:
    show_spam = st.checkbox("Also show comments held back as spam", key="show_spam")
    rows = db.recent_comments(site["id"], limit=25, include_spam=show_spam)
    if not rows:
        st.caption("No comments yet.")
    for c in rows:
        with st.container(border=True):
            st.write(c["text"])
            tag = "🚫 held back as spam" if c["spam_label"] == "spam" else chip_line(c["aspects"])
            st.caption(f"{tag}  ·  {c['created_at'][:16].replace('T', ' ')} UTC")
