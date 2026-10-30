from opinion_mining.preprocessing import clean_text, split_clauses, preprocess
from opinion_mining.aspect_keyword import KeywordAspectDetector


def test_clean_urls_emoji_repeats():
    out = clean_text("Sooooo slow!!!! see https://x.com/a 😡")
    assert "http" not in out and "😡" not in out
    assert "soo" in out.lower() and "ooo" not in out
    assert "angry" in out


def test_contractions_become_explicit_negation():
    assert clean_text("Didn't work, can't refund, won't reply") == \
        "Did not work, cannot refund, will not reply"


def test_clean_handles_non_string():
    assert clean_text(None) == ""


def test_brief_example_splits_into_three():
    c = preprocess("Product was original but delivery took 12 days and support never replied.")
    assert c == ["Product was original", "delivery took 12 days", "support never replied"]


def test_short_fragments_not_split():
    assert preprocess("Fast and cheap delivery") == ["Fast and cheap delivery"]


def test_comma_split_when_both_sides_long_enough():
    assert preprocess("Fast delivery, great price") == ["Fast delivery", "great price"]


def test_negation_stays_in_clause():
    # "not" must stay attached to "bad" -- Step 3 (VADER) depends on it.
    assert preprocess("It's not bad, but the price is too high.") == \
        ["It's not bad", "the price is too high"]


def test_detect_by_clause():
    d = KeywordAspectDetector()
    res = d.detect_by_clause("Product was original but delivery took 12 days and support never replied.")
    assert [a for _, a in res] == [["genuineness"], ["delivery"], ["support"]]
