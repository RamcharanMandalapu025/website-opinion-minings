from opinion_mining.aspect_keyword import KeywordAspectDetector

d = KeywordAspectDetector()


def aspects(text):
    return set(d.detect_aspects(text))


def test_example_from_brief():
    got = aspects("Product was original but delivery took 12 days and support never replied.")
    assert {"genuineness", "delivery", "support"} <= got


def test_whole_word_matching():
    assert "delivery" not in aspects("The latest model feels sturdy.")


def test_inflections():
    assert "delivery" in aspects("It was delivered quickly")
    assert "pricing" in aspects("Prices are too high")


def test_phrases():
    assert "support" in aspects("Their customer service was rude")
    assert "refund" in aspects("I want my money back")


def test_no_aspect():
    assert aspects("Nice colours.") == set()
