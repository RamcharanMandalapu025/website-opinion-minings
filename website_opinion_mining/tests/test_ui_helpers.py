from opinion_mining.aggregation import AspectRecord, aggregate
from opinion_mining.ui_helpers import (chips, sentiment_mix_figure, star_rgb, stars_figure,
                                       trust_gauge, trust_label)
from opinion_mining.config import ASPECT_DISPLAY_NAMES


def recs(aspect, score, n, start=0):
    return [AspectRecord(aspect, score, comment_id=start + i) for i in range(n)]


def test_star_colours_run_red_to_green():
    r1, g1, _ = star_rgb(1.0); r5, g5, _ = star_rgb(5.0)
    assert r1 > g1 and g5 > r5
    assert star_rgb(0) == star_rgb(1) and star_rgb(9) == star_rgb(5)      # clamped


def test_trust_label_bands():
    assert trust_label(None)[0] == "Not enough data"
    assert [trust_label(s)[0] for s in (90, 65, 50, 10)] == \
        ["Highly trusted", "Generally trusted", "Mixed reviews", "Low trust - be careful"]


def test_chips():
    assert chips({"delivery": "Negative", "pricing": "Positive"}, ASPECT_DISPLAY_NAMES) == \
        "🔴 Delivery Time · 🟢 Pricing"
    assert chips({}, ASPECT_DISPLAY_NAMES) == "no topics detected"


def test_stars_figure_empty_and_ordering():
    assert stars_figure(aggregate([])) is None and sentiment_mix_figure(aggregate([])) is None
    r = aggregate(recs("delivery", 0.8, 20) + recs("pricing", -0.8, 3, start=100))
    fig = stars_figure(r)
    bar = fig.data[0]
    assert len(bar.y) == 2                                   # only aspects with data
    assert "Pricing" in bar.y[0] and "Delivery" in bar.y[1]   # reversed -> delivery drawn on top
    assert bar.base == 1 and all(0 <= x <= 4 for x in bar.x)        # bars live inside 1..5 stars


def test_low_confidence_bars_are_fainter():
    r = aggregate(recs("delivery", 0.8, 40) + recs("pricing", 0.8, 1, start=100))
    colors = stars_figure(r).data[0].marker.color
    alpha = lambda c: float(c.rstrip(")").split(",")[-1])
    assert alpha(colors[0]) < alpha(colors[1])               # pricing (n=1) fainter than delivery


def test_mix_and_gauge():
    r = aggregate(recs("delivery", 0.8, 3) + recs("delivery", -0.8, 2, start=50))
    fig = sentiment_mix_figure(r)
    assert [t.name for t in fig.data] == ["Positive", "Neutral", "Negative"]
    assert (fig.data[0].x[0], fig.data[2].x[0]) == (3, 2)
    assert trust_gauge(72).data[0].value == 72 and trust_gauge(None).data[0].value == 0
