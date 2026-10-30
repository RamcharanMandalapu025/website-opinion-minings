import pytest
from opinion_mining.aggregation import (AspectRecord, aggregate, confidence_label,
                                        confidence_of, rank_comments, records_from_analysis,
                                        render_stars, score_to_stars)
from opinion_mining.pipeline import PipelineV1


def recs(aspect, score, n, weight=1.0, start_id=0):
    return [AspectRecord(aspect, score, weight=weight, comment_id=start_id + i) for i in range(n)]


def test_score_to_stars_endpoints_and_saturation():
    assert score_to_stars(0) == 3.0
    assert score_to_stars(0.6) == 5.0 and score_to_stars(0.99) == 5.0      # saturates
    assert score_to_stars(-0.6) == 1.0 and score_to_stars(-1) == 1.0
    assert score_to_stars(0.3) == pytest.approx(4.0)


def test_one_lucky_review_does_not_give_five_stars():
    r = aggregate(recs("delivery", 0.9, 1)).aspects["delivery"]
    assert r.raw_stars == 5.0                      # the raw mean is fooled...
    assert r.stars == pytest.approx(3.33, abs=0.01)  # ...the smoothed rating is not
    assert r.confidence_label == "Low"


def test_many_reviews_converge_to_the_raw_mean():
    r = aggregate(recs("delivery", 0.9, 200)).aspects["delivery"]
    assert r.stars > 4.8 and r.confidence_label == "High"


def test_confidence_bands():
    assert confidence_of(1) == pytest.approx(1 / 6)
    assert [confidence_label(confidence_of(n)) for n in (1, 4, 5, 14, 15, 60)] == \
        ["Low", "Low", "Medium", "Medium", "High", "High"]
    assert confidence_label(0) == "No data"


def test_weights_scale_influence():
    full = aggregate(recs("pricing", -0.9, 10, weight=1.0)).aspects["pricing"]
    half = aggregate(recs("pricing", -0.9, 10, weight=0.5)).aspects["pricing"]
    ignored = aggregate(recs("pricing", -0.9, 10, weight=0.0)).aspects["pricing"]
    assert half.n == 5.0 and full.n == 10.0
    assert half.stars > full.stars                  # less evidence -> closer to prior 3.0
    assert ignored.stars is None                    # weight 0 (spam) = invisible


def test_no_data():
    w = aggregate([])
    assert w.trust_score is None and w.n_comments == 0
    assert all(a.stars is None for a in w.aspects.values())
    assert render_stars(None) == "no ratings yet"


def test_good_site_beats_bad_site():
    good = [r for a in ("genuineness", "delivery", "quality") for r in recs(a, 0.7, 30)]
    bad = [r for a in ("genuineness", "delivery", "quality") for r in recs(a, -0.7, 30)]
    g, b = aggregate(good).trust_score, aggregate(bad).trust_score
    assert g > 80 and b < 25 and g > b


def test_genuineness_veto_caps_trust():
    # Fakes (negative) but everything else glowing and cheap
    records = (recs("genuineness", -0.8, 30) + recs("pricing", 0.8, 100)
               + recs("delivery", 0.8, 100) + recs("quality", 0.8, 100) + recs("support", 0.8, 100))
    w = aggregate(records)
    assert w.veto_applied and w.trust_score <= 40


def test_veto_needs_enough_evidence():
    records = recs("genuineness", -0.8, 2) + recs("pricing", 0.8, 100) + recs("delivery", 0.8, 100)
    assert not aggregate(records).veto_applied      # 2 comments are not proof


def test_trust_follows_confident_aspects():
    # 1 terrible support comment must not drown 40 good delivery comments
    w = aggregate(recs("support", -0.9, 1) + recs("delivery", 0.8, 40))
    assert w.trust_score > 70


def test_counts_and_rows():
    records = recs("delivery", 0.8, 3) + recs("delivery", -0.8, 2) + recs("delivery", 0.0, 1)
    a = aggregate(records).aspects["delivery"]
    assert (a.n_pos, a.n_neu, a.n_neg) == (3, 1, 2)
    row = [r for r in aggregate(records).to_rows() if r["aspect"] == "Delivery Time"][0]
    assert row["negative"] == 2


def test_render_stars():
    assert render_stars(4.3).startswith("★★★★☆") and render_stars(1.0).startswith("★☆☆☆☆")


def test_integration_with_pipeline_and_ranking():
    pipe = PipelineV1()
    texts = ["Product was original but delivery took 12 days and support never replied.",
             "Excellent quality and great price",
             "Fake product and no refund",
             "Great price"]
    analyses = [pipe.analyze(t) for t in texts]
    records = [r for i, a in enumerate(analyses) for r in records_from_analysis(a, comment_id=i)]
    w = aggregate(records)
    assert w.n_comments == 4
    assert w.aspects["pricing"].stars > 3 and w.aspects["refund"].stars < 3
    pos, neg = rank_comments(analyses, top_n=1)
    assert "price" in pos[0]["comment"].lower() and "Fake" in neg[0]["comment"]
