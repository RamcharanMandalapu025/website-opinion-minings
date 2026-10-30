"""
Step 9a: presentation helpers (NO Streamlit import -> unit-testable).

Colour/label rules and the three Plotly figures used by the website page:
  * trust_gauge          overall trust score 0-100
  * stars_figure         stars per aspect; bar OPACITY encodes confidence
  * sentiment_mix_figure share of positive / neutral / negative mentions per aspect
"""
from typing import Optional, Tuple

import plotly.graph_objects as go

from .aggregation import WebsiteRating

CONF_OPACITY = {"High": 1.0, "Medium": 0.75, "Low": 0.5}
CONF_EMOJI = {"High": "🟢", "Medium": "🟡", "Low": "🟠", "No data": "⚪"}
LABEL_EMOJI = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}
_RED, _AMBER, _GREEN = (214, 39, 40), (242, 169, 0), (44, 160, 44)


def _mix(c1, c2, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def star_rgb(stars: float) -> Tuple[int, int, int]:
    """1 star = red, 3 = amber, 5 = green (linear blend in between)."""
    s = max(1.0, min(5.0, stars))
    return _mix(_RED, _AMBER, (s - 1) / 2) if s <= 3 else _mix(_AMBER, _GREEN, (s - 3) / 2)


def star_color(stars: float, alpha: float = 1.0) -> str:
    r, g, b = star_rgb(stars)
    return f"rgba({r},{g},{b},{alpha})"


def trust_label(score: Optional[float]) -> Tuple[str, str]:
    """(text, emoji) for a 0-100 trust score."""
    if score is None:
        return "Not enough data", "⚪"
    if score >= 75:
        return "Highly trusted", "🟢"
    if score >= 60:
        return "Generally trusted", "🟢"
    if score >= 40:
        return "Mixed reviews", "🟡"
    return "Low trust - be careful", "🔴"


def chips(aspects: dict, names: dict) -> str:
    """{'delivery': 'Negative'} -> '🔴 Delivery Time'."""
    return " · ".join(f"{LABEL_EMOJI[l]} {names[a]}" for a, l in aspects.items()) or "no topics detected"


def _transparent(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def trust_gauge(score: Optional[float]) -> go.Figure:
    value = 0 if score is None else score
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": " / 100"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": star_color(1 + value / 25) if score is not None else "gray"},
               "steps": [{"range": [0, 40], "color": "rgba(214,39,40,0.18)"},
                         {"range": [40, 60], "color": "rgba(242,169,0,0.18)"},
                         {"range": [60, 100], "color": "rgba(44,160,44,0.18)"}]}))
    return _transparent(fig, 230)


def stars_figure(rating: WebsiteRating) -> Optional[go.Figure]:
    """Horizontal bars (1-5 stars). Fainter bar = lower confidence. None if no data."""
    items = [a for a in rating.aspects.values() if a.stars is not None][::-1]  # first aspect on top
    if not items:
        return None
    fig = go.Figure(go.Bar(
        orientation="h", base=1, x=[a.stars - 1 for a in items],
        y=[f"{a.display_name}<br><sup>{a.n:g} mentions · {a.confidence_label}</sup>" for a in items],
        marker={"color": [star_color(a.stars, CONF_OPACITY.get(a.confidence_label, 0.5)) for a in items]},
        text=[f"{a.stars:.1f} ★" for a in items], textposition="inside", insidetextanchor="end",
        customdata=[[a.stars, a.raw_stars, a.n, a.confidence_label] for a in items],
        hovertemplate=("<b>%{y}</b><br>Rating: %{customdata[0]:.2f}★ (raw average "
                       "%{customdata[1]:.2f}★)<br>Confidence: %{customdata[3]}<extra></extra>")))
    fig.add_vline(x=3, line_dash="dot", line_color="gray")
    fig.update_xaxes(range=[1, 5], tickvals=[1, 2, 3, 4, 5], title="Stars (faded bar = low confidence)")
    fig.update_layout(title="Rating per aspect")
    return _transparent(fig, 80 + 62 * len(items))


def sentiment_mix_figure(rating: WebsiteRating) -> Optional[go.Figure]:
    items = [a for a in rating.aspects.values() if a.stars is not None][::-1]
    if not items:
        return None
    names = [a.display_name for a in items]
    fig = go.Figure()
    for label, attr, color in [("Positive", "n_pos", "#2ca02c"), ("Neutral", "n_neu", "#9aa0a6"),
                               ("Negative", "n_neg", "#d62728")]:
        fig.add_bar(name=label, y=names, x=[getattr(a, attr) for a in items], orientation="h",
                    marker_color=color, hovertemplate=f"{label}: %{{x}} mentions<extra></extra>")
    fig.update_layout(barmode="stack", barnorm="percent", title="Sentiment mix (% of mentions)",
                      legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(ticksuffix="%")
    return _transparent(fig, 80 + 62 * len(items))
