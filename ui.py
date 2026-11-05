"""Visual design for the churn app: palette, page CSS, Plotly styling and small HTML blocks.

Presentation only. Nothing in this file touches the model or the data.
"""
import plotly.graph_objects as go
import streamlit as st

# Palette: warm paper background, ink text, deep teal for "stays" and neutral data,
# terracotta for churn and risk, amber for search time.
PAPER = "#F6F3EC"
CARD = "#FFFFFF"
INK = "#1F2421"
MUTED = "#6B6F68"
LINE = "#E6E0D4"
STAY = "#1D5C63"
CHURN = "#C65D3B"
AMBER = "#C9962E"
SOFT = "#9A9A92"

HEAD = "'Fraunces', Georgia, serif"
BODY = "'Inter', system-ui, sans-serif"
HEAD_PLAIN = "Fraunces, Georgia, serif"
BODY_PLAIN = "Inter, system-ui, sans-serif"

RISK_STYLE = {"Low": ("#E2EEEB", STAY), "Medium": ("#F7EBD2", "#7A5410"), "High": ("#F6DFD6", "#8E3A1C")}
SEG_COLOR = {"High": CHURN, "Medium": AMBER, "Low": STAY}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');

.stApp { background: __PAPER__; }
.stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp button, .stApp input {
  font-family: __BODY__;
}
.block-container { max-width: 1180px; padding-top: 2.6rem; padding-bottom: 4rem; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

h1, h2, h3, h4 { font-family: __HEAD__ !important; color: __INK__; letter-spacing: -0.01em; }
h1 { font-size: 2.5rem !important; font-weight: 600 !important; margin-bottom: .25rem !important; }
h2 { font-size: 1.3rem !important; font-weight: 500 !important; margin: 2.2rem 0 .8rem !important; }
h3, h4 { font-weight: 500 !important; }

[data-testid="stCaptionContainer"] p { color: __MUTED__; font-size: .85rem; line-height: 1.5; }

[data-testid="stMetric"] { background: __CARD__; border: 1px solid __LINE__ !important; border-radius: 14px; padding: .9rem 1.2rem; }
[data-testid="stMetricLabel"] p { font-size: .72rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: .07em; color: __MUTED__; }
[data-testid="stMetricValue"] { font-family: __HEAD__; font-weight: 500; font-size: 1.8rem; color: __INK__; font-variant-numeric: tabular-nums; }
[data-testid="stMetricDelta"] { font-size: .8rem; }

[data-testid="stPlotlyChart"] { background: __CARD__; border: 1px solid __LINE__; border-radius: 14px; padding: .5rem .8rem .2rem; }
[data-testid="stDataFrame"] { border: 1px solid __LINE__; border-radius: 12px; overflow: hidden; }
[data-testid="stForm"] { background: __CARD__; border: 1px solid __LINE__; border-radius: 14px; padding: 1.2rem 1.3rem .7rem; }

div[data-baseweb="tab-list"] { gap: 1.8rem; border-bottom: 1px solid __LINE__; }
button[data-baseweb="tab"] { font-size: .95rem; font-weight: 500; color: __MUTED__; padding: .6rem 0 .7rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: __INK__; }
div[data-baseweb="tab-highlight"] { background: __STAY__ !important; height: 2px !important; }
div[data-baseweb="tab-border"] { display: none !important; }

div[data-testid="stFormSubmitButton"] button { background: __STAY__; color: #FFFFFF; border: none; border-radius: 999px; padding: .5rem 1.4rem; font-weight: 600; }
div[data-testid="stFormSubmitButton"] button:hover { background: #164A50; color: #FFFFFF; }

.verdict { background: __CARD__; border: 1px solid __LINE__; border-left: 3px solid __CHURN__; border-radius: 14px; padding: 1rem 1.3rem; margin: .4rem 0 .8rem; }
.verdict p { font-family: __HEAD__; font-size: 1.18rem; line-height: 1.5; margin: 0; color: __INK__; }

.pill { display: inline-block; padding: .2rem .75rem; border-radius: 999px; font-weight: 600; font-size: .85rem; }
.pill.big { font-family: __HEAD__; font-size: 1.5rem; padding: .05rem 1rem; font-weight: 500; }

.seg { background: __CARD__; border: 1px solid __LINE__; border-top: 3px solid; border-radius: 14px; padding: 1rem 1.1rem; }
.seg-name { font-family: __HEAD__; font-size: 1.1rem; font-weight: 500; color: __INK__; }
.seg-stat { color: __MUTED__; font-size: .82rem; margin: .1rem 0 .6rem; }
.seg p { font-size: .92rem; line-height: 1.5; margin: 0 0 .7rem; color: __INK__; }
.seg-value { font-size: .82rem; color: __MUTED__; border-top: 1px solid __LINE__; padding-top: .5rem; font-variant-numeric: tabular-nums; }
"""


def _css() -> str:
    tokens = {"__PAPER__": PAPER, "__CARD__": CARD, "__INK__": INK, "__MUTED__": MUTED, "__LINE__": LINE,
              "__STAY__": STAY, "__CHURN__": CHURN, "__HEAD__": HEAD, "__BODY__": BODY}
    out = _CSS
    for k, v in tokens.items():
        out = out.replace(k, v)
    return out


def inject_css() -> None:
    """Add the page styles once, at the top of app.py."""
    st.html(f"<style>{_css()}</style>")


def dollar(text: str) -> str:
    """Stop Streamlit's markdown from reading two '$' signs as a math formula."""
    return text.replace("$", "&#36;")


def verdict(sentence: str) -> str:
    return f'<div class="verdict"><p>{dollar(sentence)}</p></div>'


def risk_pill(band: str, big: bool = False) -> str:
    bg, fg = RISK_STYLE[band]
    cls = "pill big" if big else "pill"
    return f'<span class="{cls}" style="background:{bg};color:{fg}">{band}</span>'


def segment_card(name: str, stat: str, text: str, value: str) -> str:
    color = SEG_COLOR[name]
    return (f'<div class="seg" style="border-top-color:{color}">'
            f'<div class="seg-name">{name} risk</div>'
            f'<div class="seg-stat">{dollar(stat)}</div>'
            f'<p>{dollar(text)}</p>'
            f'<div class="seg-value">{dollar(value)}</div></div>')


def style_fig(fig: go.Figure, height: int | None = 320) -> go.Figure:
    """One look for every chart: transparent background (the card shows through), Inter text,
    Fraunces titles, light grid lines, left-aligned titles."""
    title = fig.layout.title.text
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left",
                                     font=dict(family=HEAD_PLAIN, size=16, color=INK)))
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_PLAIN, size=12, color=INK),
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE, font=dict(family=BODY_PLAIN, color=INK)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED), orientation="h", y=-0.18, yanchor="top"),
        colorway=[STAY, CHURN, AMBER, SOFT],
    )
    axis = dict(linecolor=LINE, gridcolor=LINE, zerolinecolor=LINE, tickfont=dict(color=MUTED),
                title_font=dict(color=MUTED, size=11))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if height:
        fig.update_layout(height=height)
    return fig
