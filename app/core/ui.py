"""Shared UI kit: theme CSS, page headers, KPI cards, Plotly styling and
session-state helpers.  Keeps every page visually consistent."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from .config import APP_NAME, PROCESSED_CSV, PROVINCE_COLORS, llm_available
from .data_loader import PROVINCE_SHORT

# ---------------------------------------------------------------- plotly theme
pio.templates["census"] = go.layout.Template(
    layout=dict(
        font=dict(family="Plus Jakarta Sans, Inter, Segoe UI, sans-serif", size=13, color="#0f172a"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=["#7c3aed", "#ec4899", "#f97316", "#14b8a6", "#2563eb", "#f59e0b", "#10b981", "#f43f5e"],
        xaxis=dict(gridcolor="#eef0f6", linecolor="#d5d4cf", zerolinecolor="#d5d4cf", tickfont=dict(color="#52514e")),
        yaxis=dict(gridcolor="#eef0f6", linecolor="#d5d4cf", zerolinecolor="#d5d4cf", tickfont=dict(color="#52514e")),
        hoverlabel=dict(bgcolor="#0f172a", bordercolor="#0f172a", font=dict(color="#ffffff", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
        margin=dict(l=10, r=10, t=78, b=10), title=dict(x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=15)),
        bargap=0.28,
    ))
pio.templates.default = "plotly_white+census"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea { font-family: 'Plus Jakarta Sans', 'Inter', 'Segoe UI', sans-serif; }
.stApp { background: #f6f7fb; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.hero { position: relative; overflow: hidden; border-radius: 24px; padding: 28px 32px; margin-bottom: 16px; color: #fff;
  background: radial-gradient(600px 300px at 10% 0%, rgba(217,70,239,.55), transparent 60%),
              radial-gradient(700px 360px at 95% 10%, rgba(249,115,22,.45), transparent 60%),
              radial-gradient(800px 400px at 60% 120%, rgba(20,184,166,.45), transparent 60%),
              linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%); }
.hero h1 { color:#fff; font-size: 2.3rem; font-weight: 800; margin: 0 0 6px 0; }
.hero h1 span { background: linear-gradient(90deg,#fde68a,#f9a8d4,#a5f3fc); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: #e0e7ff; font-size: 1.02rem; margin: 0; max-width: 820px; line-height: 1.55; }
.hero .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .7rem; color: #f5d0fe; font-weight: 700; margin-bottom: 8px; }
.hero .chips { margin-top: 14px; }
.chip { display:inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
        color:#fff; border-radius: 999px; padding: 5px 13px; font-size: .78rem; margin: 0 6px 6px 0; }
.page-head { margin: 0 0 16px 0; padding-bottom: 12px; border-bottom: 1px solid #e8eaf2; }
.page-head .mod { display:inline-block; font-weight:800; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase;
  background: linear-gradient(120deg,#7c3aed,#d946ef,#f97316); -webkit-background-clip:text; background-clip:text; color:transparent; }
.page-head h2 { margin: 2px 0 4px 0; font-weight: 800; font-size: 1.8rem; color:#0f172a; }
.page-head p { color:#475569; margin:0; font-size: .98rem; }
.kpi { position:relative; overflow:hidden; background:#ffffff; border:1px solid #e8eaf2; border-radius:18px; padding:16px 18px; height: 100%;
       box-shadow: 0 8px 24px -14px rgba(76,29,149,.25); }
.kpi .label { color:#475569; font-size:.72rem; font-weight:700; text-transform: uppercase; letter-spacing:.06em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kpi .value { color:#0f172a; font-size:clamp(1.15rem, 1.9vw, 1.9rem); white-space:nowrap; font-weight:800; line-height:1.2; margin-top:4px; letter-spacing:-.02em; }
.kpi .sub { color:#64748b; font-size:.8rem; margin-top:2px; }
.kpi .bar { position:absolute; left:0; right:0; top:0; height:4px; border-radius:0; width:auto; margin:0; }
.card { background:#fff; border:1px solid #e8eaf2; border-radius:18px; padding:18px 20px; height:100%;
        box-shadow: 0 8px 24px -16px rgba(76,29,149,.25); transition: transform .15s; }
.card:hover { transform: translateY(-2px); }
.card h4 { margin:0 0 6px 0; font-size: 1rem; font-weight: 800; }
.card p { margin:0; color:#475569; font-size:.9rem; }
.card .tag { display:inline-block; font-size:.68rem; font-weight:800; color:#7c3aed; background:#f3e8ff;
             border-radius:999px; padding:3px 10px; margin-bottom:8px; letter-spacing:.06em; }
.card.ai .tag { color:#be185d; background:#fce7f3; }
.insight { border-left: 4px solid #7c3aed; background: linear-gradient(90deg,#f5f3ff,#fff); padding: 10px 14px; border-radius: 0 12px 12px 0;
           margin: 7px 0; color:#1e293b; font-size: .94rem; line-height:1.5; }
.insight.ai { border-left-color:#d946ef; background: linear-gradient(90deg,#fdf4ff,#fff); }
.badge-ok { color:#0f766e; background:#ccfbf1; padding:3px 9px; border-radius:999px; font-size:.75rem; font-weight:800; }
.badge-off { color:#92400e; background:#fef3c7; padding:3px 9px; border-radius:999px; font-size:.75rem; font-weight:800; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#1e1b4b 0%,#312e81 60%,#4c1d95 100%); }
[data-testid="stSidebar"] * { color: #e0e7ff; }
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"],
[data-testid="stSidebar"] a[aria-current="page"] { background: rgba(255,255,255,.16); border-radius: 10px; }
[data-testid="stSidebar"] .badge-off { color:#92400e !important; }
[data-testid="stSidebar"] .badge-ok { color:#0f766e !important; }
[data-testid="stSidebar"] code { color:#fde68a; background: rgba(255,255,255,.1); }
[data-testid="stMetricValue"] { font-weight: 800; }
div[data-testid="stDataFrame"] { border-radius: 12px; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(120deg,#7c3aed,#d946ef,#f97316); border: 0; box-shadow: 0 6px 16px -6px rgba(217,70,239,.6); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; padding: 6px 16px; background: #eef0f8; }
.stTabs [aria-selected="true"] { background: linear-gradient(120deg,#7c3aed,#d946ef); color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
iframe { border-radius: 24px; }
footer {visibility: hidden;}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(module: str, title: str, subtitle: str):
    st.markdown(f"<div class='page-head'><div class='mod'>{module}</div><h2>{title}</h2><p>{subtitle}</p></div>",
                unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", color: str = "#7c3aed"):
    st.markdown(f"<div class='kpi'><div class='bar' style='background:{color}'></div><div class='label'>{label}</div>"
                f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def kpi_row(items: list[tuple]):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        with c:
            kpi(*it)


def insight(text: str, ai: bool = False):
    st.markdown(f"<div class='insight{' ai' if ai else ''}'>{text}</div>", unsafe_allow_html=True)


def card(tag: str, title: str, body: str, ai: bool = False):
    st.markdown(f"<div class='card{' ai' if ai else ''}'><span class='tag'>{tag}</span><h4>{title}</h4><p>{body}</p></div>",
                unsafe_allow_html=True)


def style(fig, height=420, title=None):
    fig.update_layout(height=height, template="plotly_white+census")
    if title:
        fig.update_layout(title=title)
    return fig


def province_filter(df: pd.DataFrame, key: str, label="Province", include_all=True):
    opts = (["All provinces"] if include_all else []) + list(dict.fromkeys(df["province"]))
    choice = st.selectbox(label, opts, key=key)
    return df if choice == "All provinces" else df[df["province"] == choice], choice


# ---------------------------------------------------------------- data/session
@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_CSV)


def get_df() -> pd.DataFrame:
    if "df" not in st.session_state:
        st.session_state["df"] = load_default()
        st.session_state["data_source"] = "Bundled PBS Census 2023 district tables (136 districts)"
    return st.session_state["df"]


def register_chart(title: str, png: bytes):
    """Charts/maps registered here are embedded in the PDF report (Module 9)."""
    imgs = st.session_state.setdefault("session_images", [])
    imgs[:] = [i for i in imgs if i[0] != title] + [(title, png)]


def sidebar_status():
    with st.sidebar:
        st.markdown(f"### 🌍 {APP_NAME}")
        st.caption(st.session_state.get("data_source", "Bundled PBS Census 2023 data"))
        if llm_available():
            st.markdown("<span class='badge-ok'>● LLM agent online</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-off'>● Offline AI mode</span>", unsafe_allow_html=True)
            st.caption("Set `OPENAI_API_KEY` to enable the LLM brain. All features work offline.")


def short(p: str) -> str:
    return PROVINCE_SHORT.get(p, p)


PCOLORS = PROVINCE_COLORS
