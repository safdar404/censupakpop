"""
CensusInsight PK - entry point
Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="CensusInsight PK", page_icon="🇵🇰", layout="wide",
                   initial_sidebar_state="expanded")

from core.ui import apply_theme, get_df, sidebar_status  # noqa: E402

apply_theme()
get_df()

APP_DIR = Path(__file__).resolve().parent

# Page files live in app/views/ (older copies used app/pages/). Pick whichever exists.
V = next((d for d in (APP_DIR / "views", APP_DIR / "pages") if d.is_dir()), APP_DIR / "views")

PAGE_SPEC = {
    "Start": [
        ("home.py", "Home", "🏠"),
        ("m1_data.py", "1 · Data Loading & Cleaning", "🗂️"),
    ],
    "Analysis": [
        ("m2_dashboard.py", "2 · National Dashboard", "📊"),
        ("m3_urban_rural.py", "3 · Urban vs Rural", "🏙️"),
        ("m4_density_growth.py", "4 · Density & Growth", "📈"),
        ("m5_gender.py", "5 · Gender & Demography", "👥"),
        ("m6_maps.py", "6 · GeoAI Maps", "🗺️"),
        ("infographics.py", "Geo-infographics", "🎨"),
        ("stats_lab.py", "Statistics Lab", "📐"),
    ],
    "AI & Machine Learning": [
        ("m7_ml.py", "7 · ML Model Lab", "🧠"),
        ("m8_projector.py", "8 · Population Projector", "🔮"),
        ("ai_agent.py", "AI Census Agent", "🤖"),
        ("ai_insights.py", "AI Insights & Anomalies", "✨"),
    ],
    "Deliver": [
        ("m9_report.py", "9 · Report Generator", "📄"),
        ("about.py", "Architecture & About", "ℹ️"),
    ],
}

pages, missing, has_default = {}, [], False
for section, items in PAGE_SPEC.items():
    for fname, title, icon in items:
        path = V / fname
        if not path.is_file():
            missing.append(fname)
            continue
        pages.setdefault(section, []).append(
            st.Page(path, title=title, icon=icon, default=not has_default))
        has_default = True

if missing:
    st.sidebar.warning(f"{len(missing)} page file(s) missing from `{V}`: " + ", ".join(missing))

if not pages:
    st.error(
        "No page files found. The app expects the page scripts in:\n\n"
        f"`{V}`\n\n"
        "Your copy of the project is incomplete. Re-extract the full project zip so that "
        "`app/views/home.py`, `app/views/m1_data.py`, ... exist, then restart Streamlit."
    )
    st.stop()

nav = st.navigation(pages, expanded=True)
sidebar_status()
nav.run()
