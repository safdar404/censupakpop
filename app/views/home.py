import streamlit as st
import streamlit.components.v1 as components

from core import analytics as A
from core.config import PROGRAM, STUDENT, SUPERVISOR
from core.ui import card, get_df, insight, kpi_row
from core.webviz import render

df = get_df()
nat = A.national_summary(df)

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">{PROGRAM}</div>
  <h1>CensusInsight <span>PK</span></h1>
  <p>An AI-powered population intelligence system built on Pakistan's first digital census, the
  <b>7th Population &amp; Housing Census 2023</b> by the Pakistan Bureau of Statistics. Clean the data, explore it, map it,
  predict it and ask questions in plain English.</p>
  <div class="chips">
    <span class="chip">👤 Student: {STUDENT}</span><span class="chip">🎓 Supervisor: {SUPERVISOR}</span>
    <span class="chip">🗂️ {nat['districts']} districts · {nat['provinces']} provinces</span>
    <span class="chip">🌍 3D globe</span><span class="chip">🤖 Agentic AI + ML</span><span class="chip">🐳 Docker-ready</span>
  </div>
</div>""", unsafe_allow_html=True)

components.html(render("globe.html", df), height=612, scrolling=False)

kpi_row([
    ("Population 2023", A.fmt_pop(nat["population"]), f"{nat['growth_rate']:.2f}% a year since 2017", "#7c3aed"),
    ("Households", A.fmt_pop(nat["households"]), f"avg {nat['avg_hh_size']:.2f} persons", "#14b8a6"),
    ("Urban", f"{nat['urban_pct']:.1f}%", f"rural {nat['rural_pct']:.1f}%", "#ec4899"),
    ("Literacy (10+)", f"{nat['literacy_rate']:.1f}%", f"{nat['females_per_100_males']:.1f} females / 100 males", "#f97316"),
])

st.markdown("#### Key findings")
c1, c2 = st.columns([3, 2])
with c1:
    for s in A.auto_insights(df)[:5]:
        insight(s)
with c2:
    st.markdown("##### Try the AI agent")
    q = st.text_input("Ask anything about the census", placeholder="e.g. top 5 fastest growing districts in Punjab")
    if q:
        st.session_state["agent_prefill"] = q
        st.switch_page("views/ai_agent.py")
    st.caption("Examples: *compare Lahore and Faisalabad* · *project Quetta to 2035* · *most rural districts in Sindh*")

st.markdown("#### What's inside")
mods = [
    ("MODULE 1", "Data loading & cleaning", "Upload PBS Excel/CSV tables; auto-detects headers, merged cells, blanks and units.", False),
    ("MODULE 2", "National dashboard", "KPI cards, province bars, share donut, male vs female, click-to-drill-down.", False),
    ("MODULE 3", "Urban vs rural", "Stacked bars, most urban/rural districts, 2017→2023 urbanisation shift.", False),
    ("MODULE 4", "Density & growth", "Density ranking, high/low growth zones, household size.", False),
    ("MODULE 5", "Gender & demography", "Sex ratio, gender outliers, age pyramid groups, literacy gap.", False),
    ("MODULE 6", "GeoAI maps", "Interactive choropleth for 7 indicators + exportable static map.", False),
    ("MODULE 7", "ML model lab", "4 regressors compared with CV, MAE/R², feature importance, saved model.", True),
    ("MODULE 8", "Population projector", "Pick a district, tune growth, project 2030/2035 with auto-insight.", True),
    ("AI AGENT", "AI census analyst", "Agentic tool-calling assistant answering questions from real data.", True),
    ("AI INSIGHTS", "Anomalies & typologies", "Isolation-Forest anomaly flags and K-Means district clusters.", True),
    ("NEW", "Geo-infographics", "Province silhouettes, district bubble map, 'Pakistan as 100 people'.", False),
    ("NEW", "Statistics lab", "Treemap, sunburst, sankey, radar, violin, waterfall, correlation, Lorenz/Gini.", True),
    ("MODULE 9", "Report generator", "One-click professional PDF with all charts, tables and ML results.", False),
    ("DEPLOY", "Docker & cloud", "Containerised; deploy to Hugging Face Spaces, Render or any VPS.", False),
]
for i in range(0, len(mods), 4):
    cols = st.columns(4)
    for c, m in zip(cols, mods[i:i + 4]):
        with c:
            card(m[0], m[1], m[2], ai=m[3])
    st.write("")
