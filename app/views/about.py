import streamlit as st

from core.config import PROGRAM, STUDENT, SUPERVISOR
from core.ui import page_header

page_header("About", "Architecture, Technology & Deployment",
            f"{PROGRAM} · Student: {STUDENT} · Supervisor: {SUPERVISOR}")

st.markdown("#### System architecture")
st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#d5d4cf", fillcolor="#ffffff"];
  edge [color="#9a9993"];
  subgraph cluster0 {label="DATA"; fontname="Helvetica"; color="#ecebe7";
    pbs [label="PBS Census 2023\\nExcel / CSV tables", fillcolor="#f3e8ff"];
    hdx [label="District boundaries\\nGeoJSON (GADM / HDX)", fillcolor="#f3e8ff"]; }
  load [label="Module 1\\nLoader & cleaner\\n(pandas, openpyxl)"];
  store [label="Clean district dataset\\n136 districts × 33 fields", fillcolor="#fdf3dc"];
  subgraph cluster1 {label="ANALYTICS"; fontname="Helvetica"; color="#ecebe7";
    m2 [label="2 Dashboard"]; m3 [label="3 Urban/Rural"]; m4 [label="4 Density/Growth"]; m5 [label="5 Gender/Age"]; m6 [label="6 GeoAI maps\\n(Folium)"]; }
  subgraph cluster2 {label="AI / ML"; fontname="Helvetica"; color="#ecebe7";
    m7 [label="7 ML lab\\n4 regressors + CV", fillcolor="#efedfc"]; mdl [label="Saved model\\n(joblib)", fillcolor="#efedfc"];
    m8 [label="8 Projector", fillcolor="#efedfc"]; ag [label="AI Agent\\ntool-calling LLM\\n+ offline planner", fillcolor="#efedfc"];
    an [label="Isolation Forest\\nK-Means", fillcolor="#efedfc"]; }
  rep [label="9 PDF report\\n(FPDF2 + Matplotlib)", fillcolor="#e6f6ef"];
  ui [label="Streamlit web UI\\nDocker container", fillcolor="#e6f6ef"];
  pbs -> load -> store; hdx -> m6;
  store -> {m2 m3 m4 m5 m6 m7 an ag};
  m7 -> mdl -> {m8 ag}; {m2 m3 m4 m5 m6 m7 m8 an} -> rep; rep -> ui; ag -> ui;
}
""")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
#### Technology stack
| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data | Pandas, NumPy, OpenPyXL |
| Visualisation | Plotly, Matplotlib, Seaborn-style theme |
| Geospatial | Folium, streamlit-folium, GeoJSON (GADM / HDX) |
| Machine learning | Scikit-learn: Linear, Ridge, Random Forest, Gradient Boosting, Isolation Forest, K-Means |
| Model saving | Joblib |
| Agentic AI | OpenAI-compatible function calling (OpenAI / Groq / Ollama) + offline intent planner |
| Report | FPDF2 + Matplotlib |
| UI | Streamlit (multi-page, custom CSS) |
| DevOps | Docker, docker-compose, Hugging Face Spaces / Render |
""")
with c2:
    st.markdown("""
#### AI advancements beyond the proposal
1. **Agentic AI analyst** - plans → calls 8 data tools → answers with verified numbers; reasoning trace shown.
2. **Grounded generative AI** - executive briefing written only from computed facts (no hallucinated figures).
3. **Explainable ML** - permutation importance + Random-Forest importance + naive baseline comparison.
4. **Anomaly detection** - Isolation Forest flags districts for data-quality review.
5. **District typologies** - K-Means groups districts into policy-relevant clusters.
6. **Scenario projector** with uncertainty whiskers and planning-need estimates.
7. **Offline-first** - every AI feature works without an API key (safe for a live viva).

#### Data source
Pakistan Bureau of Statistics - *7th Population & Housing Census 2023, Detailed Results* (Tables 1, 5, 12, district level).
""")
