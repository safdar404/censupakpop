import datetime as dt

import streamlit as st

from core import ml
from core.config import REPORT_DIR, STUDENT, SUPERVISOR
from core.report import generate_report
from core.ui import get_df, page_header

df = get_df()
page_header("Module 9", "Automated Report Generation",
            "One click produces a professional PDF with tables, charts, ML results, projections and your session's maps.")

c1, c2 = st.columns([2, 3])
with c1:
    student = st.text_input("Student name", STUDENT)
    supervisor = st.text_input("Supervisor", SUPERVISOR)
    defaults = [x["district"] for x in st.session_state.get("projections", [])] or ["Lahore", "Peshawar", "Quetta"]
    picks = st.multiselect("Districts to project in the report (2-3)", sorted(df["district"]),
                           default=[d for d in defaults if d in set(df["district"])][:3], max_selections=4)
    save_copy = st.checkbox("Also save a copy on the server (reports/ folder)", value=True)
    build = st.button("📄 Generate PDF report", type="primary", width="stretch")
with c2:
    st.markdown("**The report will contain:**")
    imgs = st.session_state.get("session_images", [])
    st.markdown(f"""
- Header with project, student, supervisor and date · KPI strip
- Dataset introduction · **AI executive summary** {'(from AI Insights ✓)' if 'exec_summary' in st.session_state else '(auto facts)'}
- National summary table · province comparison table, bar & donut charts
- Top 10 most / least dense districts + density chart
- Urban vs rural charts & high-growth zones · gender ratio & literacy findings
- ML leaderboard (MAE, R²) + predicted-vs-actual plot
- Projections for: **{', '.join(picks) or '—'}**
- Session charts/maps: **{len(imgs)}** {'(' + ', '.join(t for t, _ in imgs) + ')' if imgs else '(export maps in Module 6 or save projections in Module 8)'}
""")

if build:
    with st.spinner("Building charts and PDF…"):
        res = st.session_state.get("ml_result") or ml.cached_result(df) or ml.train_and_compare(df)
        st.session_state["ml_result"] = res
        bundle = ml.ensure_model(df)
        saved = {x["district"]: x for x in st.session_state.get("projections", [])}
        projs = [saved.get(d) or ml.project_district(df, d, bundle) for d in picks]
        pdf = generate_report(df, student, supervisor, res, projs, st.session_state.get("session_images", []),
                              st.session_state.get("exec_summary"))
    name = f"CensusInsight_PK_Report_{dt.datetime.now():%Y%m%d_%H%M}.pdf"
    if save_copy:
        (REPORT_DIR / name).write_bytes(pdf)
        st.caption(f"Saved to `reports/{name}`")
    st.success(f"Report ready - {len(pdf) / 1024:,.0f} KB")
    st.download_button("⬇️ Download PDF report", pdf, name, "application/pdf", type="primary")
