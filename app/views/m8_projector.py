import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from core import analytics as A  # noqa: E402
from core import ml  # noqa: E402
from core.ui import get_df, insight, kpi_row, page_header, style  # noqa: E402

df = get_df()
page_header("Module 8", "Population Projector",
            "Pick any district, adjust the growth scenario and get an ML-assisted projection for 2030 and 2035.")

bundle = ml.ensure_model(df)
names = sorted(df["district"])
c1, c2 = st.columns([1, 2])
with c1:
    dist = st.selectbox("District", names, index=names.index("Sheikhupura") if "Sheikhupura" in names else 0)
    base = ml.project_district(df, dist, bundle)
    r = df[df["district"] == dist].iloc[0]
    st.markdown("**2023 census values (auto-filled)**")
    st.dataframe({
        "Indicator": ["Province", "Population 2023", "Population 2017", "Observed growth", "Urban %",
                      "Household size", "Area km²", "Literacy"],
        "Value": [r["province"], f"{r['pop_2023']:,}", f"{r['pop_2017']:,}", f"{r['growth_rate']:.2f}%",
                  f"{r['urban_pct']:.1f}%", f"{r['avg_hh_size']:.2f}", f"{r['area_sqkm']:,.0f}",
                  f"{r['literacy_rate']:.1f}%"]}, hide_index=True, width="stretch")
    method = st.radio("Default growth assumption", ["blend", "model", "trend"], horizontal=True,
                      format_func={"blend": "Blend (50/50)", "model": "ML model", "trend": "2017-23 trend"}.get)
    default = {"blend": 0.5 * base["model_rate"] + 0.5 * base["trend_rate"], "model": base["model_rate"],
               "trend": base["trend_rate"]}[method]
    rate = st.slider("Growth rate scenario (% per year)", -1.0, 8.0, float(round(default, 2)), 0.05,
                     help="Simulate e.g. a new motorway or industrial zone raising growth, or family-planning lowering it.")
    st.caption(f"Observed trend **{base['trend_rate']:.2f}%** · ML model expects **{base['model_rate']:.2f}%**")
    go_btn = st.button("🔮 Project", type="primary", width="stretch")

with c2:
    p = ml.project_district(df, dist, bundle, rate=rate)
    pj = p["projections"]
    kpi_row([
        ("2023 census", A.fmt_pop(p["pop_2023"]), "baseline", "#9a9993"),
        ("Projected 2030", A.fmt_pop(pj[2030]), f"+{A.fmt_pop(pj[2030] - p['pop_2023'])}", "#7c3aed"),
        ("Projected 2035", A.fmt_pop(pj[2035]), f"+{A.fmt_pop(pj[2035] - p['pop_2023'])}", "#5b21b6"),
    ])
    years = ["2017 (census)", "2023 (census)", "2030 (projected)", "2035 (projected)"]
    vals = [p["pop_2017"], p["pop_2023"], pj[2030], pj[2035]]
    fig = go.Figure(go.Bar(x=years, y=vals, marker_color=["#c4b5fd", "#a78bfa", "#7c3aed", "#5b21b6"],
                           text=[A.fmt_pop(v) for v in vals], textposition="inside", insidetextanchor="start",
                           textfont=dict(color="#ffffff", size=13), showlegend=False,
                           hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"))
    # uncertainty band from model vs trend disagreement
    lo_r, hi_r = sorted([base["model_rate"], base["trend_rate"]])
    lo_r, hi_r = min(lo_r, rate) - 0.3, max(hi_r, rate) + 0.3
    for i, y in [(2, 2030), (3, 2035)]:
        lo_v = p["pop_2023"] * (1 + lo_r / 100) ** (y - 2023)
        hi_v = p["pop_2023"] * (1 + hi_r / 100) ** (y - 2023)
        fig.add_scatter(x=[years[i], years[i]], y=[lo_v, hi_v], mode="lines+markers", showlegend=False,
                        line=dict(color="#0b0b0b", width=1.5), marker=dict(symbol="line-ew", size=14,
                                                                          line=dict(width=1.5, color="#0b0b0b")),
                        hovertemplate=f"Scenario range {lo_r:.2f}-{hi_r:.2f}%<br>%{{y:,.0f}}<extra></extra>")
    fig.update_layout(yaxis_tickformat="~s", yaxis_range=[0, max(vals) * 1.25])
    st.plotly_chart(style(fig, 430, f"{dist}: census vs projection at {rate:.2f}%/yr (whiskers = scenario range)"),
                    width="stretch")
    insight("🤖 " + ml.insight_sentence(p), ai=True)
    add = pj[2035] - p["pop_2023"]
    st.markdown("##### Planning implications by 2035 (rule-of-thumb)")
    kpi_row([
        ("New households", A.fmt_pop(add / p["avg_hh_size"]), "extra homes needed", "#7c3aed"),
        ("School-age kids*", A.fmt_pop(add * r["age_0_14_pct"] / 100 * 0.6), "additional pupils", "#f97316"),
        ("Water**", f"+{add * 120 / 1e6:,.0f} ML", "megalitres per day", "#14b8a6"),
        ("Health units***", f"+{add / 25000:,.0f}", "basic health units", "#ec4899"),
    ])
    st.caption("*~60% of the 0-14 share is school-going age · **120 litres/person/day · ***1 BHU per 25,000 people. "
               "Illustrative planning ratios.")

if go_btn:
    projs = st.session_state.setdefault("projections", [])
    projs[:] = [x for x in projs if x["district"] != dist] + [p]
    fig2, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.bar(["2017", "2023", "2030", "2035"], [v / 1e6 for v in vals], color=["#c4b5fd", "#a78bfa", "#7c3aed", "#5b21b6"])
    ax.set_title(f"{dist}: population projection at {rate:.2f}%/yr (millions)", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    buf = io.BytesIO()
    fig2.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig2)
    from core.ui import register_chart
    register_chart(f"Projection - {dist}", buf.getvalue())
    st.toast(f"Projection for {dist} saved to the PDF report", icon="✅")

if st.session_state.get("projections"):
    st.markdown("##### Projections saved for the report")
    st.dataframe([{"District": x["district"], "Rate %/yr": round(x["rate_used"], 2),
                   "2023": f"{x['pop_2023']:,}", "2030": f"{x['projections'][2030]:,.0f}",
                   "2035": f"{x['projections'][2035]:,.0f}"} for x in st.session_state["projections"]],
                 hide_index=True, width="stretch")
