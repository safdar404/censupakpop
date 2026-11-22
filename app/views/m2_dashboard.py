import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import analytics as A
from core.ui import PCOLORS, get_df, kpi_row, page_header, short, style

df = get_df()
nat = A.national_summary(df)
p = A.province_summary(df)

page_header("Module 2", "National & Provincial Population Overview",
            "Pakistan's population at a glance - click a province bar to drill down into its districts.")

kpi_row([
    ("Total national population", A.fmt_pop(nat["population"]), f"{nat['population']:,} people", "#7c3aed"),
    ("Total households", A.fmt_pop(nat["households"]), f"{nat['households']:,}", "#14b8a6"),
    ("Urban population", f"{nat['urban_pct']:.1f}%", A.fmt_pop(df['urban_pop'].sum()) + " people", "#ec4899"),
    ("Rural population", f"{nat['rural_pct']:.1f}%", A.fmt_pop(df['rural_pop'].sum()) + " people", "#f97316"),
])
st.write("")

c1, c2 = st.columns([3, 2])
pp = p.reset_index()
pp["label"] = pp["pop_2023"].map(A.fmt_pop)
with c1:
    fig = px.bar(pp.iloc[::-1], x="pop_2023", y="province", orientation="h", color="province",
                 color_discrete_map=PCOLORS, text="label", custom_data=["province", "share_pct", "districts"])
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="<b>%{customdata[0]}</b><br>Population: %{x:,}<br>Share: %{customdata[1]:.1f}%"
                                    "<br>Districts: %{customdata[2]}<extra></extra>", marker_line_width=0)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None, xaxis_tickformat="~s",
                      xaxis_range=[0, pp["pop_2023"].max() * 1.15])
    ev = st.plotly_chart(style(fig, 380, "Population by province (click a bar to drill down)"),
                         width="stretch", on_select="rerun", selection_mode="points", key="prov_bar")
with c2:
    fig = go.Figure(go.Pie(labels=[short(i) for i in pp["province"]], values=pp["pop_2023"], hole=0.55,
                           marker=dict(colors=[PCOLORS[i] for i in pp["province"]], line=dict(color="#fff", width=2)),
                           sort=False, direction="clockwise", textinfo="label+percent", textposition="inside",
                           insidetextorientation="horizontal",
                           hovertemplate="<b>%{label}</b><br>%{value:,} people<br>%{percent}<extra></extra>"))
    fig.update_layout(showlegend=False, uniformtext_minsize=10, uniformtext_mode="hide", annotations=[dict(text=f"<b>{A.fmt_pop(nat['population'])}</b><br>Pakistan",
                                                          showarrow=False, font_size=14)])
    st.plotly_chart(style(fig, 380, "Share of national population"), width="stretch")

c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_bar(x=[short(i) for i in p.index], y=p["male"], name="Male", marker_color="#7c3aed",
                hovertemplate="%{x}<br>Male: %{y:,}<extra></extra>")
    fig.add_bar(x=[short(i) for i in p.index], y=p["female"], name="Female", marker_color="#f97316",
                hovertemplate="%{x}<br>Female: %{y:,}<extra></extra>")
    fig.update_layout(barmode="group", yaxis_tickformat="~s", yaxis_title=None)
    st.plotly_chart(style(fig, 360, "Male vs female population by province"), width="stretch")
with c2:
    t = p[["pop_2023", "share_pct", "growth_rate", "urban_pct", "households"]].copy()
    t.columns = ["Population", "Share %", "Growth %/yr", "Urban %", "Households"]
    st.markdown("##### Province summary")
    st.dataframe(t.style.format({"Population": "{:,.0f}", "Share %": "{:.1f}", "Growth %/yr": "{:.2f}",
                                 "Urban %": "{:.1f}", "Households": "{:,.0f}"}), width="stretch")

# ------------------------------------------------ drill down
st.markdown("### 🔎 District drill-down")
sel = None
try:
    pts = ev.selection.points if ev else []
    if pts:
        sel = pts[0].get("y") or pts[0].get("customdata", [None])[0]
except Exception:  # noqa: BLE001
    pts = []
provs = list(p.index)
sel = st.selectbox("Province", provs, index=provs.index(sel) if sel in provs else 0, key="drill_prov")
d = df[df["province"] == sel].sort_values("pop_2023", ascending=False)
c1, c2 = st.columns([3, 2])
with c1:
    fig = px.bar(d, x="district", y="pop_2023", color_discrete_sequence=[PCOLORS[sel]],
                 hover_data={"pop_2023": ":,", "growth_rate": ":.2f", "urban_pct": ":.1f", "density": ":,.0f"})
    fig.update_layout(xaxis_title=None, yaxis_title=None, yaxis_tickformat="~s", xaxis_tickangle=-45)
    st.plotly_chart(style(fig, 420, f"Districts of {sel} by population"), width="stretch")
with c2:
    t = d[["district", "pop_2023", "growth_rate", "urban_pct", "density"]].rename(columns={
        "district": "District", "pop_2023": "Population", "growth_rate": "Growth %", "urban_pct": "Urban %",
        "density": "Density"})
    t["Population"] = t["Population"].map(A.fmt_pop)
    st.dataframe(t.style.format({"Growth %": "{:.2f}", "Urban %": "{:.1f}", "Density": "{:,.0f}"}),
                 hide_index=True, width="stretch", height=420)
