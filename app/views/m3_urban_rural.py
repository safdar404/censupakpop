import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import analytics as A
from core.ui import PCOLORS, get_df, insight, page_header, province_filter, short, style

df = get_df()
p = A.province_summary(df)

page_header("Module 3", "Urban vs Rural Analysis",
            "How Pakistan's population splits between cities and villages, and where urbanisation is accelerating.")

sub, prov = province_filter(df, "ur_prov", "Filter all charts by province")
scope = "Pakistan" if prov == "All provinces" else prov
u_rate = A.cagr(sub["urban_pop"].sum(), sub["urban_pop_2017"].sum())
r_rate = A.cagr(sub["rural_pop"].sum(), sub["rural_pop_2017"].sum())
insight(f"<b>{scope}</b>: {sub['urban_pop'].sum() / sub['pop_2023'].sum() * 100:.1f}% urban. Urban population grew "
        f"<b>{u_rate:.2f}%/yr</b> vs <b>{r_rate:.2f}%/yr</b> rural (2017-23) - "
        f"{'cities are absorbing most of the growth.' if u_rate > r_rate else 'rural areas still grow faster.'}")

c1, c2 = st.columns(2)
pv = p if prov == "All provinces" else p.loc[[prov]]
with c1:
    fig = go.Figure()
    fig.add_bar(x=[short(i) for i in pv.index], y=pv["urban_pop"], name="Urban", marker_color="#7c3aed",
                hovertemplate="%{x}<br>Urban: %{y:,}<extra></extra>")
    fig.add_bar(x=[short(i) for i in pv.index], y=pv["rural_pop"], name="Rural", marker_color="#f97316",
                hovertemplate="%{x}<br>Rural: %{y:,}<extra></extra>", marker_line=dict(color="#fff", width=2))
    fig.update_layout(barmode="stack", yaxis_tickformat="~s")
    st.plotly_chart(style(fig, 380, "Urban and rural population by province"), width="stretch")
with c2:
    fig = go.Figure()
    fig.add_bar(x=[short(i) for i in pv.index], y=pv["urban_growth_rate"], name="Urban growth",
                marker_color="#7c3aed", hovertemplate="%{x}<br>Urban: %{y:.2f}%/yr<extra></extra>")
    fig.add_bar(x=[short(i) for i in pv.index], y=pv["rural_growth_rate"], name="Rural growth",
                marker_color="#f97316", hovertemplate="%{x}<br>Rural: %{y:.2f}%/yr<extra></extra>")
    fig.update_layout(barmode="group", yaxis_ticksuffix="%")
    st.plotly_chart(style(fig, 380, "Annual growth 2017-23: urban vs rural (migration signal)"), width="stretch")

st.markdown("### District urbanisation")
ps = st.selectbox("Province for district breakdown", list(p.index), key="ur_dist_prov",
                  index=list(p.index).index(prov) if prov in p.index else 0)
d = df[df["province"] == ps].sort_values("urban_pct", ascending=True)
fig = go.Figure()
fig.add_bar(y=d["district"], x=d["urban_pct"], orientation="h", name="Urban %", marker_color="#7c3aed",
            hovertemplate="<b>%{y}</b><br>Urban %{x:.1f}%<extra></extra>")
fig.add_bar(y=d["district"], x=d["rural_pct"], orientation="h", name="Rural %", marker_color="#f97316",
            hovertemplate="<b>%{y}</b><br>Rural %{x:.1f}%<extra></extra>", marker_line=dict(color="#fff", width=1))
fig.update_layout(barmode="stack", xaxis_ticksuffix="%", xaxis_range=[0, 100])
st.plotly_chart(style(fig, max(380, 20 * len(d)), f"{ps}: most urbanised (top) to most rural (bottom)"), width="stretch")

c1, c2 = st.columns(2)
cols = ["district", "province", "pop_2023", "urban_pct"]
fmt = {"pop_2023": "{:,}", "urban_pct": "{:.1f}%"}
with c1:
    st.markdown("##### 🏙️ Top 10 most urbanised districts")
    st.dataframe(A.top_bottom(sub, "urban_pct", 10, cols=cols).style.format(fmt), width="stretch")
with c2:
    st.markdown("##### 🌾 Top 10 most rural districts")
    t = A.top_bottom(sub, "rural_pct", 10, cols=["district", "province", "pop_2023", "rural_pct"])
    st.dataframe(t.style.format({"pop_2023": "{:,}", "rural_pct": "{:.1f}%"}), width="stretch")

st.markdown("### 2017 → 2023: which districts urbanised the most?")
ch = sub.dropna(subset=["urbanisation_change_pp"]).sort_values("urbanisation_change_pp", ascending=False).head(15)
fig = go.Figure()
for _, r in ch.iterrows():
    fig.add_shape(type="line", x0=r["urban_pct_2017"], x1=r["urban_pct"], y0=r["district"], y1=r["district"],
                  line=dict(color="#c3c2b7", width=2))
fig.add_scatter(x=ch["urban_pct_2017"], y=ch["district"], mode="markers", name="2017",
                marker=dict(size=11, color="#c4b5fd", line=dict(color="#fff", width=2)),
                hovertemplate="%{y}<br>2017: %{x:.1f}%<extra></extra>")
fig.add_scatter(x=ch["urban_pct"], y=ch["district"], mode="markers", name="2023",
                marker=dict(size=11, color="#5b21b6", line=dict(color="#fff", width=2)),
                hovertemplate="%{y}<br>2023: %{x:.1f}%<extra></extra>")
fig.update_layout(xaxis_ticksuffix="%", yaxis=dict(autorange="reversed"))
st.plotly_chart(style(fig, 520, "Urban share 2017 vs 2023 - 15 biggest shifts (percentage points)"), width="stretch")
top = ch.iloc[0]
insight(f"<b>{top['district']}</b> moved from {top['urban_pct_2017']:.1f}% to {top['urban_pct']:.1f}% urban "
        f"(+{top['urbanisation_change_pp']:.1f} pp). Large jumps can reflect real migration <i>or</i> re-classification "
        "of localities as urban between censuses.")

fig = px.scatter(sub, x="urban_pct", y="growth_rate", size="pop_2023", color="province",
                 color_discrete_map=PCOLORS,
                 hover_name="district", size_max=40,
                 hover_data={"pop_2023": ":,", "urban_pct": ":.1f", "growth_rate": ":.2f", "province": False})
fig.update_traces(marker=dict(line=dict(color="#fff", width=1)))
fig.update_layout(xaxis_title="Urban %", yaxis_title="Growth %/yr")
st.plotly_chart(style(fig, 460, "Urbanisation vs growth (bubble = population)"), width="stretch")
