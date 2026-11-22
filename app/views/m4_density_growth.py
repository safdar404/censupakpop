import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import analytics as A
from core.config import HIGH_GROWTH, LOW_GROWTH
from core.ui import PCOLORS, get_df, insight, kpi_row, page_header, province_filter, short, style

df = get_df()
p = A.province_summary(df)
page_header("Module 4", "Population Density & Growth Rate Analysis",
            "How crowded each district is and how fast it is growing - the basis for planning schools, hospitals and roads.")

sub, prov = province_filter(df, "dg_prov")
nat = A.national_summary(sub)
kpi_row([
    ("Density", f"{nat['density']:,.0f} /km²", "people per square km", "#7c3aed"),
    ("Growth 2017-23", f"{nat['growth_rate']:.2f}%", "compound annual rate", "#f97316"),
    (f"High growth (>{HIGH_GROWTH:g}%)", f"{(sub['growth_rate'] > HIGH_GROWTH).sum()}", "districts", "#f43f5e"),
    ("Avg household size", f"{nat['avg_hh_size']:.2f}", "persons per household", "#14b8a6"),
])
st.write("")

log = st.toggle("Log scale (density spans 6 to 55,000 people/km²)", value=True)
d = sub.sort_values("density", ascending=False)
fig = px.bar(d, x="district", y="density", color="province", color_discrete_map=PCOLORS,
             hover_data={"density": ":,.0f", "pop_2023": ":,", "area_sqkm": ":,.0f", "province": True})
fig.update_layout(xaxis_title=None, yaxis_title="people / km²", xaxis_tickangle=-60, bargap=0.15,
                  yaxis_type="log" if log else "linear", xaxis=dict(tickfont=dict(size=9)))
st.plotly_chart(style(fig, 470, "Population density of every district (highest → lowest), coloured by province"), width="stretch")

c1, c2 = st.columns(2)
cols = ["district", "province", "pop_2023", "area_sqkm", "density"]
fmt = {"pop_2023": "{:,}", "area_sqkm": "{:,.0f}", "density": "{:,.1f}"}
with c1:
    st.markdown("##### Top 10 most densely populated")
    st.dataframe(A.top_bottom(sub, "density", 10, cols=cols).style.format(fmt), width="stretch")
with c2:
    st.markdown("##### 10 least densely populated")
    st.dataframe(A.top_bottom(sub, "density", 10, ascending=True, cols=cols).style.format(fmt), width="stretch")

st.markdown("### Growth rate 2017 → 2023")
c1, c2 = st.columns([2, 3])
with c1:
    fig = go.Figure(go.Bar(x=[short(i) for i in p.index], y=p["growth_rate"],
                           marker_color=[PCOLORS[i] for i in p.index], text=p["growth_rate"].map("{:.2f}%".format),
                           textposition="outside", hovertemplate="%{x}: %{y:.2f}%/yr<extra></extra>"))
    nat_all = A.national_summary(df)["growth_rate"]
    fig.add_hline(y=nat_all, line_dash="dash", line_color="#52514e",
                  annotation_text=f"Pakistan {nat_all:.2f}%", annotation_position="top left")
    fig.update_layout(yaxis_ticksuffix="%", yaxis_range=[0, p["growth_rate"].max() * 1.25])
    st.plotly_chart(style(fig, 380, "Annual growth by province"), width="stretch")
with c2:
    picks = st.multiselect("Selected districts", sorted(sub["district"]),
                           default=list(sub.nlargest(8, "pop_2023")["district"]))
    dd = sub[sub["district"].isin(picks)].sort_values("growth_rate")
    fig = go.Figure(go.Bar(y=dd["district"], x=dd["growth_rate"], orientation="h",
                           marker_color=np.where(dd["growth_rate"] > HIGH_GROWTH, "#f43f5e",
                                                 np.where(dd["growth_rate"] < LOW_GROWTH, "#9a9993", "#7c3aed")),
                           hovertemplate="%{y}: %{x:.2f}%/yr<extra></extra>"))
    fig.update_layout(xaxis_ticksuffix="%")
    st.plotly_chart(style(fig, 380, "Growth of selected districts (red = high-growth zone)"), width="stretch")

hi, lo = A.growth_zones(sub)
c1, c2 = st.columns(2)
gfmt = {"pop_2017": "{:,}", "pop_2023": "{:,}", "growth_rate": "{:.2f}%"}
with c1:
    st.markdown(f"##### 🔴 High Growth Zones (> {HIGH_GROWTH:g}% / yr) - {len(hi)} districts")
    st.dataframe(hi.style.format(gfmt), hide_index=True, width="stretch", height=360)
with c2:
    st.markdown(f"##### ⚪ Low Growth Zones (< {LOW_GROWTH:g}% / yr) - {len(lo)} districts")
    st.dataframe(lo.style.format(gfmt), hide_index=True, width="stretch")
    insight("Very slow or negative growth may indicate out-migration, boundary changes between censuses, or "
            "under-counting. These districts deserve verification before planning decisions.")

st.markdown("### Average household size")
fig = go.Figure(go.Bar(x=[short(i) for i in p.index], y=p["avg_hh_size"], marker_color="#14b8a6",
                       text=p["avg_hh_size"].map("{:.2f}".format), textposition="outside",
                       hovertemplate="%{x}: %{y:.2f} persons<extra></extra>"))
fig.update_layout(yaxis_range=[0, p["avg_hh_size"].max() * 1.2])
c1, c2 = st.columns([2, 3])
with c1:
    st.plotly_chart(style(fig, 360, "Persons per household by province"), width="stretch")
with c2:
    fig = px.box(df, x="province", y="avg_hh_size", color="province", color_discrete_map=PCOLORS, points="all",
                 hover_name="district")
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="persons / household")
    st.plotly_chart(style(fig, 360, "District spread of household size"), width="stretch")
insight("Larger households are a common proxy for crowding and lower income: KP and Balochistan have the largest, "
        "Sindh (dominated by Karachi) the smallest.")
