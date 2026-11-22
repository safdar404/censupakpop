import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import analytics as A
from core.ui import get_df, insight, page_header, short, style

df = get_df()
p = A.province_summary(df)
nat = A.national_summary(df)
page_header("Module 5", "Gender Ratio & Demographic Breakdown",
            "Sex ratio, age structure and the literacy gap between men and women across provinces and districts.")

c1, c2 = st.columns([2, 3])
with c1:
    x = [short(i) for i in p.index] + ["Pakistan"]
    y = list(p["females_per_100_males"]) + [nat["females_per_100_males"]]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=["#7c3aed"] * len(p) + ["#4c1d95"],
                           text=[f"{v:.1f}" for v in y], textposition="outside",
                           hovertemplate="%{x}: %{y:.1f} females per 100 males<extra></extra>"))
    fig.add_hline(y=100, line_dash="dot", line_color="#9a9993", annotation_text="parity (100)")
    fig.update_layout(yaxis_range=[80, 104])
    st.plotly_chart(style(fig, 380, "Females per 100 males"), width="stretch")
with c2:
    out = A.gender_outliers(df)
    st.markdown(f"##### ⚖️ Significantly unbalanced districts ({len(out)}) - more than 1.5 SD from the mean")
    st.dataframe(out.style.format({"male": "{:,}", "female": "{:,}", "females_per_100_males": "{:.1f}"}),
                 hide_index=True, width="stretch", height=330)

st.markdown("### Gender ratio grid - every district, by province")
grid = df[["province", "district", "females_per_100_males"]].copy()
grid["rank"] = grid.groupby("province").cumcount()
mx = grid["rank"].max() + 1
ncol = 12
grid["row"] = grid["rank"] // ncol
grid["col"] = grid["rank"] % ncol
grid["prow"] = grid["province"].map(short) + " " + (grid["row"] + 1).astype(str)
fig = go.Figure(go.Heatmap(
    x=grid["col"], y=grid["prow"], z=grid["females_per_100_males"], text=grid["district"],
    colorscale=[[0, "#c4521f"], [0.35, "#f3b496"], [0.6, "#e4e3df"], [0.8, "#c4b5fd"], [1, "#5b21b6"]],
    zmin=82, zmax=104, xgap=3, ygap=3, colorbar=dict(title="F/100M"),
    hovertemplate="<b>%{text}</b><br>%{z:.1f} females per 100 males<extra></extra>"))
fig.add_trace(go.Scatter(x=grid["col"], y=grid["prow"], mode="text", text=grid["district"].str[:9],
                         textfont=dict(size=9, color="#0b0b0b"), hoverinfo="skip"))
fig.update_layout(xaxis=dict(visible=False), yaxis=dict(autorange="reversed", title=None), showlegend=False)
st.plotly_chart(style(fig, 520, "Orange = fewer females, blue = more females (hover for exact values)"), width="stretch")

st.markdown("### Age structure")
age = p[["age_0_14_pct", "age_15_64_pct", "age_65_plus_pct"]].reset_index()
age = age.melt(id_vars="province", var_name="group", value_name="pct")
age["group"] = age["group"].map({"age_0_14_pct": "Children 0-14", "age_15_64_pct": "Working age 15-64",
                                 "age_65_plus_pct": "Elderly 65+"})
age["province"] = age["province"].map(short)
c1, c2 = st.columns([3, 2])
with c1:
    fig = px.bar(age, x="province", y="pct", color="group", barmode="group",
                 color_discrete_sequence=["#f97316", "#7c3aed", "#14b8a6"], text=age["pct"].map("{:.1f}".format))
    fig.update_traces(textposition="outside", hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>")
    fig.update_layout(yaxis_ticksuffix="%", xaxis_title=None, yaxis_title=None)
    st.plotly_chart(style(fig, 400, "Population by broad age group (% of province)"), width="stretch")
with c2:
    dep = p.assign(dependency=(p["age_0_14_pct"] + p["age_65_plus_pct"]) / p["age_15_64_pct"] * 100)
    insight(f"The youth share ranges from {p['age_0_14_pct'].min():.1f}% ({short(p['age_0_14_pct'].idxmin())}) to "
            f"{p['age_0_14_pct'].max():.1f}% ({short(p['age_0_14_pct'].idxmax())}).")
    insight(f"Dependency ratio (dependants per 100 working-age): highest in {short(dep['dependency'].idxmax())} "
            f"({dep['dependency'].max():.0f}), lowest in {short(dep['dependency'].idxmin())} ({dep['dependency'].min():.0f}).")
    insight("A young population means rising demand for schools today and for jobs within 10-15 years.")

st.markdown("### Literacy: male vs female")
c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_bar(x=[short(i) for i in p.index], y=p["literacy_male"], name="Male", marker_color="#7c3aed",
                hovertemplate="%{x}<br>Male literacy %{y:.1f}%<extra></extra>")
    fig.add_bar(x=[short(i) for i in p.index], y=p["literacy_female"], name="Female", marker_color="#f97316",
                hovertemplate="%{x}<br>Female literacy %{y:.1f}%<extra></extra>")
    fig.update_layout(barmode="group", yaxis_ticksuffix="%", yaxis_range=[0, 100])
    st.plotly_chart(style(fig, 380, "Literacy rate (age 10+) by sex"), width="stretch")
with c2:
    gap = (df["literacy_male"] - df["literacy_female"])
    t = df.assign(gap=gap).nlargest(10, "gap")[["district", "province", "literacy_male", "literacy_female", "gap"]]
    st.markdown("##### Widest literacy gender gaps")
    st.dataframe(t.style.format({"literacy_male": "{:.1f}%", "literacy_female": "{:.1f}%", "gap": "{:.1f} pp"}),
                 hide_index=True, width="stretch")
