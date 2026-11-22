import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import analytics as A
from core.ui import PCOLORS, get_df, insight, page_header, short, style

df = get_df()
p = A.province_summary(df)
provs = list(p.index)
page_header("Statistics Lab", "Statistical Graphics",
            "Eleven views of the 136 districts: composition, flows, distributions, correlation and inequality.")


def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


tab1, tab2, tab3 = st.tabs(["🧩 Composition & flows", "📊 Distributions & profiles", "🔗 Relationships & inequality"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:  # treemap province -> district, shade by growth
        ids, labels, parents, values, colors, cd = ["Pakistan"], ["Pakistan"], [""], [0], ["#ffffff"], [None]
        for pr in provs:
            ids.append(pr); labels.append(short(pr)); parents.append("Pakistan"); values.append(0)
            colors.append(PCOLORS[pr]); cd.append(p.loc[pr, "growth_rate"])
        for r in df.itertuples():
            ids.append("d-" + r.district); labels.append(r.district); parents.append(r.province)
            values.append(r.pop_2023); cd.append(r.growth_rate)
            colors.append(rgba(PCOLORS[r.province], 0.35 + 0.65 * min(1, max(r.growth_rate, 0) / 5)))
        fig = go.Figure(go.Treemap(ids=ids, labels=labels, parents=parents, values=values, branchvalues="remainder",
                                   customdata=cd, marker=dict(colors=colors, line=dict(color="#fff", width=1.5)),
                                   texttemplate="<b>%{label}</b><br>%{value:.3s}", textfont=dict(color="#fff"),
                                   hovertemplate="<b>%{label}</b><br>%{value:,} people<br>Growth %{customdata:.2f}%/yr<extra></extra>"))
        fig.update_layout(margin=dict(l=4, r=4, t=56, b=4))
        st.plotly_chart(style(fig, 480, "Treemap: where Pakistanis live (shade = growth rate)"), width="stretch")
    with c2:  # sunburst province -> division -> district
        sid, sl, sp, sv, sc = [], [], [], [], []
        for pr in provs:
            sid.append(pr); sl.append(short(pr)); sp.append(""); sv.append(0); sc.append(PCOLORS[pr])
        seen = set()
        for r in df.itertuples():
            dv = f"{r.province}|{r.division or 'Other'}"
            if dv not in seen:
                seen.add(dv); sid.append(dv); sl.append(r.division or "Other"); sp.append(r.province); sv.append(0)
                sc.append(rgba(PCOLORS[r.province], 0.7))
            sid.append("s-" + r.district); sl.append(r.district); sp.append(dv); sv.append(r.pop_2023)
            sc.append(rgba(PCOLORS[r.province], 0.42))
        fig = go.Figure(go.Sunburst(ids=sid, labels=sl, parents=sp, values=sv, branchvalues="remainder", maxdepth=2,
                                    marker=dict(colors=sc, line=dict(color="#fff", width=1)), insidetextorientation="radial",
                                    hovertemplate="<b>%{label}</b><br>%{value:,} people<br>%{percentRoot:.1%} of Pakistan<extra></extra>"))
        fig.update_layout(margin=dict(l=4, r=4, t=56, b=4))
        st.plotly_chart(style(fig, 480, "Sunburst: province → division (click to open districts)"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:  # sankey
        nodes = [short(x) for x in provs] + ["Urban", "Rural"]
        src, tgt, val, lc = [], [], [], []
        for i, pr in enumerate(provs):
            src += [i, i]; tgt += [len(provs), len(provs) + 1]
            val += [p.loc[pr, "urban_pop"], p.loc[pr, "rural_pop"]]
            lc += [rgba(PCOLORS[pr], .45), rgba(PCOLORS[pr], .25)]
        fig = go.Figure(go.Sankey(arrangement="snap",
                                  node=dict(label=nodes, color=[PCOLORS[x] for x in provs] + ["#7c3aed", "#f97316"], pad=16, thickness=20),
                                  link=dict(source=src, target=tgt, value=val, color=lc,
                                            hovertemplate="%{source.label} → %{target.label}<br>%{value:,} people<extra></extra>")))
        st.plotly_chart(style(fig, 420, "Sankey: provinces flowing into urban and rural Pakistan"), width="stretch")
    with c2:  # waterfall
        nat = A.national_summary(df)
        x = ["2017"] + [f"+ {short(x)}" for x in provs] + ["2023"]
        y = [nat["population_2017"]] + [p.loc[x, "pop_2023"] - p.loc[x, "pop_2017"] for x in provs] + [0]
        txt = [A.fmt_pop(nat["population_2017"])] + ["+" + A.fmt_pop(v) for v in y[1:-1]] + [A.fmt_pop(nat["population"])]
        fig = go.Figure(go.Waterfall(x=x, y=y, measure=["absolute"] + ["relative"] * len(provs) + ["total"], text=txt,
                                     textposition="outside", connector=dict(line=dict(color="#c4b5fd", dash="dot")),
                                     increasing=dict(marker=dict(color="#ec4899")), totals=dict(marker=dict(color="#7c3aed")),
                                     hovertemplate="%{x}: %{y:,}<extra></extra>"))
        fig.update_layout(yaxis=dict(range=[nat["population_2017"] * 0.92, nat["population"] * 1.04], tickformat="~s"),
                          showlegend=False)
        st.plotly_chart(style(fig, 420, f"Waterfall: who added the {A.fmt_pop(nat['population'] - nat['population_2017'])} new people"),
                        width="stretch")

with tab2:
    c1, c2 = st.columns(2)
    with c1:  # radar
        dims = [("urban_pct", "Urban %"), ("growth_rate", "Growth"), ("literacy_rate", "Literacy"),
                ("avg_hh_size", "Household size"), ("age_0_14_pct", "Youth share"), ("females_per_100_males", "F/100 M")]
        fig = go.Figure()
        for pr in provs:
            r = [20 + 80 * (p.loc[pr, k] - p[k].min()) / ((p[k].max() - p[k].min()) or 1) for k, _ in dims]
            fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=[d[1] for d in dims] + [dims[0][1]], fill="toself", name=short(pr),
                                          fillcolor=rgba(PCOLORS[pr], .12), line=dict(color=PCOLORS[pr], width=2.5),
                                          customdata=[p.loc[pr, k] for k, _ in dims] + [p.loc[pr, dims[0][0]]],
                                          hovertemplate=short(pr) + "<br>%{theta}: %{customdata:.2f}<extra></extra>"))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False, range=[0, 105])))
        st.plotly_chart(style(fig, 440, "Radar: province profiles (scaled min → max)"), width="stretch")
    with c2:  # violin
        fig = go.Figure()
        for pr in provs:
            s = df[df["province"] == pr]
            fig.add_trace(go.Violin(y=s["growth_rate"], name=short(pr), text=s["district"], box_visible=True, meanline_visible=True,
                                    points="all", jitter=.4, pointpos=0, fillcolor=rgba(PCOLORS[pr], .25), line_color=PCOLORS[pr],
                                    marker=dict(color=PCOLORS[pr], size=5), hovertemplate="%{text}: %{y:.2f}%/yr<extra></extra>"))
        fig.update_layout(yaxis=dict(ticksuffix="%", range=[-1, 10]), showlegend=False)
        st.plotly_chart(style(fig, 440, "Violin: spread of district growth rates by province"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:  # histogram
        g = df["growth_rate"]
        fig = go.Figure(go.Histogram(x=g, xbins=dict(start=-1, end=10, size=.5), marker=dict(color="#c026d3", line=dict(color="#fff", width=1.5)),
                                     hovertemplate="%{x}%: %{y} districts<extra></extra>"))
        fig.add_vline(x=g.mean(), line_dash="dash", line_color="#f97316",
                      annotation_text=f"mean {g.mean():.2f}% · median {g.median():.2f}%", annotation_position="top right")
        fig.update_layout(xaxis_ticksuffix="%", yaxis_title="Districts", bargap=.04, showlegend=False)
        st.plotly_chart(style(fig, 380, "Distribution of growth rates"), width="stretch")
    with c2:  # 100% age bars
        fig = go.Figure()
        for k, lab, col in [("age_0_14_pct", "0-14", "#f97316"), ("age_15_64_pct", "15-64", "#7c3aed"), ("age_65_plus_pct", "65+", "#14b8a6")]:
            fig.add_bar(y=[short(x) for x in provs], x=p[k], name=lab, orientation="h", marker=dict(color=col, line=dict(color="#fff", width=1.5)),
                        text=[f"{v:.0f}%" for v in p[k]], textposition="inside", textfont=dict(color="#fff"),
                        hovertemplate="%{y} " + lab + ": %{x:.1f}%<extra></extra>")
        fig.update_layout(barmode="stack", xaxis=dict(ticksuffix="%", range=[0, 100]), yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(fig, 380, "Age structure (100%)"), width="stretch")

with tab3:
    c1, c2 = st.columns(2)
    vars_ = [("growth_rate", "Growth"), ("urban_pct", "Urban %"), ("density", "Density (log)"), ("literacy_rate", "Literacy"),
             ("literacy_female", "Female lit."), ("avg_hh_size", "HH size"), ("age_0_14_pct", "Youth 0-14"),
             ("females_per_100_males", "F/100 M")]
    X = df[[v for v, _ in vars_]].copy()
    X["density"] = np.log10(X["density"])
    corr = X.corr().values
    with c1:
        labs = [l for _, l in vars_]
        fig = go.Figure(go.Heatmap(z=corr, x=labs, y=labs, zmin=-1, zmax=1, xgap=2, ygap=2,
                                   colorscale=[[0, "#7c3aed"], [.25, "#c4b5fd"], [.5, "#f8fafc"], [.75, "#fdba74"], [1, "#ea580c"]],
                                   text=np.round(corr, 2), texttemplate="%{text}", hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-35))
        st.plotly_chart(style(fig, 480, "Correlation matrix across districts (Pearson r)"), width="stretch")
    with c2:
        x, y = df["age_0_14_pct"], df["literacy_rate"]
        b1, b0 = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        fig = go.Figure()
        for pr in provs:
            s = df[df["province"] == pr]
            fig.add_scatter(x=s["age_0_14_pct"], y=s["literacy_rate"], mode="markers", name=short(pr), text=s["district"],
                            marker=dict(size=7 + np.sqrt(s["pop_2023"]) / 260, color=PCOLORS[pr], opacity=.85, line=dict(color="#fff", width=1)),
                            hovertemplate="<b>%{text}</b><br>Youth %{x:.1f}% · Literacy %{y:.1f}%<extra></extra>")
        xr = np.array([x.min(), x.max()])
        fig.add_scatter(x=xr, y=b0 + b1 * xr, mode="lines", name="OLS fit", line=dict(color="#0f172a", dash="dash", width=2))
        fig.update_layout(xaxis_title="Children 0-14 (%)", yaxis_title="Literacy (%)")
        st.plotly_chart(style(fig, 480, "Regression: more children ↔ lower literacy"), width="stretch")
        insight(f"Each extra percentage point of children is associated with <b>{b1:.2f}</b> points of literacy "
                f"(R² = {r2:.2f}, n = {len(df)}). Correlation, not causation.")
    c1, c2 = st.columns(2)
    with c1:  # Lorenz
        pops = np.sort(df["pop_2023"].values)
        cum = np.concatenate([[0], np.cumsum(pops) / pops.sum() * 100])
        xs = np.linspace(0, 100, len(cum))
        trap = getattr(np, "trapezoid", None) or np.trapz
        gini = 1 - 2 * trap(cum, xs) / 10000
        fig = go.Figure()
        fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color="#cbd5e1", dash="dot"), name="Equality", hoverinfo="skip")
        fig.add_scatter(x=xs, y=cum, mode="lines", fill="tonexty", fillcolor="rgba(124,58,237,.15)", line=dict(color="#7c3aed", width=3),
                        name="Districts", hovertemplate="Smallest %{x:.0f}% of districts hold %{y:.1f}% of people<extra></extra>")
        fig.update_layout(xaxis=dict(title="% of districts (smallest first)", range=[0, 100]),
                          yaxis=dict(title="% of population", range=[0, 100]), showlegend=False)
        st.plotly_chart(style(fig, 400, f"Lorenz curve of district populations: Gini = {gini:.2f}"), width="stretch")
    with c2:  # density vs growth bubble
        fig = go.Figure()
        for pr in provs:
            s = df[df["province"] == pr]
            fig.add_scatter(x=s["density"], y=s["growth_rate"], mode="markers", name=short(pr), text=s["district"],
                            marker=dict(size=6 + np.sqrt(s["pop_2023"]) / 220, color=PCOLORS[pr], opacity=.8, line=dict(color="#fff", width=1)),
                            hovertemplate="<b>%{text}</b><br>Density %{x:,.0f}/km²<br>Growth %{y:.2f}%<extra></extra>")
        fig.update_layout(xaxis=dict(type="log", title="Density (people/km², log)"), yaxis=dict(title="Growth %/yr", ticksuffix="%"))
        st.plotly_chart(style(fig, 400, "Density vs growth: emptier districts grew faster"), width="stretch")
