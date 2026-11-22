"""
Module 9 - Automated PDF report (FPDF2 + Matplotlib)
====================================================
One call -> a professional, multi-section PDF:
  cover header (project, student, supervisor, date) - dataset introduction -
  AI executive summary - national & provincial tables - density top/bottom 10 -
  urban vs rural - gender & literacy - ML accuracy - district projections -
  every chart/map the user created during the session.
"""
from __future__ import annotations

import datetime as dt
import io
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402

from . import analytics as A  # noqa: E402
from .config import APP_NAME, C_PRIMARY, C_SECOND, PROGRAM, PROVINCE_COLORS  # noqa: E402
from .data_loader import PROVINCE_SHORT  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b8b7b1", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.titleweight": "bold", "axes.titlesize": 10,
                     "axes.titlelocation": "left", "axes.grid": True, "grid.color": "#ecebe7",
                     "grid.linewidth": 0.6, "axes.axisbelow": True})


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _safe(t: str) -> str:
    rep = {"–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...", "→": "->", "≈": "~", "•": "-"}
    for a, b in rep.items():
        t = t.replace(a, b)
    t = t.replace("**", "")
    return t.encode("latin-1", "replace").decode("latin-1")


# ------------------------------------------------------------------ charts
def chart_province_population(p):
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    s = p["pop_2023"].iloc[::-1] / 1e6
    ax.barh([PROVINCE_SHORT[i] for i in s.index], s.values, color=[PROVINCE_COLORS[i] for i in s.index], height=0.6)
    for i, v in enumerate(s.values):
        ax.text(v + 1, i, f"{v:.1f} M", va="center", fontsize=8, color="#0b0b0b")
    ax.set_title("Population by province, 2023 (millions)")
    ax.grid(axis="y", visible=False)
    return _png(fig)


def chart_province_share(p):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.pie(p["pop_2023"], labels=[f"{PROVINCE_SHORT[i]}\n{v:.1f}%" for i, v in p["share_pct"].items()],
           colors=[PROVINCE_COLORS[i] for i in p.index], startangle=90, counterclock=False,
           wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2}, textprops={"fontsize": 8})
    ax.set_title("Share of national population")
    return _png(fig)


def chart_urban_rural(p):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = [PROVINCE_SHORT[i] for i in p.index]
    ax.bar(x, p["urban_pop"] / 1e6, color=C_PRIMARY, label="Urban", width=0.6)
    ax.bar(x, p["rural_pop"] / 1e6, bottom=p["urban_pop"] / 1e6, color=C_SECOND, label="Rural", width=0.6,
           edgecolor="white", linewidth=1.5)
    ax.set_title("Urban vs rural population by province (millions)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    return _png(fig)


def chart_growth_ur(p):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(p))
    ax.bar(x - 0.18, p["urban_growth_rate"], 0.34, color=C_PRIMARY, label="Urban growth")
    ax.bar(x + 0.18, p["rural_growth_rate"], 0.34, color=C_SECOND, label="Rural growth")
    ax.set_xticks(x, [PROVINCE_SHORT[i] for i in p.index])
    ax.set_title("Annual growth 2017-2023: urban vs rural (%)")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    return _png(fig)


def chart_density(df):
    t = df.sort_values("density", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(t["district"], t["density"], color=[PROVINCE_COLORS[p] for p in t["province"]], height=0.65)
    ax.set_title("15 most densely populated districts (people per km²)")
    ax.grid(axis="y", visible=False)
    return _png(fig)


def chart_gender(p, nat):
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    s = p["females_per_100_males"]
    ax.bar([PROVINCE_SHORT[i] for i in s.index], s.values, color=C_PRIMARY, width=0.55)
    ax.axhline(nat, color="#52514e", lw=1, ls="--")
    ax.text(len(s) - 0.5, nat + 0.3, f"National {nat:.1f}", ha="right", fontsize=8, color="#52514e")
    ax.set_ylim(80, 100)
    ax.set_title("Females per 100 males")
    ax.grid(axis="x", visible=False)
    return _png(fig)


def chart_literacy(p):
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    x = np.arange(len(p))
    ax.bar(x - 0.18, p["literacy_male"], 0.34, color=C_PRIMARY, label="Male")
    ax.bar(x + 0.18, p["literacy_female"], 0.34, color=C_SECOND, label="Female")
    ax.set_xticks(x, [PROVINCE_SHORT[i] for i in p.index])
    ax.set_title("Literacy rate (10+) by sex (%)")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    return _png(fig)


def chart_ml(test_frame):
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    a, p = test_frame["pop_2023"] / 1e6, test_frame["pred_pop_2023"] / 1e6
    ax.scatter(a, p, s=26, color=C_PRIMARY, edgecolor="white", linewidth=0.8)
    m = max(a.max(), p.max()) * 1.05
    ax.plot([0, m], [0, m], color="#9a9993", lw=1, ls="--")
    ax.set_xlabel("Actual 2023 population (M)")
    ax.set_ylabel("Predicted (M)")
    ax.set_title("Predicted vs actual - test set")
    return _png(fig)


def chart_projections(projs):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    years = ["2017", "2023", "2030", "2035"]
    w = 0.8 / len(projs)
    colours = ["#7c3aed", "#f97316", "#14b8a6", "#eda100"]
    for i, pr in enumerate(projs):
        vals = [pr["pop_2017"], pr["pop_2023"], pr["projections"][2030], pr["projections"][2035]]
        ax.bar(np.arange(4) + (i - (len(projs) - 1) / 2) * w, np.array(vals) / 1e6, w * 0.92,
               color=colours[i % 4], label=pr["district"])
    ax.set_xticks(np.arange(4), years)
    ax.set_title("District population: census & projections (millions)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    return _png(fig)


# ------------------------------------------------------------------ PDF
class ReportPDF(FPDF):
    def __init__(self, student, supervisor):
        super().__init__(format="A4")
        self.student, self.supervisor = student, supervisor
        self.set_auto_page_break(True, 16)
        self.set_margins(16, 16, 16)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(110, 110, 105)
        self.cell(0, 6, _safe(f"{APP_NAME} - Census 2023 Analysis Report"), align="L")
        self.cell(0, 6, _safe(self.student), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(225, 224, 220)
        self.line(16, self.get_y(), 194, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(130, 130, 125)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}  |  Source: Pakistan Bureau of Statistics, 7th Population & Housing Census 2023", align="C")

    def h1(self, t):
        self.ln(2)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(13, 54, 107)
        self.cell(0, 9, _safe(t), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(42, 120, 214)
        self.set_line_width(0.6)
        self.line(16, self.get_y(), 40, self.get_y())
        self.set_line_width(0.2)
        self.ln(3)

    def para(self, t, size=10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.2, _safe(t), new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def bullets(self, items):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(30, 30, 30)
        for it in items:
            self.set_x(18)
            self.multi_cell(0, 5, _safe("- " + it), new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def table(self, df: pd.DataFrame, widths=None, font=8.5):
        cols = list(df.columns)
        widths = widths or [178 / len(cols)] * len(cols)
        if self.get_y() + 6.5 + 6 * min(len(df), 40) > self.h - 18:
            self.add_page()
        self.set_font("Helvetica", "B", font)
        self.set_fill_color(13, 54, 107)
        self.set_text_color(255, 255, 255)
        for c, w in zip(cols, widths):
            self.cell(w, 6.5, _safe(str(c)), border=0, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", font)
        self.set_text_color(30, 30, 30)
        for i, (_, r) in enumerate(df.iterrows()):
            self.set_fill_color(*((244, 247, 252) if i % 2 == 0 else (255, 255, 255)))
            for j, (c, w) in enumerate(zip(cols, widths)):
                v = str(r[c])
                numeric = bool(__import__("re").fullmatch(r"[\d,.\-%/ ]+", v))
                self.cell(w, 6, _safe(v), border=0, align="R" if numeric else "L", fill=True)
            self.ln()
        self.ln(3)

    def image_bytes(self, png: bytes, w=178):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png)
            path = f.name
        from PIL import Image
        iw, ih = Image.open(path).size
        h = w * ih / iw
        if self.get_y() + h > self.h - 18:
            self.add_page()
        self.image(path, x=(210 - w) / 2, w=w)
        self.ln(3)


def generate_report(df: pd.DataFrame, student: str, supervisor: str, ml_result=None,
                    projections: list | None = None, session_images: list | None = None,
                    executive_summary: str | None = None) -> bytes:
    nat = A.national_summary(df)
    p = A.province_summary(df)
    pdf = ReportPDF(student, supervisor)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ---------- cover band
    pdf.set_fill_color(13, 54, 107)
    pdf.rect(0, 0, 210, 62, "F")
    pdf.set_xy(16, 14)
    pdf.set_text_color(205, 226, 251)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _safe(PROGRAM), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "CensusInsight PK", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, "Pakistan Population Data Analysis System - Census 2023 Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(205, 226, 251)
    pdf.cell(0, 7, _safe(f"Student: {student}    |    Supervisor: {supervisor}    |    Generated: "
                         f"{dt.datetime.now():%d %B %Y, %H:%M}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(70)

    # ---------- KPI strip
    kpis = [("Population", A.fmt_pop(nat["population"])), ("Households", A.fmt_pop(nat["households"])),
            ("Urban", f"{nat['urban_pct']:.1f}%"), ("Growth / yr", f"{nat['growth_rate']:.2f}%")]
    for i, (k, v) in enumerate(kpis):
        x = 16 + i * 45
        pdf.set_fill_color(244, 247, 252)
        pdf.rect(x, 70, 42, 20, "F")
        pdf.set_xy(x + 3, 72)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(82, 81, 78)
        pdf.cell(36, 5, k)
        pdf.set_xy(x + 3, 78)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(13, 54, 107)
        pdf.cell(36, 8, v)
    pdf.set_y(96)

    pdf.h1("1. Dataset introduction")
    pdf.para("Pakistan's 7th Population and Housing Census (2023) was the country's first digital census, conducted by "
             "the Pakistan Bureau of Statistics (PBS). This report uses the official district-level PBS tables: Table 1 "
             "(area, population by sex, sex ratio, density, urban share, household size, 2017 population and growth rate), "
             "Table 5 (population by age group) and Table 12 (literacy). The data were cleaned and merged by the "
             f"CensusInsight PK pipeline into one dataset of {nat['districts']} districts across {nat['provinces']} "
             "provinces/territories (AJK and Gilgit-Baltistan are not part of the PBS census tables).")

    pdf.h1("2. Executive summary (AI-generated)")
    pdf.para(executive_summary or " ".join(A.auto_insights(df)))

    pdf.h1("3. National & provincial summary")
    nt = pd.DataFrame([
        ["Total population 2023", f"{nat['population']:,}"], ["Population 2017", f"{nat['population_2017']:,}"],
        ["Annual growth 2017-23", f"{nat['growth_rate']:.2f}%"], ["Households", f"{nat['households']:,}"],
        ["Average household size", f"{nat['avg_hh_size']:.2f}"], ["Urban / Rural", f"{nat['urban_pct']:.1f}% / {nat['rural_pct']:.1f}%"],
        ["Density (people per km²)", f"{nat['density']:.0f}"], ["Females per 100 males", f"{nat['females_per_100_males']:.1f}"],
        ["Literacy rate (10+)", f"{nat['literacy_rate']:.1f}%"], ["Districts / provinces", f"{nat['districts']} / {nat['provinces']}"],
    ], columns=["Indicator", "Value"])
    pdf.table(nt, [110, 68])
    pt = pd.DataFrame({
        "Province": [PROVINCE_SHORT[i] for i in p.index],
        "Population": [f"{v:,.0f}" for v in p["pop_2023"]],
        "Share": [f"{v:.1f}%" for v in p["share_pct"]],
        "Growth": [f"{v:.2f}%" for v in p["growth_rate"]],
        "Urban": [f"{v:.1f}%" for v in p["urban_pct"]],
        "HH size": [f"{v:.2f}" for v in p["avg_hh_size"]],
        "Literacy": [f"{v:.1f}%" for v in p["literacy_rate"]],
    })
    pdf.table(pt, [30, 34, 20, 22, 22, 22, 28])
    pdf.image_bytes(chart_province_population(p))
    pdf.image_bytes(chart_province_share(p), w=105)

    pdf.h1("4. Population density")
    for title, asc in [("Top 10 most densely populated districts", False), ("10 least densely populated districts", True)]:
        t = A.top_bottom(df, "density", 10, ascending=asc).reset_index()
        t = pd.DataFrame({"#": t["index"], "District": t["district"], "Province": t["province"].map(PROVINCE_SHORT),
                          "Population": t["pop_2023"].map("{:,}".format), "Density /km²": t["density"].map("{:,.1f}".format)})
        pdf.para(title, 10)
        pdf.table(t, [12, 56, 34, 38, 38])
    pdf.image_bytes(chart_density(df))

    pdf.h1("5. Urban vs rural")
    pdf.para(f"{nat['urban_pct']:.1f}% of Pakistanis live in urban areas. Urban population grew "
             f"{A.cagr(df['urban_pop'].sum(), df['urban_pop_2017'].sum()):.2f}% a year between 2017 and 2023, "
             f"against {A.cagr(df['rural_pop'].sum(), df['rural_pop_2017'].sum()):.2f}% for rural areas.")
    pdf.image_bytes(chart_urban_rural(p))
    pdf.image_bytes(chart_growth_ur(p))
    hi, lo = A.growth_zones(df)
    pdf.para(f"High growth zones (>3% per year): {len(hi)} districts - " + ", ".join(hi["district"].head(12)) + ".", 9.5)

    pdf.h1("6. Gender ratio & literacy")
    go = A.gender_outliers(df)
    pdf.bullets([f"National: {nat['females_per_100_males']:.1f} females per 100 males "
                 f"({nat['transgender']:,} transgender persons counted)."] +
                [f"{r.district} ({PROVINCE_SHORT.get(r.province, r.province)}): {r.females_per_100_males:.1f} - {r.status}"
                 for r in go.head(8).itertuples()])
    pdf.image_bytes(chart_gender(p, nat["females_per_100_males"]))
    pdf.image_bytes(chart_literacy(p))

    if ml_result is not None:
        pdf.h1("7. Machine learning - growth prediction")
        lb = ml_result.leaderboard
        pdf.para(f"Four regression models were trained to predict each district's annual growth rate from its 2017 "
                 f"characteristics (80/20 split, 5-fold cross-validation). Best model: {ml_result.best_name}. "
                 f"Naive baseline MAE (everyone grows at the mean): {lb.attrs.get('baseline_mae_rate', float('nan')):.2f} pp.")
        t = pd.DataFrame({"Model": lb["Model"], "CV MAE (pp)": lb["CV MAE (rate, pp)"].map("{:.3f}".format),
                          "R² rate": lb["Test R² (rate)"].map("{:.3f}".format),
                          "MAE pop.": lb["Test MAE (population)"].map("{:,.0f}".format),
                          "R² pop.": lb["Test R² (population)"].map("{:.4f}".format),
                          "MAPE": lb["Test MAPE (population %)"].map("{:.2f}%".format)})
        pdf.table(t, [44, 26, 22, 32, 26, 28])
        pdf.image_bytes(chart_ml(ml_result.test_frame), w=110)

    if projections:
        pdf.h1("8. District population projections")
        from .ml import insight_sentence
        for pr in projections:
            pdf.para(insight_sentence(pr), 9.5)
        pdf.image_bytes(chart_projections(projections))

    if session_images:
        pdf.add_page()
        pdf.h1("Appendix - charts & maps from this session")
        for title, png in session_images:
            pdf.para(title, 9.5)
            pdf.image_bytes(png, w=150)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(110, 110, 105)
    pdf.multi_cell(0, 4.5, _safe("Methodology note: growth rates are compound annual rates between census 2017 and 2023 "
                                 "as published by PBS. Projections assume a constant rate and are scenarios, not official forecasts."))
    return bytes(pdf.output())
