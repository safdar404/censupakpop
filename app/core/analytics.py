"""
Analytics engine for Modules 2-5 plus the AI "smart analytics" layer
(anomaly detection and district typology clustering).
All functions are pure: dataframe in -> dataframe / dict out, so the same
logic is reused by the UI, the AI agent and the PDF report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HIGH_GROWTH, LOW_GROWTH
from .data_loader import PROVINCE_ORDER


# ------------------------------------------------------------- formatting
def fmt_pop(n: float) -> str:
    """241499431 -> '241.50 M'  (human readable for presentations)."""
    if n is None or pd.isna(n):
        return "n/a"
    n = float(n)
    if abs(n) >= 1e6:
        return f"{n / 1e6:,.2f} M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:,.1f} K"
    return f"{n:,.0f}"


def cagr(p_new, p_old, years=6.0):
    return ((p_new / p_old) ** (1 / years) - 1) * 100


# ------------------------------------------------------------- Module 2
def national_summary(df: pd.DataFrame) -> dict:
    pop = df["pop_2023"].sum()
    urban = df["urban_pop"].sum()
    return {
        "population": int(pop),
        "population_2017": int(df["pop_2017"].sum()),
        "households": int(df["households"].sum()),
        "urban_pct": urban / pop * 100,
        "rural_pct": 100 - urban / pop * 100,
        "male": int(df["male"].sum()),
        "female": int(df["female"].sum()),
        "transgender": int(df["transgender"].sum()),
        "females_per_100_males": df["female"].sum() / df["male"].sum() * 100,
        "growth_rate": cagr(pop, df["pop_2017"].sum()),
        "area_sqkm": float(df["area_sqkm"].sum()),
        "density": pop / df["area_sqkm"].sum(),
        "avg_hh_size": pop / df["households"].sum(),
        "literacy_rate": _wavg(df, "literacy_rate"),
        "districts": int(len(df)),
        "provinces": int(df["province"].nunique()),
    }


def _wavg(df, col, w="pop_2023"):
    d = df.dropna(subset=[col])
    return float(np.average(d[col], weights=d[w])) if len(d) else np.nan


def province_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("province", observed=True)
    out = pd.DataFrame({
        "districts": g.size(),
        "pop_2023": g["pop_2023"].sum(),
        "pop_2017": g["pop_2017"].sum(),
        "male": g["male"].sum(),
        "female": g["female"].sum(),
        "transgender": g["transgender"].sum(),
        "urban_pop": g["urban_pop"].sum(),
        "rural_pop": g["rural_pop"].sum(),
        "urban_pop_2017": g["urban_pop_2017"].sum(),
        "rural_pop_2017": g["rural_pop_2017"].sum(),
        "households": g["households"].sum(),
        "area_sqkm": g["area_sqkm"].sum(),
    })
    out["share_pct"] = out["pop_2023"] / out["pop_2023"].sum() * 100
    out["urban_pct"] = out["urban_pop"] / out["pop_2023"] * 100
    out["rural_pct"] = 100 - out["urban_pct"]
    out["growth_rate"] = cagr(out["pop_2023"], out["pop_2017"])
    out["urban_growth_rate"] = cagr(out["urban_pop"], out["urban_pop_2017"])
    out["rural_growth_rate"] = cagr(out["rural_pop"], out["rural_pop_2017"])
    out["density"] = out["pop_2023"] / out["area_sqkm"]
    out["avg_hh_size"] = out["pop_2023"] / out["households"]
    out["females_per_100_males"] = out["female"] / out["male"] * 100
    for c in ["literacy_rate", "literacy_male", "literacy_female", "age_0_14_pct", "age_15_64_pct", "age_65_plus_pct"]:
        out[c] = [ _wavg(df[df.province == p], c) for p in out.index]
    order = [p for p in PROVINCE_ORDER if p in out.index]
    return out.loc[order]


# ------------------------------------------------------------- generic ranks
def top_bottom(df: pd.DataFrame, col: str, n: int = 10, ascending=False, cols=None) -> pd.DataFrame:
    cols = cols or ["district", "province", "pop_2023", col]
    cols = list(dict.fromkeys(cols))
    r = df.dropna(subset=[col]).sort_values(col, ascending=ascending).head(n)[cols].reset_index(drop=True)
    r.index = r.index + 1
    return r


# ------------------------------------------------------------- Module 4
def growth_zones(df: pd.DataFrame, high=HIGH_GROWTH, low=LOW_GROWTH):
    cols = ["district", "province", "pop_2017", "pop_2023", "growth_rate"]
    hi = df[df["growth_rate"] > high].sort_values("growth_rate", ascending=False)[cols]
    lo = df[df["growth_rate"] < low].sort_values("growth_rate")[cols]
    return hi.reset_index(drop=True), lo.reset_index(drop=True)


# ------------------------------------------------------------- Module 5
def gender_outliers(df: pd.DataFrame, z=1.5) -> pd.DataFrame:
    s = df["females_per_100_males"]
    zs = (s - s.mean()) / s.std()
    out = df.loc[zs.abs() > z, ["district", "province", "male", "female", "females_per_100_males"]].copy()
    out["status"] = np.where(zs[zs.abs() > z] > 0, "High (more females)", "Low (fewer females)")
    return out.sort_values("females_per_100_males").reset_index(drop=True)


# ------------------------------------------------------------- AI layer
ANOMALY_FEATURES = ["growth_rate", "females_per_100_males", "avg_hh_size", "urban_pct",
                    "urbanisation_change_pp", "age_0_14_pct", "literacy_rate"]


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.08, seed: int = 42) -> pd.DataFrame:
    """Isolation Forest: flags districts whose combination of indicators is
    statistically unusual (possible over/under-count, data entry issues or
    genuinely exceptional places).  Returns df with anomaly_score & reason."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    feats = [f for f in ANOMALY_FEATURES if f in df.columns]
    X = df[feats].fillna(df[feats].median())
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(Xs)
    out = df[["district", "province", "pop_2023"] + feats].copy()
    out["anomaly_score"] = -iso.score_samples(Xs)
    out["is_anomaly"] = iso.predict(Xs) == -1
    z = pd.DataFrame(Xs, columns=feats, index=df.index)
    reasons = []
    for i in df.index:
        top = z.loc[i].abs().sort_values(ascending=False).head(2)
        reasons.append("; ".join(f"{k.replace('_', ' ')} {'high' if z.loc[i, k] > 0 else 'low'} "
                                 f"({df.loc[i, k]:.1f})" for k in top.index))
    out["why"] = reasons
    return out.sort_values("anomaly_score", ascending=False)


CLUSTER_FEATURES = ["urban_pct", "density", "growth_rate", "avg_hh_size", "literacy_rate",
                    "age_0_14_pct", "females_per_100_males"]


def cluster_districts(df: pd.DataFrame, k: int = 4, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-Means district typologies with automatic, human-readable names."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = df[CLUSTER_FEATURES].copy()
    X["density"] = np.log10(X["density"].clip(lower=1))
    X = X.fillna(X.median())
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Xs)
    lab = df[["district", "province", "pop_2023"] + CLUSTER_FEATURES].copy()
    lab["cluster"] = km.labels_
    prof = lab.groupby("cluster")[CLUSTER_FEATURES + ["pop_2023"]].agg(
        {**{c: "mean" for c in CLUSTER_FEATURES}, "pop_2023": "sum"})
    prof["districts"] = lab.groupby("cluster").size()
    # name clusters by their relative profile (rank based -> always unique)
    names, left = {}, list(prof.index)
    pick = lambda col: max(left, key=lambda c: prof.loc[c, col])  # noqa: E731
    for col, label in [("urban_pct", "Metropolitan hubs"), ("literacy_rate", "Urbanising, higher-literacy"),
                       ("growth_rate", "Fast-growing, rural & young"), ("avg_hh_size", "Rural heartland")]:
        if not left:
            break
        c = pick(col)
        names[c] = label
        left.remove(c)
    for i, c in enumerate(left):
        names[c] = f"Mixed profile {i + 1}"
    lab["typology"] = lab["cluster"].map(names)
    prof.index = prof.index.map(names)
    return lab, prof


# ------------------------------------------------------------- narrative
def auto_insights(df: pd.DataFrame) -> list[str]:
    """Rule-based insights (used offline and as grounding facts for the LLM)."""
    n = national_summary(df)
    p = province_summary(df)
    big = p["pop_2023"].idxmax()
    fast = p["growth_rate"].idxmax()
    urb = df.sort_values("urbanisation_change_pp", ascending=False).iloc[0]
    dens = df.sort_values("density", ascending=False).iloc[0]
    hi, lo = growth_zones(df)
    lit_gap = p["literacy_male"] - p["literacy_female"]
    return [
        f"Pakistan's 2023 population is {fmt_pop(n['population'])}, up from {fmt_pop(n['population_2017'])} "
        f"in 2017 - an average growth of {n['growth_rate']:.2f}% per year.",
        f"{big} holds {p.loc[big, 'share_pct']:.1f}% of the national population; "
        f"{fast} grew fastest at {p.loc[fast, 'growth_rate']:.2f}% per year.",
        f"{n['urban_pct']:.1f}% of people live in urban areas. Urban population grew "
        f"{cagr(df['urban_pop'].sum(), df['urban_pop_2017'].sum()):.2f}%/yr vs "
        f"{cagr(df['rural_pop'].sum(), df['rural_pop_2017'].sum()):.2f}%/yr for rural areas.",
        f"{urb['district']} ({urb['province']}) urbanised the most: +{urb['urbanisation_change_pp']:.1f} "
        f"percentage points between 2017 and 2023.",
        f"{dens['district']} is the most densely populated district at {dens['density']:,.0f} people per km².",
        f"{len(hi)} districts are 'high growth zones' (>3%/yr); {len(lo)} grew below 1%/yr.",
        f"There are {n['females_per_100_males']:.1f} females per 100 males nationally; the largest male-female "
        f"literacy gap is in {lit_gap.idxmax()} ({lit_gap.max():.1f} points).",
    ]
