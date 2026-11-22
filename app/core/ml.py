"""
Module 7 - Population Growth Prediction (Machine Learning)
==========================================================
Problem framing (important for the viva):
  We have two census snapshots (2017 and 2023) for 136 districts.  We train
  regressors that learn the district's ANNUAL GROWTH RATE from its structural
  characteristics, then convert the predicted rate into a population:

        population(year) = population(base) * (1 + rate/100) ** (year - base)

  Why predict the rate and not the raw population?  Raw population is almost
  perfectly explained by the 2017 population (R² ~ 0.99 - a trivial model).
  Predicting the *rate* forces the model to learn real demographic drivers
  (urbanisation, household size, literacy, youth share ...).  We still report
  accuracy on the population scale, as the proposal requires.

  Training uses 2017 baseline features -> 2017-23 growth.  For projections we
  feed the 2023 values of the same features -> expected 2023-2035 growth.

Models compared: Linear Regression, Ridge, Random Forest, Gradient Boosting.
Validation: 80/20 train-test split + 5-fold cross-validation.
Metrics: MAE and R² (on growth rate AND on 2023 population).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR
from .data_loader import PROVINCE_ORDER

MODEL_PATH = MODEL_DIR / "best_growth_model.joblib"
RESULT_PATH = MODEL_DIR / "training_report.joblib"

FEATURES = ["log_population", "urban_pct", "avg_hh_size", "log_area", "log_density",
            "province_code", "literacy_rate", "age_0_14_pct"]
FEATURE_LABELS = {
    "log_population": "Current population (log)", "urban_pct": "Urban percentage",
    "avg_hh_size": "Household size", "log_area": "Area km² (log)",
    "log_density": "Density (log)", "province_code": "Province (encoded)",
    "literacy_rate": "Literacy rate", "age_0_14_pct": "Youth share (0-14)",
}
TARGET = "growth_rate"


def make_features(df: pd.DataFrame, base: str = "2017") -> pd.DataFrame:
    """Feature matrix at a baseline year ('2017' for training, '2023' for projecting)."""
    if base == "2017":
        pop = df["pop_2017"].astype(float)
        urban = df.get("urban_pct_2017", df["urban_pct"])
    else:
        pop = df["pop_2023"].astype(float)
        urban = df["urban_pct"]
    X = pd.DataFrame(index=df.index)
    X["log_population"] = np.log10(pop.clip(lower=1))
    X["urban_pct"] = urban
    X["avg_hh_size"] = df["avg_hh_size"]
    X["log_area"] = np.log10(df["area_sqkm"].clip(lower=1))
    X["log_density"] = np.log10((pop / df["area_sqkm"]).clip(lower=0.1))
    X["province_code"] = df["province"].map({p: i for i, p in enumerate(PROVINCE_ORDER)}).fillna(-1)
    X["literacy_rate"] = df["literacy_rate"]
    X["age_0_14_pct"] = df["age_0_14_pct"]
    return X.fillna(X.median(numeric_only=True))


def candidate_models(seed=42) -> dict:
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                       subsample=0.8, random_state=seed),
    }


@dataclass
class TrainResult:
    leaderboard: pd.DataFrame
    best_name: str
    test_frame: pd.DataFrame          # district, actual vs predicted
    importances: pd.DataFrame         # Random Forest feature importance
    permutation: pd.DataFrame         # model-agnostic importance for best model
    models: dict


def train_and_compare(df: pd.DataFrame, test_size=0.2, seed=42, clip_outliers=True) -> TrainResult:
    data = df.dropna(subset=[TARGET, "pop_2017", "pop_2023"]).copy()
    if clip_outliers:  # a few districts have extreme rates (boundary changes) - winsorise
        lo, hi = data[TARGET].quantile([0.02, 0.98])
        data["_y"] = data[TARGET].clip(lo, hi)
    else:
        data["_y"] = data[TARGET]
    X = make_features(data, "2017")
    y = data["_y"]
    idx_tr, idx_te = train_test_split(data.index, test_size=test_size, random_state=seed)
    kf = KFold(5, shuffle=True, random_state=seed)
    rows, models, preds = [], {}, {}
    for name, model in candidate_models(seed).items():
        cv = -cross_val_score(model, X.loc[idx_tr], y.loc[idx_tr], cv=kf, scoring="neg_mean_absolute_error")
        model.fit(X.loc[idx_tr], y.loc[idx_tr])
        p = model.predict(X.loc[idx_te])
        actual_rate = data.loc[idx_te, TARGET]
        pop_pred = data.loc[idx_te, "pop_2017"] * (1 + p / 100) ** 6
        pop_act = data.loc[idx_te, "pop_2023"]
        rows.append({
            "Model": name,
            "CV MAE (rate, pp)": cv.mean(),
            "Test MAE (rate, pp)": mean_absolute_error(actual_rate, p),
            "Test R² (rate)": r2_score(actual_rate, p),
            "Test MAE (population)": mean_absolute_error(pop_act, pop_pred),
            "Test R² (population)": r2_score(pop_act, pop_pred),
            "Test MAPE (population %)": float(np.mean(np.abs(pop_pred - pop_act) / pop_act) * 100),
        })
        models[name] = model
        preds[name] = p
    lb = pd.DataFrame(rows).sort_values("CV MAE (rate, pp)").reset_index(drop=True)
    best = lb.iloc[0]["Model"]
    # naive baseline: everyone grows at the training-set mean rate
    base_p = np.full(len(idx_te), y.loc[idx_tr].mean())
    lb.attrs["baseline_mae_rate"] = mean_absolute_error(data.loc[idx_te, TARGET], base_p)
    tf = data.loc[idx_te, ["district", "province", "pop_2017", "pop_2023", TARGET]].copy()
    tf["pred_rate"] = preds[best]
    tf["pred_pop_2023"] = tf["pop_2017"] * (1 + tf["pred_rate"] / 100) ** 6
    rf = models["Random Forest"]
    imp = pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES],
                        "importance": rf.feature_importances_}).sort_values("importance")
    pi = permutation_importance(models[best], X.loc[idx_te], y.loc[idx_te], n_repeats=8, random_state=seed)
    perm = pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES],
                         "importance": pi.importances_mean}).sort_values("importance")
    # refit best on ALL data for deployment
    final = candidate_models(seed)[best].fit(X, y)
    save_model(final, best, lb)
    res = TrainResult(lb, best, tf, imp, perm, {})
    joblib.dump({"key": data_key(df, test_size, seed), "result": res}, RESULT_PATH)
    return res


def data_key(df: pd.DataFrame, test_size=0.2, seed=42) -> str:
    import hashlib
    h = hashlib.md5(pd.util.hash_pandas_object(df[["district", "pop_2017", "pop_2023"]], index=False).values).hexdigest()
    return f"{h}-{test_size}-{seed}"


def cached_result(df: pd.DataFrame, test_size=0.2, seed=42) -> TrainResult | None:
    """Re-use the last training run for the same data (instant page load)."""
    try:
        blob = joblib.load(RESULT_PATH)
        return blob["result"] if blob["key"] == data_key(df, test_size, seed) else None
    except Exception:  # noqa: BLE001
        return None


def save_model(model, name: str, leaderboard: pd.DataFrame, path=MODEL_PATH):
    joblib.dump({"model": model, "name": name, "features": FEATURES,
                 "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
                 "leaderboard": leaderboard}, path)


def load_model(path=MODEL_PATH) -> dict | None:
    try:
        return joblib.load(path)
    except Exception:  # noqa: BLE001
        return None


def ensure_model(df: pd.DataFrame) -> dict:
    bundle = load_model()
    if bundle is None:
        train_and_compare(df)
        bundle = load_model()
    return bundle


def predicted_rates(df: pd.DataFrame, bundle: dict) -> pd.Series:
    """Model-expected annual growth 2023-> for every district."""
    X = make_features(df, "2023")[bundle["features"]]
    return pd.Series(bundle["model"].predict(X), index=df.index)


def project(pop_base: float, rate: float, years=(2030, 2035), base_year=2023) -> dict:
    return {y: float(pop_base * (1 + rate / 100) ** (y - base_year)) for y in years}


def project_district(df: pd.DataFrame, district: str, bundle: dict, rate: float | None = None,
                     years=(2030, 2035), method: str = "blend") -> dict:
    row = df[df["district"].str.lower() == district.lower()]
    if row.empty:
        raise KeyError(district)
    r = row.iloc[0]
    model_rate = float(predicted_rates(row, bundle).iloc[0])
    trend_rate = float(r["growth_rate"])
    if rate is None:
        rate = {"model": model_rate, "trend": trend_rate}.get(method, 0.5 * model_rate + 0.5 * trend_rate)
    proj = project(r["pop_2023"], rate, years)
    return {"district": r["district"], "province": r["province"], "pop_2017": int(r["pop_2017"]),
            "pop_2023": int(r["pop_2023"]), "trend_rate": trend_rate, "model_rate": model_rate,
            "rate_used": float(rate), "projections": proj, "households_2023": int(r["households"]),
            "avg_hh_size": float(r["avg_hh_size"])}


def insight_sentence(p: dict) -> str:
    y_last = max(p["projections"])
    pop = p["projections"][y_last]
    add = pop - p["pop_2023"]
    hh_new = add / p["avg_hh_size"]
    level = ("significant" if p["rate_used"] >= 2.5 else "steady" if p["rate_used"] >= 1.5 else "modest")
    return (f"At a growth rate of {p['rate_used']:.2f}% per year, District {p['district']} is projected to reach "
            f"{pop / 1e6:,.2f} million people by {y_last} (+{add / 1e6:,.2f} M from 2023). That implies roughly "
            f"{hh_new:,.0f} additional households, which will require {level} expansion of housing, schools, "
            f"health facilities and water-supply infrastructure.")
