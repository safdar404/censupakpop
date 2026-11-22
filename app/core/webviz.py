"""
Embedded web visualisations (3D globe + geo-infographics)
=========================================================
The globe and infographics are built with D3.js (orthographic projection,
TopoJSON world map) in app/web/*.html.  This module injects the *current*
dataset into those templates so they always reflect the loaded census data.
"""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from . import analytics as A
from .config import DATA_DIR

WEB_DIR = DATA_DIR.parent / "app" / "web"
GEO_DIR = DATA_DIR / "geo"

DISTRICT_FIELDS = ["district", "province", "pop_2017", "pop_2023", "growth_rate", "urban_pct", "density",
                   "literacy_rate", "age_0_14_pct", "avg_hh_size"]


@lru_cache(maxsize=1)
def _static_geo() -> tuple[dict, dict, pd.DataFrame]:
    provinces = json.loads((GEO_DIR / "pakistan_provinces.geojson").read_text(encoding="utf-8"))
    world = json.loads((GEO_DIR / "world-110m.json").read_text(encoding="utf-8"))
    cent = pd.read_csv(GEO_DIR / "district_centroids.csv")
    return provinces, world, cent


def payload(df: pd.DataFrame) -> dict:
    provinces, world, cent = _static_geo()
    d = df[[c for c in DISTRICT_FIELDS if c in df.columns]].merge(cent, on="district", how="left")
    d = d.round(3).astype(object).where(d.notna(), None)
    p = A.province_summary(df).round(3).reset_index()
    nat = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in A.national_summary(df).items()}
    return {"districts": d.to_dict("records"),
            "meta": {"national": nat, "provinces": json.loads(p.to_json(orient="records")),
                     "insights": A.auto_insights(df)},
            "provinces": provinces, "world": world}


def render(template: str, df: pd.DataFrame) -> str:
    html = (WEB_DIR / template).read_text(encoding="utf-8")
    return html.replace("__DATA__", json.dumps(payload(df), separators=(",", ":"), default=str))
