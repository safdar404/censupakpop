"""Builds data/processed/pakistan_districts_2023.csv from the raw PBS tables
in data/raw using the same Module-1 loader the app uses.

    python scripts/build_dataset.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from core.data_loader import load_census, quality_summary  # noqa: E402

raw = ROOT / "data" / "raw"
files = sorted(p for p in raw.glob("table_*.csv"))
div = pd.read_csv(raw / "district_divisions.csv")
df, rep = load_census(files, divisions=div)
out = ROOT / "data" / "processed" / "pakistan_districts_2023.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} districts -> {out}")
print(quality_summary(df))
for w in rep.warnings:
    print("WARN", w)
