"""Trains & saves the best growth model (Module 7) from the command line.

    python scripts/train_model.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from core import ml  # noqa: E402
from core.config import PROCESSED_CSV  # noqa: E402

res = ml.train_and_compare(pd.read_csv(PROCESSED_CSV))
print(res.leaderboard.round(3).to_string(index=False))
print(f"\nBest model: {res.best_name} -> {ml.MODEL_PATH}")
