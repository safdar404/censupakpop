"""Links the district boundary GeoJSON to census-2023 district names.

    python scripts/build_geo.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from core.config import PROCESSED_CSV  # noqa: E402
from core.geo import prepare_boundaries  # noqa: E402

df = pd.read_csv(PROCESSED_CSV)
print(prepare_boundaries(ROOT / "data" / "geo" / "source_gadm_pak_adm3.geojson", df["district"].tolist()))
