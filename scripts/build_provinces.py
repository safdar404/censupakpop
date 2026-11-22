"""Dissolves district polygons into province polygons (for the globe & infographics).

    python scripts/build_provinces.py
Former FATA districts are merged into Khyber Pakhtunkhwa (25th Amendment, 2018).
AJK and Gilgit-Baltistan are kept as separate regions (not part of the PBS census tables).
"""
import json
from pathlib import Path

from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "geo" / "source_gadm_pak_adm3.geojson"
OUT = ROOT / "data" / "geo" / "pakistan_provinces.geojson"
MAP = {"Punjab": "Punjab", "Sindh": "Sindh", "Balochistan": "Balochistan",
       "Khyber-Pakhtunkhwa": "Khyber Pakhtunkhwa", "Federally Administered Tribal Ar": "Khyber Pakhtunkhwa",
       "Islamabad": "Islamabad (ICT)", "Azad Kashmir": "Azad Jammu & Kashmir", "Gilgit-Baltistan": "Gilgit-Baltistan"}
CENSUS = {"Punjab", "Sindh", "Balochistan", "Khyber Pakhtunkhwa", "Islamabad (ICT)"}



def to_shape(geom):
    """Build polygons, skipping degenerate rings (< 4 points) found in the source file."""
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    out = []
    for rings in polys:
        if len(rings[0]) < 4:
            continue
        out.append(Polygon(rings[0], [r for r in rings[1:] if len(r) >= 4]).buffer(0))
    return MultiPolygon([q for o in out for q in (o.geoms if hasattr(o, "geoms") else [o])])


g = json.loads(SRC.read_text())
groups: dict[str, list] = {}
for f in g["features"]:
    groups.setdefault(MAP[f["properties"]["NAME_1"]], []).append(to_shape(f["geometry"]))
feats = []
for name, geoms in groups.items():
    u = unary_union(geoms).buffer(0.01).buffer(-0.01).simplify(0.015, preserve_topology=True)
    c = u.representative_point()
    feats.append({"type": "Feature", "properties": {"province": name, "in_census": name in CENSUS,
                                                    "label_lon": round(c.x, 3), "label_lat": round(c.y, 3)},
                  "geometry": json.loads(json.dumps(mapping(u)), parse_float=lambda x: round(float(x), 3))})
OUT.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
print(f"{len(feats)} regions -> {OUT} ({OUT.stat().st_size // 1024} KB)")
