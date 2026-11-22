"""
Module 6 - Geospatial helpers
=============================
* `prepare_boundaries` (build time) links district polygons to the 136
  census-2023 district names.  Boundaries come from the open GADM / pakdata
  GeoJSON; districts created after that boundary file (e.g. Keamari, Korangi,
  Upper/Lower Dir) are drawn as point markers at the parent polygon's centre.
* `build_map` (run time) renders an interactive Folium choropleth.
* `static_map_png` renders an exportable PNG of the same indicator with
  Matplotlib (no GeoPandas needed - pure GeoJSON parsing).
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GEOJSON, INDICATORS

# census name -> polygon name in the boundary file (when spelt differently)
DIRECT = {
    "Attock": "Attok", "Dera Ghazi Khan": "Dera Ghazi Kha", "Mandi Bahauddin": "Gujarat",
    "Gujranwala": ["Gujranwala 1", "Gujranwala 2"], "Narowal": ["Narowal 1", "Narowal 2"],
    "Okara": ["Okara", "Okara 1"], "Jacobabad": "Jakobabad", "Mirpur Khas": "Mirphurkhas",
    "Naushahro Feroze": "Naushahro Firoz", "Shaheed Benazirabad": "Nawab Shah",
    "Tando Ahyar": "Tando Allahyar", "Tando Muhammad Khan": "Tando M. Khan",
    "Tharparkar": "Mithi", "Batagram": "Battagram", "Malakand": "Malakand P.A.",
    "North Waziristan": "N. Waziristan", "South Waziristan": "S. Waziristan",
    "Jaffarabad": "Jafarabad", "Kachhi": "Bolan", "Killa Abdullah": "Qilla Abdullah",
    "Killa Saifullah": "Qilla Saifullah", "Kohlu": "Kholu", "Sherani": "Largha Shirani",
}
# new districts -> parent polygon (drawn as markers)
CHILD = {
    "Chiniot": "Jhang", "Kambar Shahdad Kot": "Larkana", "Keamari": "Karachi west",
    "Korangi": "Karachi East", "Sujawal": "Thatta", "Kolai Palas Kohistan": "Kohistan",
    "Lower Kohistan": "Kohistan", "Upper Kohistan": "Kohistan", "Torghar": "Mansehra",
    "Lower Chitral": "Chitral", "Upper Chitral": "Chitral", "Lower Dir": "Dir",
    "Upper Dir": "Dir", "Chaman": "Qilla Abdullah", "Duki": "Loralai", "Harnai": "Sibi",
    "Nushki": "Chagai", "Sohbatpur": "Jafarabad", "Surab": "Kalat", "Washuk": "Kharan",
}
EXCLUDE_REGIONS = {"Azad Kashmir", "Gilgit-Baltistan"}


def _centroid(geom) -> tuple[float, float]:
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    pts = np.array([pt for p in polys for pt in p[0]])
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def prepare_boundaries(src: Path, districts: list[str], out: Path = GEOJSON) -> dict:
    g = json.loads(Path(src).read_text())
    by_name = {}
    feats = []
    for f in g["features"]:
        p = f["properties"]
        if p["NAME_1"] in EXCLUDE_REGIONS:
            continue
        feats.append(f)
        by_name.setdefault(p["NAME_3"], []).append(f)
    norm = lambda s: re.sub(r"[^A-Z]", "", s.upper())  # noqa: E731
    lookup = {norm(k): k for k in by_name}
    for f in feats:
        f["properties"] = {"district": "", "boundary_name": f["properties"]["NAME_3"]}
    markers = {}
    matched = 0
    for d in districts:
        targets = DIRECT.get(d, d)
        targets = targets if isinstance(targets, list) else [targets]
        hit = False
        for t in targets:
            key = lookup.get(norm(t))
            if key:
                for f in by_name[key]:
                    f["properties"]["district"] = d
                hit = True
        if hit:
            matched += 1
        elif d in CHILD and CHILD[d] in by_name:
            markers[d] = _centroid(by_name[CHILD[d]][0]["geometry"])
    # offset sibling markers so they do not overlap
    seen: dict[str, int] = {}
    for d in markers:
        k = CHILD[d]
        i = seen.get(k, 0)
        seen[k] = i + 1
        x, y = markers[d]
        markers[d] = (round(x + 0.18 * np.cos(i * 2.1), 4), round(y + 0.18 * np.sin(i * 2.1), 4))
    for f in feats:
        if not f["properties"]["district"]:
            kids = [c for c, parent in CHILD.items() if parent == f["properties"]["boundary_name"]]
            f["properties"]["note"] = ("Split in census 2023 into: " + ", ".join(kids)) if kids else "No census match"
    for f in feats:  # reduce precision -> smaller file
        f["geometry"] = _round_geom(f["geometry"])
    fc = {"type": "FeatureCollection", "features": feats, "markers": markers}
    out.write_text(json.dumps(fc, separators=(",", ":")))
    return {"polygons_matched": matched, "markers": len(markers), "total": len(districts)}


def _round_geom(geom, nd=3):
    def r(c):
        return [r(x) for x in c] if isinstance(c[0], list) else [round(c[0], nd), round(c[1], nd)]
    geom["coordinates"] = r(geom["coordinates"])
    return geom


def load_boundaries(path: Path = GEOJSON) -> dict:
    return json.loads(Path(path).read_text())


def load_custom_boundaries(file, name_field: str | None = None) -> dict:
    """Accept any GeoJSON (e.g. HDX pak_admbnda_adm2) and tag its features
    with a `district` property using the chosen name field."""
    g = json.load(file)
    props = g["features"][0]["properties"]
    field = name_field or next((k for k in ("ADM2_EN", "district", "DISTRICT", "NAME_3", "name")
                                if k in props), list(props)[0])
    for f in g["features"]:
        f["properties"]["district"] = str(f["properties"].get(field, "")).title()
    g.setdefault("markers", {})
    return g


# ---------------------------------------------------------------------------
def _fmt(ind, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return INDICATORS[ind][1].format(v)


def build_map(df: pd.DataFrame, geo: dict, indicator: str = "pop_2023"):
    import folium
    import branca.colormap as cm

    label, _, _ = INDICATORS[indicator]
    vals = df.set_index("district")[indicator]
    rows = df.set_index("district")
    lo, hi = float(vals.quantile(0.02)), float(vals.quantile(0.98))
    if indicator == "females_per_100_males":
        cmap = cm.LinearColormap(["#c026d3", "#f1f5f9", "#2563eb"], vmin=85, vmax=105, caption=label)
    elif indicator in ("pop_2023", "density"):
        # skewed indicators -> quantile classes so the legend shows real numbers
        q = [float(vals.quantile(x)) for x in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        cmap = cm.StepColormap(["#fde047", "#fb923c", "#f43f5e", "#c026d3", "#5b21b6"], index=q,
                               vmin=q[0], vmax=q[-1], caption=label + " (quintiles)")
    else:
        cmap = cm.LinearColormap(["#fde047", "#fb923c", "#f43f5e", "#c026d3", "#5b21b6"], vmin=lo, vmax=hi, caption=label)

    def colour(v):
        if v is None or pd.isna(v):
            return "#d9d8d4"
        return cmap(min(max(v, cmap.vmin), cmap.vmax))

    m = folium.Map(location=[30.2, 69.4], zoom_start=5, tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[23.6, 60.9], [37.1, 77.8]])
    feats = []
    for f in geo["features"]:
        d = f["properties"].get("district", "")
        p = dict(f["properties"])
        if d and d in rows.index:
            r = rows.loc[d]
            p.update(province=r["province"], population=f"{int(r['pop_2023']):,}",
                     value=_fmt(indicator, r[indicator]), _v=None if pd.isna(r[indicator]) else float(r[indicator]))
        else:
            p.update(district=p.get("note", p.get("boundary_name", "")), province="", population="", value="", _v=None)
        feats.append({"type": "Feature", "geometry": f["geometry"], "properties": p})
    folium.GeoJson(
        {"type": "FeatureCollection", "features": feats},
        style_function=lambda f: {"fillColor": colour(f["properties"]["_v"]), "color": "#ffffff",
                                  "weight": 0.6, "fillOpacity": 0.85},
        highlight_function=lambda f: {"weight": 2, "color": "#0b0b0b"},
        tooltip=folium.GeoJsonTooltip(fields=["district", "province", "population", "value"],
                                      aliases=["District", "Province", "Population", label], sticky=True),
        name=label,
    ).add_to(m)
    for d, (x, y) in geo.get("markers", {}).items():
        if d not in rows.index:
            continue
        r = rows.loc[d]
        folium.CircleMarker(
            [y, x], radius=6, color="#ffffff", weight=2, fill=True, fill_opacity=0.95,
            fill_color=colour(r[indicator]),
            tooltip=(f"<b>{d}</b> (new district, shown as point)<br>Province: {r['province']}<br>"
                     f"Population: {int(r['pop_2023']):,}<br>{label}: {_fmt(indicator, r[indicator])}"),
        ).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html(cmap, label, indicator)))
    return m


def _legend_html(cmap, label, indicator) -> str:
    """Readable legend: class swatches with real values (replaces branca's crowded axis)."""
    fmt = INDICATORS[indicator][1]
    if hasattr(cmap, "index") and len(cmap.index) > 2 and cmap.__class__.__name__ == "StepColormap":
        edges = cmap.index
        rows = [(cmap.colors[i], f"{fmt.format(edges[i])} – {fmt.format(edges[i + 1])}") for i in range(len(edges) - 1)]
    else:
        vals = np.linspace(cmap.vmin, cmap.vmax, 5)
        rows = [(cmap(v), ("≤ " if i == 0 else "≥ " if i == 4 else "") + fmt.format(v)) for i, v in enumerate(vals)]
    items = "".join(
        f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
        f"<span style='width:22px;height:12px;border-radius:3px;background:{_hex(c)};display:inline-block'></span>"
        f"<span>{t}</span></div>" for c, t in rows)
    return (f"<div style='position:fixed;bottom:28px;right:12px;z-index:9999;background:rgba(255,255,255,.95);"
            f"border:1px solid #d5d4cf;border-radius:10px;padding:10px 12px;font:12px Inter,Segoe UI,sans-serif;"
            f"color:#0b0b0b;box-shadow:0 2px 8px rgba(0,0,0,.08)'><div style='font-weight:700;margin-bottom:4px'>"
            f"{label}</div>{items}<div style='display:flex;align-items:center;gap:8px;margin-top:3px;color:#6b6a65'>"
            f"<span style='width:22px;height:12px;border-radius:3px;background:#d9d8d4'></span>No match</div></div>")


def _hex(c):
    if isinstance(c, str):
        return c[:7]
    r, g, b = [int(round(x * 255)) if x <= 1 else int(x) for x in c[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def static_map_png(df: pd.DataFrame, geo: dict, indicator: str = "pop_2023") -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    from matplotlib.patches import Polygon

    label = INDICATORS[indicator][0]
    vals = df.set_index("district")[indicator]
    cmap = LinearSegmentedColormap.from_list("seq", ["#fde047", "#fb923c", "#f43f5e", "#c026d3", "#5b21b6"])
    if indicator == "females_per_100_males":
        cmap = LinearSegmentedColormap.from_list("div", ["#c026d3", "#f1f5f9", "#2563eb"])
        norm = Normalize(85, 105)
    elif indicator in ("pop_2023", "density"):
        norm = LogNorm(max(vals.quantile(0.02), 1), vals.quantile(0.98))
    else:
        norm = Normalize(vals.quantile(0.02), vals.quantile(0.98))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    patches, colours = [], []
    for f in geo["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        d = f["properties"].get("district", "")
        v = vals.get(d, np.nan) if d else np.nan
        c = "#d9d8d4" if pd.isna(v) else cmap(norm(v))
        for p in polys:
            patches.append(Polygon(np.array(p[0]), closed=True))
            colours.append(c)
    ax.add_collection(PatchCollection(patches, facecolor=colours, edgecolor="white", linewidth=0.3))
    for d, (x, y) in geo.get("markers", {}).items():
        v = vals.get(d, np.nan)
        ax.scatter(x, y, s=28, color=cmap(norm(v)) if not pd.isna(v) else "#d9d8d4",
                   edgecolor="white", linewidth=1, zorder=3)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
    cb.outline.set_visible(False)
    ax.set_title(f"Pakistan - {label}\nCensus 2023, by district", fontsize=12, loc="left", color="#0b0b0b")
    fig.text(0.02, 0.02, "Source: PBS 7th Population & Housing Census 2023 | CensusInsight PK",
             fontsize=7, color="#52514e")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()
