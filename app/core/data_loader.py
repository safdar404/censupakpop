"""
Module 1 - Data Loading, Cleaning & Preprocessing
=================================================
Loads Pakistan Bureau of Statistics (PBS) Census-2023 tables (Excel .xlsx or
CSV), cleans them and merges them into ONE standardised district dataset.

PBS publishes one file per province per table. The three tables we use:
  * Table 1  - area, population by sex, sex ratio, density, urban share,
               household size, 2017 population and 2017-23 growth rate
  * Table 5  - population by age group (for 0-14 / 15-64 / 65+)
  * Table 12 - literacy (literacy rate by sex)

Government Excel files are messy: title rows above the header, multi-row
headers made from merged cells, blank spacer rows, numbers stored as text
("1,234"), and "-" for zero.  `read_pbs_file` handles all of these.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROVINCES = {
    "punjab": "Punjab",
    "sindh": "Sindh",
    "kp": "Khyber Pakhtunkhwa",
    "khyber": "Khyber Pakhtunkhwa",
    "kpk": "Khyber Pakhtunkhwa",
    "balochistan": "Balochistan",
    "baluchistan": "Balochistan",
    "islamabad": "Islamabad (ICT)",
    "ict": "Islamabad (ICT)",
}
PROVINCE_ORDER = ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad (ICT)"]
PROVINCE_SHORT = {"Punjab": "Punjab", "Sindh": "Sindh", "Khyber Pakhtunkhwa": "KP",
                  "Balochistan": "Balochistan", "Islamabad (ICT)": "ICT"}

# Header synonyms -> standard column names. ORDER MATTERS (female before male,
# 2017 before generic population, etc.).
HEADER_RULES: list[tuple[str, str]] = [
    (r"name.*admin|administrative unit|^name$", "name"),
    (r"^admin_unit$|^admin unit$|unit type", "admin_unit"),
    (r"part.*district", "part_of_district"),
    (r"^region$|rural/urban|locality", "region"),
    (r"area", "area_sqkm"),
    (r"(gr|growth).*rate|g\.?\s*rate", "growth_rate"),
    (r"2017", "pop_2017"),
    (r"sex.?ratio", "sex_ratio"),
    (r"densit|pop_den", "density"),
    (r"urban.*(prop|%|percent|share)", "urban_pct"),
    (r"(hh|household).*size", "avg_hh_size"),
    (r"t.?gend|transgender", "transgender"),
    (r"female", "female"),
    (r"male", "male"),
    (r"all.?sexes|both.?sexes|total.*pop|population.?-?2023|^all_sexes$", "pop_2023"),
    (r"age.?group|sex.*age", "age_group"),
    (r"^vars$|indicator|variable", "vars"),
]

REQUIRED_T1 = {"name", "pop_2023", "male", "female"}


@dataclass
class LoadReport:
    files: list[str] = field(default_factory=list)
    tables: dict[str, str] = field(default_factory=dict)  # file -> detected table type
    warnings: list[str] = field(default_factory=list)
    rows_dropped: int = 0


# ----------------------------------------------------------------------------
# Low level helpers
# ----------------------------------------------------------------------------
def _clean_header(text) -> str:
    s = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    s = re.sub(r"\s+", " ", s.replace("\n", " ")).strip().lower()
    return s


def standardise_column(raw: str) -> str:
    h = _clean_header(raw)
    # keep rural/urban/overall suffix info for wide tables (Table 5 / 12)
    suffix = ""
    for part in ("overall", "rural", "urban"):
        if h.endswith(part) and not h.startswith(part) and h != "region":
            suffix = "_" + part
            h = h[: -len(part)].strip(" _-")
    for pattern, std in HEADER_RULES:
        if re.search(pattern, h):
            return std + suffix
    return re.sub(r"[^a-z0-9]+", "_", h).strip("_") + suffix


def to_number(series: pd.Series) -> pd.Series:
    """'1,234' -> 1234, '-' -> 0, '' -> NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"-": "0", "--": "0", "": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _find_header_row(raw: pd.DataFrame, max_scan: int = 20) -> int:
    """Return the index of the row that looks most like a header."""
    best, best_score = 0, -1
    keywords = ["area", "population", "male", "female", "sex", "ratio", "district",
                "name", "urban", "rural", "age", "literate", "household", "region"]
    for i in range(min(max_scan, len(raw))):
        cells = [_clean_header(c) for c in raw.iloc[i].tolist()]
        score = sum(any(k in c for k in keywords) for c in cells if c)
        if score > best_score:
            best, best_score = i, score
    return best


def _flatten_excel_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw (header=None) Excel sheet into a tidy frame.
    Handles title rows and 2-row merged headers."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    h = _find_header_row(raw)
    top = raw.iloc[h].ffill()  # merged cells -> forward fill across
    header = [_clean_header(x) for x in top.tolist()]
    body_start = h + 1
    # If the next row is also text (second header line) combine them
    if body_start < len(raw):
        nxt = raw.iloc[body_start]
        n_text = sum(isinstance(x, str) and not re.fullmatch(r"[\d,.\- ]+", x) for x in nxt)
        n_num = sum(isinstance(x, (int, float)) and not pd.isna(x) for x in nxt)
        if n_text > n_num:
            header = [f"{a} {_clean_header(b)}".strip() if _clean_header(b) else a
                      for a, b in zip(header, nxt.tolist())]
            body_start += 1
    df = raw.iloc[body_start:].copy()
    df.columns = [c if c else f"col_{i}" for i, c in enumerate(header)]
    return df


def read_pbs_file(source, filename: str | None = None) -> list[pd.DataFrame]:
    """Read a CSV or Excel file (path or uploaded file-like). Returns one
    frame per sheet with standardised column names."""
    name = filename or str(getattr(source, "name", source))
    frames: list[pd.DataFrame] = []
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, engine="openpyxl")
        for _, raw in sheets.items():
            if raw.dropna(how="all").empty:
                continue
            frames.append(_flatten_excel_sheet(raw))
    else:
        frames.append(pd.read_csv(source, dtype=str))
    out = []
    for df in frames:
        df.columns = [standardise_column(c) for c in df.columns]
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
        df = df.dropna(how="all")
        # merged cells down the "name" column -> forward fill
        for col in ("name", "part_of_district", "admin_unit"):
            if col in df.columns:
                df[col] = df[col].replace("", np.nan).ffill()
        out.append(df)
    return out


def detect_table(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if "age_group" in cols:
        return "age"
    if "vars" in cols:
        return "literacy"
    if REQUIRED_T1 <= cols:
        return "table1"
    return "unknown"


def detect_province(filename: str, df: pd.DataFrame | None = None) -> str | None:
    f = Path(filename).name.lower()
    for key, prov in PROVINCES.items():
        if re.search(rf"(^|[_\-\s.]){key}([_\-\s.]|$)", f):
            return prov
    if df is not None and "province" in df.columns:
        v = str(df["province"].dropna().iloc[0]).lower()
        for key, prov in PROVINCES.items():
            if key in v:
                return prov
    return None


# Same district spelled differently across PBS tables
NAME_ALIASES = {"TANDOAHYAR": "TANDOALLAHYAR", "MALAKAND": "MALAKANDPROTECTEDAREA",
                "ISLAMABAD": "ISLAMABADICT", "UMERKOT": "UMERKOT"}


def norm_name(s: str) -> str:
    k = re.sub(r"[^A-Z]", "", str(s).upper())
    return NAME_ALIASES.get(k, k)


# ----------------------------------------------------------------------------
# Table specific parsers
# ----------------------------------------------------------------------------
def _district_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "admin_unit" in df.columns:
        mask = df["admin_unit"].astype(str).str.upper().str.contains("DISTRICT")
        if mask.any():
            df = df[mask]
    return df


def parse_table1(df: pd.DataFrame, province: str) -> pd.DataFrame:
    df = _district_rows(df.copy())
    num_cols = ["area_sqkm", "pop_2023", "male", "female", "transgender", "sex_ratio",
                "density", "urban_pct", "avg_hh_size", "pop_2017", "growth_rate"]
    for c in num_cols:
        if c in df.columns:
            df[c] = to_number(df[c])
    df["name"] = df["name"].astype(str).str.strip().str.upper()
    if "region" not in df.columns:
        df["region"] = "OVERALL"
    df["region"] = df["region"].astype(str).str.upper().str.strip()
    overall = df[df["region"].str.contains("OVERALL|ALL|TOTAL")].set_index("name")
    rural = df[df["region"].str.contains("RURAL")].set_index("name")
    urban = df[df["region"].str.contains("URBAN")].set_index("name")
    out = overall[[c for c in num_cols if c in overall.columns]].copy()
    out["province"] = province
    if not rural.empty:
        out["rural_pop"] = rural["pop_2023"].reindex(out.index)
        out["rural_pop_2017"] = rural.get("pop_2017", pd.Series(dtype=float)).reindex(out.index)
        out["rural_growth_rate"] = rural.get("growth_rate", pd.Series(dtype=float)).reindex(out.index)
    if not urban.empty:
        out["urban_pop"] = urban["pop_2023"].reindex(out.index)
        out["urban_pop_2017"] = urban.get("pop_2017", pd.Series(dtype=float)).reindex(out.index)
        out["urban_growth_rate"] = urban.get("growth_rate", pd.Series(dtype=float)).reindex(out.index)
    return out.reset_index().rename(columns={"name": "district"})


def parse_age(df: pd.DataFrame) -> pd.DataFrame:
    df = _district_rows(df.copy())
    tot = "pop_2023_overall" if "pop_2023_overall" in df.columns else "pop_2023"
    df[tot] = to_number(df[tot])
    df["age_group"] = df["age_group"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    df["name"] = df["name"].astype(str).str.strip().str.upper()
    piv = df.pivot_table(index="name", columns="age_group", values=tot, aggfunc="first")
    out = pd.DataFrame(index=piv.index)
    allages = piv.get("ALL AGES")
    if allages is None:
        return pd.DataFrame(columns=["district"])
    out["age_0_14_pct"] = piv.get("UNDER 15") / allages * 100
    out["age_15_64_pct"] = piv.get("15 - 64") / allages * 100
    out["age_65_plus_pct"] = piv.get("65 & ABOVE") / allages * 100
    return out.reset_index().rename(columns={"name": "district"})


def parse_literacy(df: pd.DataFrame) -> pd.DataFrame:
    df = _district_rows(df.copy())
    df["name"] = df["name"].astype(str).str.strip().str.upper()
    lit = df[df["vars"].astype(str).str.contains("Literate %", case=False, regex=False)]
    out = pd.DataFrame({"district": lit["name"]})
    for src, dst in [("pop_2023_overall", "literacy_rate"), ("male_overall", "literacy_male"),
                     ("female_overall", "literacy_female"), ("pop_2023_urban", "literacy_urban"),
                     ("pop_2023_rural", "literacy_rural")]:
        if src in lit.columns:
            out[dst] = to_number(lit[src]).values
    return out.drop_duplicates("district")


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def load_census(files: list, divisions: pd.DataFrame | None = None) -> tuple[pd.DataFrame, LoadReport]:
    """files: list of paths or (name, file-like) tuples. Returns the unified
    district dataframe and a report of what was found/cleaned."""
    rep = LoadReport()
    t1, age, lit = [], [], []
    for item in files:
        name, src = (item if isinstance(item, tuple) else (str(item), item))
        rep.files.append(Path(name).name)
        try:
            frames = read_pbs_file(src, name)
        except Exception as e:  # noqa: BLE001
            rep.warnings.append(f"{Path(name).name}: could not read ({e})")
            continue
        for df in frames:
            kind = detect_table(df)
            prov = detect_province(name, df)
            rep.tables[Path(name).name] = kind
            if kind == "unknown":
                rep.warnings.append(f"{Path(name).name}: table type not recognised - skipped")
                continue
            if kind == "table1":
                if not prov:
                    rep.warnings.append(f"{Path(name).name}: province not detected from file name")
                    prov = "Unknown"
                t1.append(parse_table1(df, prov))
            elif kind == "age":
                age.append(parse_age(df))
            else:
                lit.append(parse_literacy(df))
    if not t1:
        raise ValueError("No Table-1 (area / population / sex) file found. Table 1 is required.")
    base = pd.concat(t1, ignore_index=True)
    before = len(base)
    base = base.dropna(subset=["pop_2023"]).drop_duplicates(["province", "district"])
    rep.rows_dropped = before - len(base)
    base["_key"] = base["district"].map(norm_name)
    for extra in (age, lit):
        if extra:
            e = pd.concat(extra).drop_duplicates("district")
            e["_key"] = e["district"].map(norm_name)
            base = base.merge(e.drop(columns="district"), on="_key", how="left")
    base = base.drop(columns="_key")
    df = derive_indicators(base)
    if divisions is not None:
        df = attach_divisions(df, divisions)
    return df, rep


def derive_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["district"] = df["district"].str.title().str.replace(r"\bOf\b", "of", regex=True)
    if "urban_pop" not in df.columns:
        df["urban_pop"] = df["pop_2023"] * df.get("urban_pct", 0) / 100
    df["urban_pop"] = df["urban_pop"].fillna(0)
    df["rural_pop"] = df.get("rural_pop", df["pop_2023"] - df["urban_pop"]).fillna(df["pop_2023"] - df["urban_pop"])
    df["urban_pct"] = df["urban_pop"] / df["pop_2023"] * 100
    df["rural_pct"] = 100 - df["urban_pct"]
    if "urban_pop_2017" in df.columns:
        df["urban_pop_2017"] = df["urban_pop_2017"].fillna(0)
        df["urban_pct_2017"] = df["urban_pop_2017"] / df["pop_2017"] * 100
        df["urbanisation_change_pp"] = df["urban_pct"] - df["urban_pct_2017"]
    df["density"] = df.get("density", df["pop_2023"] / df["area_sqkm"])
    df["density"] = df["density"].fillna(df["pop_2023"] / df["area_sqkm"])
    df["females_per_100_males"] = df["female"] / df["male"] * 100
    df["households"] = (df["pop_2023"] / df["avg_hh_size"]).round()
    if "growth_rate" not in df.columns or df["growth_rate"].isna().all():
        df["growth_rate"] = ((df["pop_2023"] / df["pop_2017"]) ** (1 / 6) - 1) * 100
    for c in ("urban_growth_rate", "rural_growth_rate"):
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df["province"] = pd.Categorical(df["province"], categories=PROVINCE_ORDER + ["Unknown"])
    df = df.sort_values(["province", "district"]).reset_index(drop=True)
    df["province"] = df["province"].astype(str)
    int_cols = ["pop_2023", "male", "female", "transgender", "pop_2017", "urban_pop", "rural_pop", "households"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).round().astype("int64")
    return df


def attach_divisions(df: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    div = div.copy()
    div["_key"] = div["District"].map(norm_name)
    div.loc[div["Region"] == "ICT", "_key"] = "ISLAMABADICT"
    m = dict(zip(div["_key"], div["Division"]))
    hh = dict(zip(div["_key"], div.get("Households", pd.Series(dtype=float))))
    df = df.copy()
    keys = df["district"].map(norm_name)
    df["division"] = keys.map(m).fillna("")
    # use the official household count where PBS publishes it
    official = keys.map(hh)
    df["households"] = official.fillna(df["households"]).round().astype("int64")
    return df


def quality_summary(df: pd.DataFrame) -> dict:
    key = ["pop_2023", "male", "female", "area_sqkm", "avg_hh_size", "pop_2017",
           "urban_pct", "literacy_rate", "age_0_14_pct"]
    missing = {c: int(df[c].isna().sum()) for c in key if c in df.columns}
    incomplete = df[[c for c in key if c in df.columns]].isna().any(axis=1)
    return {
        "districts": int(len(df)),
        "provinces": int(df["province"].nunique()),
        "population": int(df["pop_2023"].sum()),
        "missing_by_column": missing,
        "incomplete_records": int(incomplete.sum()),
        "incomplete_districts": df.loc[incomplete, "district"].tolist(),
    }
