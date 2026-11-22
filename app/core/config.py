"""Central configuration: paths, colours and indicator metadata."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_CSV = DATA_DIR / "processed" / "pakistan_districts_2023.csv"
GEOJSON = DATA_DIR / "geo" / "pakistan_districts_census2023.geojson"
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", ROOT / "reports"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

APP_NAME = "CensusInsight PK"
APP_TAGLINE = "AI-powered analytics on Pakistan's 7th Population & Housing Census 2023"
STUDENT = os.getenv("STUDENT_NAME", "Student")
SUPERVISOR = os.getenv("SUPERVISOR_NAME", "Supervisor")
PROGRAM = "CensusInsight PK"

# ---- colour system (validated categorical order, light mode) ----
PROVINCE_COLORS = {          # vibrant scheme, shared by every chart, map and the globe
    "Punjab": "#2563eb",
    "Sindh": "#f43f5e",
    "Khyber Pakhtunkhwa": "#10b981",
    "Balochistan": "#f59e0b",
    "Islamabad (ICT)": "#8b5cf6",
}
C_PRIMARY = "#7c3aed"   # urban / single-series (violet)
C_SECOND = "#f97316"    # rural (orange)
C_GOOD = "#14b8a6"
C_MUTED = "#9a9993"
SEQ_PLASMA = ["#fde047", "#fb923c", "#f43f5e", "#c026d3", "#5b21b6"]
DIVERGING = ["#c4521f", "#eb6834", "#f3b496", "#e4e3df", "#9ec5f4", "#2a78d6", "#184f95"]

# indicator key -> (label, unit/format, colour scale name for maps)
INDICATORS = {
    "pop_2023": ("Total population (2023)", "{:,.0f}", "Blues"),
    "density": ("Population density (people / km²)", "{:,.0f}", "Blues"),
    "growth_rate": ("Annual growth rate 2017-23 (%)", "{:.2f}%", "Oranges"),
    "urban_pct": ("Urban population (%)", "{:.1f}%", "Purples"),
    "females_per_100_males": ("Females per 100 males", "{:.1f}", "RdBu"),
    "avg_hh_size": ("Average household size", "{:.2f}", "Greens"),
    "literacy_rate": ("Literacy rate, 10+ (%)", "{:.1f}%", "Greens"),
}

HIGH_GROWTH = 3.0   # % per year
LOW_GROWTH = 1.0    # % per year

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))
