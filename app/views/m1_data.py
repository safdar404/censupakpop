import pandas as pd
import streamlit as st

from core.config import PROCESSED_CSV, RAW_DIR
from core.data_loader import load_census, quality_summary
from core.ui import get_df, kpi_row, load_default, page_header

page_header("Module 1", "Data Loading, Cleaning & Preprocessing",
            "Load the official PBS Census-2023 tables, clean them and merge them into one standardised district dataset.")

tab_load, tab_quality, tab_data, tab_pipe = st.tabs(["📥 Load data", "🩺 Data quality", "🧾 Clean dataset", "⚙️ Pipeline"])

with tab_load:
    st.markdown("""
**Where the data comes from:** pbs.gov.pk → *Population Census 2023 → Detailed Results (Excel)*. Download, per province,
**Table 1** (area, population, sex ratio, urban share, household size, 2017 population, growth), and optionally **Table 5**
(age groups) and **Table 12** (literacy). Name files with the province (e.g. `table_01_punjab_district.xlsx`).
""")
    c1, c2 = st.columns([3, 2])
    with c1:
        up = st.file_uploader("Browse PBS census files (.xlsx / .csv) - select several at once",
                              type=["xlsx", "xls", "csv"], accept_multiple_files=True)
        if up and st.button("🧹 Clean & merge uploaded files", type="primary"):
            with st.spinner("Detecting headers, fixing merged cells, standardising columns..."):
                try:
                    div = pd.read_csv(RAW_DIR / "district_divisions.csv")
                    new_df, rep = load_census([(f.name, f) for f in up], divisions=div)
                    st.session_state["df"] = new_df
                    st.session_state["data_source"] = f"Uploaded: {len(up)} file(s), {len(new_df)} districts"
                    st.session_state.pop("ml_result", None)
                    st.success(f"Loaded {len(new_df)} districts from {len(up)} file(s).")
                    st.json({"files": rep.tables, "rows_dropped": rep.rows_dropped, "warnings": rep.warnings})
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not build dataset: {e}")
    with c2:
        st.info("The app ships with the **official PBS district tables already loaded** (136 districts, "
                "Tables 1, 5 and 12), so every module works immediately.")
        if st.button("↺ Reset to bundled PBS dataset"):
            st.session_state["df"] = load_default()
            st.session_state["data_source"] = "Bundled PBS Census 2023 district tables (136 districts)"
            st.rerun()
        with st.expander("Bundled raw files"):
            st.write(sorted(p.name for p in RAW_DIR.glob("*.csv")))

df = get_df()
q = quality_summary(df)

with tab_quality:
    kpi_row([
        ("Districts loaded", f"{q['districts']}", "district-level records", "#7c3aed"),
        ("National population", f"{q['population']:,}", "sum of districts", "#14b8a6"),
        ("Provinces / territories", f"{q['provinces']}", "Punjab, Sindh, KP, Balochistan, ICT", "#ec4899"),
        ("Incomplete records", f"{q['incomplete_records']}", "rows with any missing key field", "#f97316"),
    ])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Missing values per column")
        st.dataframe(pd.DataFrame({"column": q["missing_by_column"].keys(),
                                   "missing": q["missing_by_column"].values()}), hide_index=True, width="stretch")
    with c2:
        st.markdown("##### Districts per province")
        st.dataframe(df.groupby("province").agg(districts=("district", "count"), population=("pop_2023", "sum"))
                     .style.format({"population": "{:,}"}), width="stretch")
        if q["incomplete_districts"]:
            st.warning("Incomplete: " + ", ".join(q["incomplete_districts"]))
    st.markdown("##### Data types after cleaning")
    st.dataframe(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)}).T, width="stretch")

with tab_data:
    st.dataframe(df, width="stretch", height=520, hide_index=True)
    st.download_button("⬇️ Download cleaned dataset (CSV)", df.to_csv(index=False).encode(),
                       "pakistan_districts_2023_clean.csv", "text/csv")

with tab_pipe:
    st.markdown("""
| Step | What the code does | Why |
|---|---|---|
| 1. Read | Reads every sheet of each Excel/CSV (`openpyxl`) | PBS ships one file per province per table |
| 2. Find header | Scores the first 20 rows for header keywords; merges 2-row headers | Government files have titles above the table |
| 3. Merged cells | Forward-fills merged header cells and the district-name column | Merged cells become blanks when read |
| 4. Standardise | Regex rules map *“ALL SEXES”, “Population-2023”…* → `pop_2023` etc. | Same meaning, different spellings across files |
| 5. Convert types | `"1,234"`→1234, `"-"`→0, blanks→NaN; integers for counts | Charts & ML need numbers |
| 6. Reshape | OVERALL / RURAL / URBAN rows → urban & rural columns | One row per district |
| 7. Merge | Joins Table 1 + Table 5 (age) + Table 12 (literacy) with name aliases | e.g. *Tando Ahyar* = *Tando Allahyar* |
| 8. Derive | Density, females per 100 males, households, urbanisation change, growth | Indicators used by Modules 2-8 |
| 9. Store | Saved to `data/processed/pakistan_districts_2023.csv` and session memory | Shared by every module |
""")
    st.caption(f"Processed file: `{PROCESSED_CSV.relative_to(PROCESSED_CSV.parents[2])}`")
