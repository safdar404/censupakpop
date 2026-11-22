---
title: CensusInsight PK
emoji: 🇵🇰
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# 🇵🇰 CensusInsight PK - Pakistan Population Data Analysis System

**AI-powered analytics on Pakistan's 7th Population & Housing Census 2023 (Pakistan Bureau of Statistics)**

| | |
|---|---|
| **Domain** | Data Science · Data Visualisation · Demographic Analysis · GeoAI · Agentic AI |

The app turns the official PBS district tables into an interactive web application with 9 modules, an
**agentic AI analyst**, unsupervised anomaly detection, ML population projections and one-click PDF reports.
It is packaged as a **Docker image** and can be deployed online for free.

---

## ✨ Features

| # | Module | Highlights |
|---|---|---|
| 1 | Data Loading & Cleaning | Upload PBS Excel/CSV; auto header detection, merged cells, `"1,234"`→number, name aliases, quality panel |
| 2 | National Dashboard | 4 KPI cards, province bar (click to drill down), share donut, male vs female |
| 3 | Urban vs Rural | Stacked bars, most urban/rural top-10s, 2017→2023 urbanisation dumbbell, urban vs rural growth |
| 4 | Density & Growth | Density of all 136 districts, top/bottom 10, High/Low Growth Zones, household size |
| 5 | Gender & Demography | Females per 100 males, outliers, colour grid of all districts, age groups, literacy gap |
| 6 | GeoAI Maps | Folium choropleth for 7 indicators, hover tooltips, top/bottom 10 sidebar, PNG export, HDX upload |
| 7 | ML Model Lab | Linear, Ridge, Random Forest, Gradient Boosting · 80/20 + 5-fold CV · MAE/R²/MAPE · feature importance · joblib |
| 8 | Population Projector | District dropdown auto-fills 2023 values, growth slider, 2030/2035 bars, auto-written insight, planning needs |
| 🤖 | AI Census Agent | Tool-calling agent (8 tools) with LLM brain **or** offline planner; shows its reasoning steps |
| ✨ | AI Insights | Grounded AI executive briefing, Isolation-Forest anomalies, K-Means district typologies |
| 🌍 | 3D Globe (Home) | D3 orthographic globe: auto-spin, fly-to-Pakistan, province tour, glowing province highlight, live stats card |
| 🎨 | Geo-infographics | Headline tiles, province profile cards with map silhouettes, district bubble map, "Pakistan as 100 people" |
| 📐 | Statistics Lab | Treemap, sunburst, sankey, waterfall, radar, violin, histogram, 100% age bars, correlation matrix, OLS regression, Lorenz curve + Gini |
| 9 | Report Generator | Professional PDF: header, tables, charts, ML results, projections, session maps |

## 📊 Data

* **Census tables:** PBS *7th Population & Housing Census 2023 - Detailed Results*, district level:
  Table 1 (area, population by sex, sex ratio, density, urban %, household size, 2017 population, growth),
  Table 5 (age groups), Table 12 (literacy). Source: <https://www.pbs.gov.pk>. The bundled CSVs in `data/raw`
  are PDF→CSV conversions of the PBS tables (github.com/fahad-mirza/pakistan_census_2023_tables) plus division
  & household counts from the `PakPC2023` package. Verified: national total **241,499,431** across **136 districts**.
* **World map:** Natural Earth 110m via the `world-atlas` package (globe). Province outlines are dissolved from the
  district polygons by `scripts/build_provinces.py` (needs `shapely`; former FATA merged into KP).
* **Boundaries:** GADM district polygons (github.com/pakdata/gisdata). 116 districts are drawn as polygons and the 20
  districts created after that boundary release are drawn as points. Official 2023 shapes can be uploaded from HDX
  (`data.humdata.org` → *pak_admbnda_adm2*) inside Module 6.

## 🚀 Run locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_model.py                           # optional: pre-train the model
streamlit run app/streamlit_app.py                      # open http://localhost:8501
```

## 🐳 Run with Docker

```bash
docker build -t censusinsight-pk .
docker run -p 8501:8501 censusinsight-pk
# with the LLM brain switched on:
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... censusinsight-pk
# or
docker compose up --build
```

The image rebuilds the dataset from the raw PBS tables, links the boundaries and pre-trains the ML model during
`docker build`, runs as a non-root user and exposes a health check at `/_stcore/health`.

## 🌍 Put it online (free options)

### Option A - Hugging Face Spaces (recommended, free)
1. Create an account at huggingface.co → **New Space** → SDK **Docker** → *Blank* → Public.
2. Push this folder to the Space:
   ```bash
   git init && git add . && git commit -m "CensusInsight PK"
   git remote add space https://huggingface.co/spaces/<your-username>/censusinsight-pk
   git push space main          # use an HF access token as the password
   ```
3. The YAML header at the top of this README tells Spaces to build the Dockerfile on port 8501.
4. (Optional) Space → *Settings → Variables and secrets* → add secret `OPENAI_API_KEY`.
5. Your app is live at `https://<your-username>-censusinsight-pk.hf.space`.

### Option B - Render.com
New → **Web Service** → connect the GitHub repo → Runtime **Docker** → Instance *Free* → deploy.
Render sets `$PORT` automatically; the Dockerfile already uses it.

### Option C - Docker Hub / GHCR + any VPS
The GitHub Action in `.github/workflows/docker-publish.yml` builds and pushes `ghcr.io/<user>/censusinsight-pk`
on every push to `main`. On any server: `docker run -d -p 80:8501 ghcr.io/<user>/censusinsight-pk:latest`.

## 🤖 Enabling the LLM agent (optional)

Everything works offline. To give the agent an LLM brain, set any OpenAI-compatible endpoint:

| Provider | Variables |
|---|---|
| OpenAI | `OPENAI_API_KEY=sk-...` `LLM_MODEL=gpt-4o-mini` |
| Groq (free tier) | `OPENAI_API_KEY=gsk_...` `OPENAI_BASE_URL=https://api.groq.com/openai/v1` `LLM_MODEL=llama-3.3-70b-versatile` |
| Local Ollama | `OPENAI_API_KEY=ollama` `OPENAI_BASE_URL=http://host.docker.internal:11434/v1` `LLM_MODEL=qwen2.5` |

## 🗂️ Project structure

```
censusinsight-pk/
├── app/
│   ├── streamlit_app.py        # entry point + navigation
│   ├── core/                   # engine (no UI code)
│   │   ├── data_loader.py      # Module 1 - PBS Excel/CSV loader & cleaner
│   │   ├── analytics.py        # Modules 2-5 + anomaly detection + clustering
│   │   ├── geo.py              # Module 6 - boundaries, Folium & static maps
│   │   ├── ml.py               # Modules 7-8 - training, comparison, projection
│   │   ├── agent.py            # AI Census Agent (tools + LLM loop + offline planner)
│   │   ├── report.py           # Module 9 - PDF generator
│   │   ├── ui.py / config.py   # design system, settings
│   └── views/                  # one file per page
├── data/raw/                   # PBS tables (CSV)
├── data/processed/             # clean dataset (generated)
├── app/web/                    # D3 globe + infographics templates (data injected at runtime)
├── data/geo/                   # district & province boundaries, world map, district centroids
├── models/                     # saved ML model (generated)
├── scripts/                    # build_dataset · build_geo · train_model
├── tests/                      # page smoke tests + flow tests (pytest)
├── Dockerfile · docker-compose.yml · requirements.txt
```

## ✅ Tests

```bash
pip install pytest && python -m pytest tests -q      # 18 tests: every page + agent, projector, PDF flows
```

## ⚠️ Notes & limitations
* Growth rates are PBS compound annual rates 2017-2023. Projections assume a constant rate: they are scenarios, not official forecasts.
* Districts whose borders changed between censuses can show extreme growth; the ML module clips the top/bottom 2 %.
* AJK and Gilgit-Baltistan are not part of the PBS census tables.
