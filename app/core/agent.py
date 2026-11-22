"""
AI Census Analyst - an agentic layer over the census data
=========================================================
The agent answers natural-language questions ("Which 5 districts of Sindh
grew fastest?", "Project Lahore for 2035 at 2%") by PLANNING which tools to
call, CALLING them on the real dataset, and WRITING an answer grounded in
the tool results (no made-up numbers).

Two brains, same tools:
  1. LLM mode   - OpenAI-compatible function calling (OpenAI, Groq, Together,
                  a local Ollama server ...).  Set OPENAI_API_KEY and optionally
                  OPENAI_BASE_URL / LLM_MODEL.
  2. Offline mode - a deterministic intent planner (regex + fuzzy matching)
                  that calls the same tools.  Always available, so the viva
                  demo never depends on internet or API credits.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from . import analytics as A
from . import ml
from .config import INDICATORS, LLM_MODEL, llm_available
from .data_loader import PROVINCE_ORDER

INDICATOR_ALIASES = {
    "pop_2023": ["population", "populous", "people", "biggest", "largest", "smallest", "size"],
    "density": ["density", "dense", "crowded", "per km", "per square"],
    "growth_rate": ["growth", "growing", "grew", "fastest", "slowest"],
    "urban_pct": ["urban", "urbanised", "urbanized", "city", "rural"],
    "urbanisation_change_pp": ["urbanisation change", "urbanization change", "urbanised the most", "urbanized the most"],
    "females_per_100_males": ["gender", "sex ratio", "female", "women", "male"],
    "avg_hh_size": ["household size", "household", "hh size", "family size"],
    "literacy_rate": ["literacy", "literate", "education"],
    "literacy_female": ["female literacy", "women literacy", "girls"],
    "age_0_14_pct": ["children", "youth", "young", "0-14"],
    "age_65_plus_pct": ["elderly", "old age", "65"],
}
LABELS = {**{k: v[0] for k, v in INDICATORS.items()},
          "urbanisation_change_pp": "Urbanisation change 2017-23 (pp)",
          "literacy_female": "Female literacy (%)", "age_0_14_pct": "Children 0-14 (%)",
          "age_65_plus_pct": "Elderly 65+ (%)"}


@dataclass
class AgentResult:
    answer: str
    steps: list = field(default_factory=list)   # [{"tool":..,"args":..,"result":..}]
    table: pd.DataFrame | None = None
    chart: dict | None = None                    # {"type":"bar","x":[..],"y":[..],"title":..}
    mode: str = "offline"


# =========================================================== TOOLS
class CensusTools:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.names = df["district"].tolist()
        self._bundle = None

    # -- helpers
    def _district(self, name: str) -> pd.Series:
        hit = self.match_district(name)
        if not hit:
            raise ValueError(f"District '{name}' not found")
        return self.df[self.df["district"] == hit].iloc[0]

    def match_district(self, name: str) -> str | None:
        low = {n.lower(): n for n in self.names}
        if name.lower() in low:
            return low[name.lower()]
        m = difflib.get_close_matches(name.lower(), list(low), n=1, cutoff=0.75)
        return low[m[0]] if m else None

    @staticmethod
    def match_province(text: str) -> str | None:
        t = text.lower()
        for key, p in [("punjab", "Punjab"), ("sindh", "Sindh"), ("khyber", "Khyber Pakhtunkhwa"),
                       ("kpk", "Khyber Pakhtunkhwa"), (" kp", "Khyber Pakhtunkhwa"),
                       ("balochistan", "Balochistan"), ("baluchistan", "Balochistan"),
                       ("islamabad", "Islamabad (ICT)"), ("ict", "Islamabad (ICT)")]:
            if re.search(rf"(^|\W){key.strip()}(\W|$)", t):
                return p
        return None

    @property
    def bundle(self):
        if self._bundle is None:
            self._bundle = ml.ensure_model(self.df)
        return self._bundle

    # -- the tools (each returns JSON-serialisable data)
    def national_overview(self) -> dict:
        n = A.national_summary(self.df)
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in n.items()}

    def province_profile(self, province: str | None = None) -> list | dict:
        p = A.province_summary(self.df).round(2)
        cols = ["districts", "pop_2023", "share_pct", "urban_pct", "growth_rate", "density",
                "avg_hh_size", "females_per_100_males", "literacy_rate", "literacy_male", "literacy_female"]
        if province:
            prov = self.match_province(province) or province
            if prov in p.index:
                return {"province": prov, **p.loc[prov, cols].to_dict()}
        return p[cols].reset_index().to_dict("records")

    def district_profile(self, district: str) -> dict:
        r = self._district(district)
        keys = ["district", "province", "division", "area_sqkm", "pop_2017", "pop_2023", "growth_rate",
                "urban_pct", "urbanisation_change_pp", "density", "households", "avg_hh_size",
                "females_per_100_males", "literacy_rate", "literacy_male", "literacy_female",
                "age_0_14_pct", "age_15_64_pct", "age_65_plus_pct"]
        out = {k: (round(float(r[k]), 2) if isinstance(r[k], float) else r[k]) for k in keys if k in r}
        for k in ("pop_2017", "pop_2023", "households"):
            out[k] = int(out[k])
        out["national_rank_population"] = int((self.df["pop_2023"] > r["pop_2023"]).sum() + 1)
        return out

    def rank_districts(self, indicator: str = "pop_2023", n: int = 10, order: str = "desc",
                       province: str | None = None) -> list:
        if indicator not in self.df.columns:
            raise ValueError(f"Unknown indicator {indicator}")
        d = self.df
        if province:
            prov = self.match_province(province) or province
            d = d[d["province"] == prov]
        r = A.top_bottom(d, indicator, int(n), ascending=(order == "asc"),
                         cols=["district", "province", "pop_2023", indicator])
        return r.round(2).to_dict("records")

    def compare_districts(self, districts: list) -> list:
        return [self.district_profile(x) for x in districts]

    def project_population(self, district: str, year: int = 2035, growth_rate: float | None = None) -> dict:
        name = self.match_district(district)
        if not name:
            raise ValueError(f"District '{district}' not found")
        years = sorted({2030, 2035, int(year)})
        p = ml.project_district(self.df, name, self.bundle, rate=growth_rate, years=tuple(years))
        p["projections"] = {str(k): int(v) for k, v in p["projections"].items()}
        p["insight"] = ml.insight_sentence({**p, "projections": {int(k): v for k, v in p["projections"].items()}})
        return p

    def filter_districts(self, indicator: str, operator: str, value: float, province: str | None = None) -> list:
        d = self.df
        if province:
            d = d[d["province"] == (self.match_province(province) or province)]
        ops = {">": d[indicator] > value, "<": d[indicator] < value,
               ">=": d[indicator] >= value, "<=": d[indicator] <= value}
        r = d[ops[operator]][["district", "province", "pop_2023", indicator]].sort_values(indicator, ascending=False)
        return r.round(2).to_dict("records")

    def find_anomalies(self, n: int = 8) -> list:
        a = A.detect_anomalies(self.df)
        return a[a["is_anomaly"]].head(n)[["district", "province", "anomaly_score", "why"]].round(3).to_dict("records")


TOOL_SPECS = [
    {"name": "national_overview", "description": "National totals for Pakistan census 2023 (population, households, urban %, growth, literacy, sex ratio).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "province_profile", "description": "Indicators for one province or all provinces.",
     "parameters": {"type": "object", "properties": {"province": {"type": "string", "description": "Punjab, Sindh, Khyber Pakhtunkhwa, Balochistan, Islamabad (ICT). Omit for all."}}}},
    {"name": "district_profile", "description": "Full census profile of one district.",
     "parameters": {"type": "object", "properties": {"district": {"type": "string"}}, "required": ["district"]}},
    {"name": "rank_districts", "description": "Top/bottom N districts by an indicator, optionally within a province.",
     "parameters": {"type": "object", "properties": {
         "indicator": {"type": "string", "enum": list(LABELS)},
         "n": {"type": "integer", "default": 10},
         "order": {"type": "string", "enum": ["desc", "asc"]},
         "province": {"type": "string"}}, "required": ["indicator"]}},
    {"name": "compare_districts", "description": "Side-by-side profiles of several districts.",
     "parameters": {"type": "object", "properties": {"districts": {"type": "array", "items": {"type": "string"}}}, "required": ["districts"]}},
    {"name": "project_population", "description": "ML-based population projection for a district (2030/2035 or a given year). growth_rate (% per year) overrides the model.",
     "parameters": {"type": "object", "properties": {"district": {"type": "string"}, "year": {"type": "integer"},
                                                     "growth_rate": {"type": "number"}}, "required": ["district"]}},
    {"name": "filter_districts", "description": "Districts where indicator <operator> value.",
     "parameters": {"type": "object", "properties": {"indicator": {"type": "string", "enum": list(LABELS)},
                                                     "operator": {"type": "string", "enum": [">", "<", ">=", "<="]},
                                                     "value": {"type": "number"}, "province": {"type": "string"}},
                    "required": ["indicator", "operator", "value"]}},
    {"name": "find_anomalies", "description": "Districts flagged as statistical anomalies by an Isolation Forest (possible data issues).",
     "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}},
]

SYSTEM_PROMPT = """You are CensusInsight PK, an AI data analyst for Pakistan's 7th Population & Housing Census 2023 (Pakistan Bureau of Statistics).
Rules:
- ALWAYS call tools to get numbers. Never invent or estimate figures from memory.
- Indicators available: """ + ", ".join(f"{k} = {v}" for k, v in LABELS.items()) + """.
- Be concise: lead with the direct answer, then 2-4 bullet points of supporting figures, then one policy-relevant implication when useful.
- Format populations in millions (e.g. 13.00 M) or with thousands separators. Growth rates are % per year (2017-2023).
- If a district is not found, say so and suggest close names. End with 'Source: PBS Census 2023'."""


# =========================================================== AGENT
class CensusAgent:
    def __init__(self, df: pd.DataFrame):
        self.tools = CensusTools(df)
        self.df = df

    def _call(self, name, args, steps):
        fn = getattr(self.tools, name)
        try:
            res = fn(**args)
        except Exception as e:  # noqa: BLE001
            res = {"error": str(e)}
        steps.append({"tool": name, "args": args, "result": res})
        return res

    # ---------------------------------------------- public
    def ask(self, question: str, history: list | None = None) -> AgentResult:
        if llm_available():
            try:
                return self._ask_llm(question, history or [])
            except Exception as e:  # noqa: BLE001 - fall back gracefully
                r = self._ask_offline(question)
                r.answer += f"\n\n_(LLM unavailable: {str(e)[:120]} - answered by offline agent)_"
                return r
        return self._ask_offline(question)

    # ---------------------------------------------- LLM agent loop
    def _ask_llm(self, question: str, history: list) -> AgentResult:
        from openai import OpenAI

        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history[-6:]:
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": question})
        tools = [{"type": "function", "function": t} for t in TOOL_SPECS]
        steps: list = []
        for _ in range(6):  # max reasoning/tool rounds
            resp = client.chat.completions.create(model=LLM_MODEL, messages=msgs, tools=tools, temperature=0.1)
            m = resp.choices[0].message
            if not m.tool_calls:
                res = AgentResult(answer=m.content or "", steps=steps, mode=f"LLM ({LLM_MODEL})")
                self._attach_visual(res)
                return res
            msgs.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
            for tc in m.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                out = self._call(tc.function.name, args, steps)
                msgs.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(out, default=str)[:12000]})
        return AgentResult(answer="I could not finish within the step limit.", steps=steps, mode="LLM")

    # ---------------------------------------------- offline planner
    def _find_districts(self, q: str) -> list[str]:
        ql = q.lower()
        found = []
        for name in sorted(self.tools.names, key=len, reverse=True):
            if re.search(rf"\b{re.escape(name.lower())}\b", ql) and not any(name.lower() in f.lower() for f in found):
                found.append(name)
        if not found:  # fuzzy on single words / bigrams (typos like 'Lahor')
            words = re.findall(r"[a-zA-Z]+", q)
            grams = words + [" ".join(p) for p in zip(words, words[1:])]
            for g in grams:
                if len(g) < 5 or self.tools.match_province(g):
                    continue
                hit = self.tools.match_district(g)
                if hit and hit not in found:
                    found.append(hit)
        order = {n: ql.find(n.lower()) for n in found}
        return sorted(found, key=lambda n: order[n] if order[n] >= 0 else 999)

    @staticmethod
    def _find_indicator(q: str) -> str:
        ql = q.lower()
        best, best_len = "pop_2023", 0
        for ind, words in INDICATOR_ALIASES.items():
            for w in words:
                if w in ql and len(w) > best_len:
                    best, best_len = ind, len(w)
        return best

    def _ask_offline(self, q: str) -> AgentResult:
        steps: list = []
        ql = q.lower()
        dists = self._find_districts(q)
        prov = self.tools.match_province(q)
        if "pakhtunkhwa" in ql and "khyber district" not in ql:
            dists = [d for d in dists if d != "Khyber"]
        ind = self._find_indicator(q)
        nums = [float(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)", q)]
        years = [int(x) for x in nums if 2024 <= x <= 2100]
        n = next((int(x) for x in nums if 1 <= x <= 50 and x not in years and "%" not in q[q.find(str(int(x))):q.find(str(int(x))) + 4]), 10)
        rate = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
        if m:
            rate = float(m.group(1))

        # 1. projection
        if dists and (years or re.search(r"project|predict|forecast|future|will .* reach|by 20", ql)):
            out = self._call("project_population", {"district": dists[0], "year": years[0] if years else 2035,
                                                    "growth_rate": rate}, steps)
            if "error" in out:
                return AgentResult(out["error"], steps)
            pj = out["projections"]
            lines = [f"**{out['district']} ({out['province']})** - population projection",
                     f"- 2017 census: **{A.fmt_pop(out['pop_2017'])}**  ·  2023 census: **{A.fmt_pop(out['pop_2023'])}**",
                     f"- Growth rate used: **{out['rate_used']:.2f}%/yr** "
                     f"({'user scenario' if rate is not None else f'blend of observed trend {out['trend_rate']:.2f}% and ML model {out['model_rate']:.2f}%'})"]
            lines += [f"- **{y}: {A.fmt_pop(v)}**" for y, v in pj.items()]
            lines += ["", out["insight"]]
            res = AgentResult("\n".join(lines), steps,
                              chart={"type": "bar", "x": ["2017", "2023"] + list(pj), "title": f"{out['district']} population",
                                     "y": [out["pop_2017"], out["pop_2023"]] + list(pj.values())})
            return res
        # 2. compare
        if len(dists) >= 2:
            out = self._call("compare_districts", {"districts": dists[:5]}, steps)
            t = pd.DataFrame(out).set_index("district")
            keep = ["province", "pop_2023", "growth_rate", "urban_pct", "density", "avg_hh_size",
                    "females_per_100_males", "literacy_rate"]
            t = t[[k for k in keep if k in t.columns]]
            lead = t["pop_2023"].idxmax()
            fast = t["growth_rate"].astype(float).idxmax()
            ans = (f"Comparing **{', '.join(t.index)}**: **{lead}** is the largest ({A.fmt_pop(t.loc[lead, 'pop_2023'])}), "
                   f"while **{fast}** grew fastest ({t.loc[fast, 'growth_rate']:.2f}%/yr). "
                   f"Literacy ranges from {t['literacy_rate'].min():.1f}% to {t['literacy_rate'].max():.1f}%.")
            return AgentResult(ans, steps, table=t.reset_index(),
                               chart={"type": "bar", "x": list(t.index), "y": list(t["pop_2023"]), "title": "Population 2023"})
        # 3. single district profile
        if len(dists) == 1:
            d = self._call("district_profile", {"district": dists[0]}, steps)
            ans = (f"**{d['district']}** ({d['province']}, {d.get('division', '')} division)\n\n"
                   f"- Population 2023: **{A.fmt_pop(d['pop_2023'])}** (rank #{d['national_rank_population']} of 136); 2017: {A.fmt_pop(d['pop_2017'])}\n"
                   f"- Growth: **{d['growth_rate']:.2f}%/yr** · Urban: **{d['urban_pct']:.1f}%** "
                   f"({d.get('urbanisation_change_pp', 0):+.1f} pp since 2017)\n"
                   f"- Density: {d['density']:,.0f} /km² over {d['area_sqkm']:,.0f} km² · Households: {d['households']:,} (avg {d['avg_hh_size']})\n"
                   f"- Females per 100 males: {d['females_per_100_males']:.1f}\n"
                   f"- Literacy: {d['literacy_rate']:.1f}% (male {d['literacy_male']:.1f}%, female {d['literacy_female']:.1f}%)\n"
                   f"- Age: {d['age_0_14_pct']:.1f}% children · {d['age_15_64_pct']:.1f}% working-age · {d['age_65_plus_pct']:.1f}% elderly")
            return AgentResult(ans, steps)
        # 4. anomalies
        if re.search(r"anomal|unusual|outlier|suspicious|undercount|data quality", ql):
            out = self._call("find_anomalies", {"n": 8}, steps)
            t = pd.DataFrame(out)
            return AgentResult("The Isolation Forest flags these districts as statistically unusual "
                               "(worth verifying for boundary changes or enumeration issues):", steps, table=t)
        # 5. threshold filter
        m = re.search(r"(above|over|more than|greater than|>|below|under|less than|<)\s*(\d+(?:\.\d+)?)", ql)
        if m:
            op = ">" if m.group(1) in ("above", "over", "more than", "greater than", ">") else "<"
            val = float(m.group(2))
            out = self._call("filter_districts", {"indicator": ind, "operator": op, "value": val, "province": prov}, steps)
            t = pd.DataFrame(out)
            return AgentResult(f"**{len(t)} districts** {'in ' + prov + ' ' if prov else ''}have "
                               f"{LABELS.get(ind, ind)} {op} {val:g}.", steps, table=t)
        # 6. ranking
        if re.search(r"top|highest|most|largest|biggest|fastest|lowest|least|smallest|slowest|rank|which", ql):
            order = "asc" if re.search(r"lowest|least|smallest|slowest|bottom|fewest", ql) else "desc"
            if "rural" in ql and ind == "urban_pct":
                order = "asc" if order == "desc" else "desc"
            out = self._call("rank_districts", {"indicator": ind, "n": n, "order": order, "province": prov}, steps)
            t = pd.DataFrame(out)
            word = "Lowest" if order == "asc" else "Top"
            ans = (f"**{word} {len(t)} districts{' in ' + prov if prov else ''} by {LABELS.get(ind, ind)}** - "
                   f"#1 is **{t.iloc[0]['district']}** ({t.iloc[0][ind]:,.2f}).")
            return AgentResult(ans, steps, table=t,
                               chart={"type": "bar", "x": list(t["district"]), "y": list(t[ind]), "title": LABELS.get(ind, ind)})
        # 7. province
        if prov:
            p = self._call("province_profile", {"province": prov}, steps)
            ans = (f"**{prov}**: {A.fmt_pop(p['pop_2023'])} people ({p['share_pct']:.1f}% of Pakistan) in {int(p['districts'])} districts. "
                   f"Growth {p['growth_rate']:.2f}%/yr, {p['urban_pct']:.1f}% urban, density {p['density']:,.0f}/km², "
                   f"household size {p['avg_hh_size']:.2f}, {p['females_per_100_males']:.1f} females per 100 males, "
                   f"literacy {p['literacy_rate']:.1f}% (male {p['literacy_male']:.1f}% / female {p['literacy_female']:.1f}%).")
            return AgentResult(ans, steps)
        if re.search(r"province|compare provinces|all provinces", ql):
            out = self._call("province_profile", {}, steps)
            return AgentResult("Province-level summary (PBS Census 2023):", steps, table=pd.DataFrame(out))
        # 8. default: national
        nat = self._call("national_overview", {}, steps)
        ans = (f"**Pakistan (Census 2023)**: {A.fmt_pop(nat['population'])} people in {nat['districts']} districts, "
               f"{nat['households']:,} households. Growth since 2017: {nat['growth_rate']:.2f}%/yr. "
               f"Urban {nat['urban_pct']:.1f}% · density {nat['density']:.0f}/km² · literacy {nat['literacy_rate']:.1f}% · "
               f"{nat['females_per_100_males']:.1f} females per 100 males.\n\n"
               "_Try: “top 10 most dense districts”, “compare Lahore and Karachi Central”, "
               "“project Peshawar to 2035 at 3%”, “districts in Sindh with growth above 3”, “any anomalies?”_")
        return AgentResult(ans, steps)

    def _attach_visual(self, res: AgentResult):
        """Turn the last tool result into a table/chart for the UI (LLM mode)."""
        if not res.steps:
            return
        last = res.steps[-1]
        r = last["result"]
        if isinstance(r, list) and r and isinstance(r[0], dict):
            res.table = pd.DataFrame(r)
        if last["tool"] == "project_population" and isinstance(r, dict) and "projections" in r:
            res.chart = {"type": "bar", "x": ["2017", "2023"] + list(r["projections"]),
                         "y": [r["pop_2017"], r["pop_2023"]] + list(r["projections"].values()),
                         "title": f"{r['district']} population"}


# =========================================================== narrative helper
def llm_narrative(facts: list[str], task: str) -> tuple[str, str]:
    """Ask the LLM to turn verified facts into an executive narrative.
    Falls back to joining the facts. Returns (text, mode)."""
    if not llm_available():
        return " ".join(facts), "offline"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        r = client.chat.completions.create(
            model=LLM_MODEL, temperature=0.3,
            messages=[{"role": "system", "content": "You are a demographic policy analyst. Use ONLY the facts given; do not add numbers."},
                      {"role": "user", "content": f"Task: {task}\n\nFacts:\n- " + "\n- ".join(facts)}])
        return r.choices[0].message.content, f"LLM ({LLM_MODEL})"
    except Exception as e:  # noqa: BLE001
        return " ".join(facts) + f" (LLM error: {str(e)[:80]})", "offline"


PROVINCES = PROVINCE_ORDER
