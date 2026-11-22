import json

import plotly.graph_objects as go
import streamlit as st

from core.agent import CensusAgent
from core.config import LLM_MODEL, llm_available
from core.ui import get_df, page_header, style

df = get_df()
page_header("Agentic AI", "AI Census Analyst",
            "Ask questions in plain English. The agent plans, calls data tools on the real census, and answers with "
            "verified numbers - every step is shown for transparency.")

if "agent" not in st.session_state or st.session_state.get("agent_df_id") != id(df):
    st.session_state["agent"] = CensusAgent(df)
    st.session_state["agent_df_id"] = id(df)
agent: CensusAgent = st.session_state["agent"]
hist = st.session_state.setdefault("chat", [])

c1, c2 = st.columns([3, 1])
with c2:
    mode = f"🟢 LLM: {LLM_MODEL}" if llm_available() else "🟡 Offline planner"
    st.markdown(f"**Brain:** {mode}")
    st.caption("Tools: national_overview · province_profile · district_profile · rank_districts · "
               "compare_districts · project_population · filter_districts · find_anomalies")
    st.markdown("**Try:**")
    examples = ["Give me a national overview", "Top 10 most densely populated districts",
                "Which districts in Sindh are most urbanized?", "Compare Lahore, Faisalabad and Rawalpindi",
                "Project Peshawar population to 2035 at 3%", "Districts in Balochistan with growth above 4",
                "Lowest female literacy districts in KP", "Are there any anomalies in the data?"]
    for e in examples:
        if st.button(e, key="ex_" + e, width="stretch"):
            st.session_state["agent_prefill"] = e
    if st.button("🗑️ Clear chat", width="stretch"):
        hist.clear()
        st.rerun()


def render(msg):
    st.markdown(msg["content"])
    if msg.get("table") is not None:
        st.dataframe(msg["table"], hide_index=True, width="stretch")
    if msg.get("chart"):
        ch = msg["chart"]
        fig = go.Figure(go.Bar(x=ch["x"], y=ch["y"], marker_color="#7c3aed",
                               hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>"))
        fig.update_layout(yaxis_tickformat="~s")
        st.plotly_chart(style(fig, 320, ch["title"]), width="stretch", key=f"ch_{id(msg)}")
    if msg.get("steps"):
        with st.expander(f"🧠 Agent reasoning - {len(msg['steps'])} tool call(s) · {msg.get('mode', '')}"):
            for i, s in enumerate(msg["steps"], 1):
                st.markdown(f"**Step {i} → `{s['tool']}`**")
                st.code(json.dumps(s["args"], default=str), language="json")
                preview = json.dumps(s["result"], default=str)
                st.caption("Result: " + (preview[:600] + "…" if len(preview) > 600 else preview))


with c1:
    if not hist:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("Assalam-o-Alaikum! I'm your census analyst. Ask me about any district, province or trend - "
                        "rankings, comparisons, projections or anomalies.")
    for m in hist:
        with st.chat_message(m["role"], avatar="🤖" if m["role"] == "assistant" else "🧑"):
            render(m)
q = st.chat_input("Ask about Pakistan's census 2023…")  # top level -> pinned to the bottom
q = q or st.session_state.pop("agent_prefill", None)
with c1:
    if q:
        hist.append({"role": "user", "content": q})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Planning and calling tools…"):
                r = agent.ask(q, [{"role": h["role"], "content": h["content"]} for h in hist[:-1]])
            msg = {"role": "assistant", "content": r.answer, "table": r.table, "chart": r.chart,
                   "steps": r.steps, "mode": r.mode}
            render(msg)
        hist.append(msg)
