import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import ml
from core.ui import get_df, insight, kpi_row, page_header, style

df = get_df()
page_header("Module 7", "Population Growth Prediction - ML Model Lab",
            "Train and compare regression models that learn each district's growth rate, then convert it to population.")

with st.expander("📘 How the model is framed (read this before the viva)", expanded=False):
    st.markdown("""
- **Data:** 136 districts with population in **2017 and 2023**.
- **Target:** the district's **annual growth rate** (2017→2023). Population is then `pop × (1+rate)^years`.
  *Why not predict population directly?* 2023 population is ~99% explained by 2017 population, which would be a trivial model.
  Predicting the rate forces the model to learn real demographic drivers.
- **Features:** current population (log), urban %, household size, area (log), density (log), province (encoded),
  literacy rate, youth share (0-14).
- **Split:** 80% train / 20% test + **5-fold cross-validation** on the training part. Extreme rates (top/bottom 2%) are
  clipped so boundary changes don't distort training.
- **Metrics:** MAE & R² on the growth rate *and* on the resulting 2023 population, plus MAPE. A naive baseline
  (every district grows at the average rate) is shown for comparison.
- **Models:** Linear Regression, Ridge, Random Forest, Gradient Boosting. The best by CV-MAE is refit on all data and saved with
  **joblib** to `models/best_growth_model.joblib`.
""")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
with c2:
    seed = st.number_input("Random seed", 0, 9999, 42)
with c3:
    st.write("")
    st.write("")
    go_btn = st.button("🚀 Train & compare models", type="primary", width="stretch")

if "ml_result" not in st.session_state and not go_btn:
    cached = ml.cached_result(df, test_size, int(seed))
    if cached is not None:
        st.session_state["ml_result"] = cached
if go_btn or "ml_result" not in st.session_state:
    with st.spinner("Training 4 models with 5-fold cross-validation (~10 s)..."):
        st.session_state["ml_result"] = ml.train_and_compare(df, test_size=test_size, seed=int(seed))
res = st.session_state["ml_result"]
lb = res.leaderboard
best = lb.iloc[0]
base = lb.attrs.get("baseline_mae_rate", float("nan"))

kpi_row([
    ("Best model", res.best_name, "lowest cross-validated MAE", "#ec4899"),
    ("MAE - growth rate", f"{best['Test MAE (rate, pp)']:.2f} pp", f"naive baseline {base:.2f} pp", "#7c3aed"),
    ("R² - population", f"{best['Test R² (population)']:.4f}", f"MAE {best['Test MAE (population)']:,.0f} people", "#14b8a6"),
    ("MAPE - population", f"{best['Test MAPE (population %)']:.2f}%", "average % error on test districts", "#f97316"),
])
st.write("")
st.markdown("##### 🏁 Model leaderboard")
st.dataframe(lb.style.format({"CV MAE (rate, pp)": "{:.3f}", "Test MAE (rate, pp)": "{:.3f}", "Test R² (rate)": "{:.3f}",
                              "Test MAE (population)": "{:,.0f}", "Test R² (population)": "{:.4f}",
                              "Test MAPE (population %)": "{:.2f}"})
             .highlight_min(subset=["CV MAE (rate, pp)"], color="#e6f6ef"), hide_index=True, width="stretch")

c1, c2 = st.columns(2)
tf = res.test_frame
with c1:
    fig = go.Figure()
    mx = max(tf["pop_2023"].max(), tf["pred_pop_2023"].max()) * 1.05
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#9a9993", dash="dash"), name="Perfect prediction")
    fig.add_scatter(x=tf["pop_2023"], y=tf["pred_pop_2023"], mode="markers", name="Test district",
                    marker=dict(size=10, color="#7c3aed", line=dict(color="#fff", width=1.5)), text=tf["district"],
                    hovertemplate="<b>%{text}</b><br>Actual %{x:,.0f}<br>Predicted %{y:,.0f}<extra></extra>")
    fig.update_layout(xaxis_title="Actual 2023 population", yaxis_title="Predicted", xaxis_tickformat="~s",
                      yaxis_tickformat="~s")
    st.plotly_chart(style(fig, 420, f"Predicted vs actual population - test set ({res.best_name})"), width="stretch")
with c2:
    fig = go.Figure()
    fig.add_scatter(x=tf["growth_rate"], y=tf["pred_rate"], mode="markers", text=tf["district"],
                    marker=dict(size=10, color="#ec4899", line=dict(color="#fff", width=1.5)),
                    hovertemplate="<b>%{text}</b><br>Actual %{x:.2f}%<br>Predicted %{y:.2f}%<extra></extra>", name="District")
    lo, hi = min(tf["growth_rate"].min(), tf["pred_rate"].min()), max(tf["growth_rate"].max(), tf["pred_rate"].max())
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#9a9993", dash="dash"), name="Perfect")
    fig.update_layout(xaxis_title="Actual growth %/yr", yaxis_title="Predicted growth %/yr", showlegend=False)
    st.plotly_chart(style(fig, 420, "Predicted vs actual growth rate - the harder task"), width="stretch")

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(res.importances, x="importance", y="feature", orientation="h", color_discrete_sequence=["#14b8a6"])
    fig.update_layout(xaxis_title="Gini importance", yaxis_title=None)
    st.plotly_chart(style(fig, 380, "Feature importance - Random Forest"), width="stretch")
with c2:
    fig = px.bar(res.permutation, x="importance", y="feature", orientation="h", color_discrete_sequence=["#ec4899"])
    fig.update_layout(xaxis_title="Increase in error when shuffled", yaxis_title=None)
    st.plotly_chart(style(fig, 380, f"Permutation importance - {res.best_name} (explainable AI)"), width="stretch")

topf = res.importances.iloc[-1]["feature"]
insight(f"The model's strongest signal is <b>{topf}</b>. Growth-rate R² is modest (~0.3-0.5) because six years of "
        f"district growth also depend on migration and boundary changes not in the census tables - this is honest, "
        f"expected behaviour on 136 records. On the population scale the error is ~{best['Test MAPE (population %)']:.1f}%.")
bundle = ml.load_model()
if bundle:
    st.success(f"💾 Saved model: **{bundle['name']}** · trained {bundle['trained_at']} · `models/best_growth_model.joblib` "
               "- reloaded by the Projector and the AI agent without retraining.")
