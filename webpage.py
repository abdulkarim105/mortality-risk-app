import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Page config + styling
# -----------------------------
st.set_page_config(page_title="Mortality Risk Predictor (XGBoost)", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 3.6rem; padding-bottom: 2rem; }
      .app-title { font-size: 2.1rem; font-weight: 800; margin-top: .8rem; margin-bottom: .25rem; line-height: 1.2; }
      .app-subtitle { color: #6b7280; margin-top: 0; margin-bottom: 1.2rem; }
      .card {
        border: 1px solid rgba(0,0,0,.08);
        border-radius: 16px;
        padding: 18px 18px;
        background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,.04);
      }
      .muted { color: #6b7280; font-size: .95rem; }
      .divider { height: 1px; background: rgba(0,0,0,.06); margin: 14px 0; }
      .kpi { display:flex; gap:14px; flex-wrap:wrap; margin-top: 10px; }
      .kpi-box{
        border: 1px solid rgba(0,0,0,.08);
        border-radius: 14px;
        padding: 12px 14px;
        min-width: 190px;
        background: rgba(249,250,251,1);
      }
      .kpi-label{ font-size: .85rem; color:#6b7280; }
      .kpi-value{ font-size: 1.55rem; font-weight: 800; margin-top: 2px; }
      .stButton>button { border-radius: 12px; padding: 0.60rem 1rem; font-weight: 750; }
      .stNumberInput input { border-radius: 12px !important; }
      .small-note { font-size: .85rem; color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Files (must be in same folder as this webpage.py)
# -----------------------------
MODEL_FILENAME = "mortality_xgboost_pipeline.joblib"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# -----------------------------
# Load pipeline object
# -----------------------------
@st.cache_resource
def load_obj():
    return joblib.load(MODEL_PATH)

obj = load_obj()
pipeline = obj["pipeline"]
feature_cols = obj["feature_cols"]
default_threshold = float(obj.get("threshold", 0.5))

# Pipeline steps
imputer = pipeline.named_steps["imputer"]
xgb_model = pipeline.named_steps["model"]

# -----------------------------
# SHAP explainer (cached) - robust to probability/log-odds
# -----------------------------
@st.cache_resource
def build_shap_explainer(_xgb_model):
    """
    Prefer probability explanation if supported; otherwise fallback.
    Some SHAP versions / model combos don't support model_output="probability".
    """
    try:
        expl = shap.TreeExplainer(_xgb_model, model_output="probability")
        shap_units = "probability"
        return expl, shap_units
    except Exception:
        expl = shap.TreeExplainer(_xgb_model)
        shap_units = "model output (often log-odds)"
        return expl, shap_units

explainer, shap_units = build_shap_explainer(xgb_model)

def _extract_shap_values(expl_result) -> np.ndarray:
    """
    Return SHAP values as numpy array, robust to:
      - shap.Explanation (preferred modern API)
      - older arrays / list-of-arrays for multiclass
    """
    # New API: shap.Explanation
    if hasattr(expl_result, "values"):
        values = expl_result.values
    else:
        values = expl_result

    # Sometimes values is a list (e.g., multiclass). For binary some versions return 2 arrays.
    if isinstance(values, list):
        # Prefer positive class if present
        if len(values) == 2:
            values = values[1]
        else:
            # Fallback: take last class
            values = values[-1]

    values = np.asarray(values)

    # Ensure 2D
    if values.ndim == 1:
        values = values.reshape(1, -1)

    return values

def compute_shap_values_single_row(X_user_df: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for one-row input.
    IMPORTANT: apply the imputer transform first because SHAP must see the same
    matrix that the underlying XGBoost model sees.
    """
    X_user_df = X_user_df[feature_cols]
    X_imp = imputer.transform(X_user_df)

    # Preferred modern call: explainer(X)
    try:
        exp = explainer(X_imp)
        sv = _extract_shap_values(exp)
    except Exception:
        # Older SHAP versions
        sv = _extract_shap_values(explainer.shap_values(X_imp))

    return sv[0]  # 1 row

# -----------------------------
# Helpers
# -----------------------------
def risk_band(p: float):
    if p < 0.30:
        return "Low", "🟢"
    elif p < 0.70:
        return "Moderate", "🟠"
    return "High", "🔴"

def build_input_df(input_dict: dict) -> pd.DataFrame:
    """
    Create 1-row DataFrame with correct feature order.
    Missing columns -> NaN, handled by pipeline imputer.
    """
    X = pd.DataFrame([input_dict])
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_cols]

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="app-title">Mortality Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">XGBoost pipeline (Median Imputation + XGBoost) — '
    'Enter patient values, predict mortality probability, and explain with SHAP.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Session state
# -----------------------------
if "last_X_user" not in st.session_state:
    st.session_state.last_X_user = None
if "last_prob" not in st.session_state:
    st.session_state.last_prob = None
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "show_shap" not in st.session_state:
    st.session_state.show_shap = False

# -----------------------------
# Sidebar inputs (FIX: allow Missing instead of forcing 0.0)
# -----------------------------
with st.sidebar:
    st.header("Patient Inputs")
    threshold = st.slider("Classification threshold", 0.05, 0.95, default_threshold, 0.01)

    st.markdown("<div class='small-note'>Tip: mark a feature as <b>Missing</b> if you don't know it. "
                "The pipeline will median-impute it.</div>", unsafe_allow_html=True)

    st.subheader("Clinical features")

    input_data = {}
    # Create compact UI: each feature has (Missing checkbox + number input)
    for c in feature_cols:
        colA, colB = st.columns([0.42, 0.58])
        with colA:
            is_missing = st.checkbox(f"{c} missing", value=False, key=f"miss__{c}")
        with colB:
            val = st.number_input(c, value=0.0, step=0.1, format="%.3f", key=f"val__{c}", disabled=is_missing)

        input_data[c] = np.nan if is_missing else float(val)

    run_pred = st.button("Predict Risk", use_container_width=True)

# Build input DataFrame
X_user = build_input_df(input_data)

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.25, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Inputs summary")
    st.dataframe(X_user, use_container_width=True, hide_index=True)
    st.caption("Missing values are handled by the pipeline (median imputation).")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Result")

    if run_pred:
        prob = float(pipeline.predict_proba(X_user)[0, 1])
        pred = int(prob >= threshold)

        st.session_state.last_X_user = X_user
        st.session_state.last_prob = prob
        st.session_state.last_pred = pred

    if st.session_state.last_prob is None:
        st.markdown("<div class='muted'>Click <b>Predict Risk</b> to generate results.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        prob = float(st.session_state.last_prob)
        pred = int(st.session_state.last_pred)
        band, icon = risk_band(prob)

        st.markdown(
            f"""
            <div class="kpi">
              <div class="kpi-box">
                <div class="kpi-label">Predicted probability</div>
                <div class="kpi-value">{prob:.3f}</div>
              </div>
              <div class="kpi-box">
                <div class="kpi-label">Risk band</div>
                <div class="kpi-value">{icon} {band}</div>
              </div>
              <div class="kpi-box">
                <div class="kpi-label">Class (threshold {threshold:.2f})</div>
                <div class="kpi-value">{pred}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Risk gauge")
        st.progress(min(max(prob, 0.0), 1.0))

        if band == "Low":
            st.success("Lower risk band based on the model output.")
        elif band == "Moderate":
            st.warning("Moderate risk band based on the model output.")
        else:
            st.error("Higher risk band based on the model output.")

        # -----------------------------
        # SHAP (stateful)
        # -----------------------------
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Explain prediction (SHAP)", use_container_width=True):
                st.session_state.show_shap = True
        with c2:
            if st.button("Hide explanation", use_container_width=True):
                st.session_state.show_shap = False

        if st.session_state.show_shap:
            st.markdown("### Explainability (SHAP)")

            X_to_explain = st.session_state.last_X_user
            if X_to_explain is None:
                st.warning("Please click **Predict Risk** first, then click **Explain prediction (SHAP)**.")
            else:
                with st.spinner("Computing SHAP explanation..."):
                    sv = compute_shap_values_single_row(X_to_explain)

                    df_shap = pd.DataFrame(
                        {"feature": feature_cols, "shap_value": sv}
                    )
                    df_shap["abs"] = df_shap["shap_value"].abs()
                    df_top = df_shap.sort_values("abs", ascending=False).head(20)

                    st.markdown("#### Top contributors (|SHAP|)")
                    st.dataframe(df_top[["feature", "shap_value"]], use_container_width=True, hide_index=True)

                    fig = plt.figure()
                    plt.barh(df_top["feature"][::-1], df_top["shap_value"][::-1])
                    plt.xlabel(f"SHAP value ({shap_units})")
                    plt.title("Top feature impacts for this patient")
                    plt.tight_layout()
                    st.pyplot(fig)

                st.caption(
                    "Note: SHAP values explain the XGBoost model output on the imputed input. "
                    "If 'probability' is not supported, SHAP may explain log-odds / margin instead."
                )
        else:
            st.caption("Click **Explain prediction (SHAP)** to compute explanations.")

    st.markdown("</div>", unsafe_allow_html=True)
