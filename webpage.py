import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Mortality Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling (professional + title not clipped)
# -----------------------------
st.markdown(
    """
    <style>
      :root{
        --card-border: rgba(0,0,0,.08);
        --muted: #6b7280;
        --bg-soft: rgba(249,250,251,1);
      }
      .block-container { padding-top: 3.4rem; padding-bottom: 2rem; max-width: 1280px; }
      .topbar{
        display:flex; flex-wrap: wrap; align-items:flex-end; justify-content:space-between;
        gap: 12px; margin-bottom: 14px;
      }
      .title-wrap{ flex: 1 1 520px; min-width: 320px; }
      .app-title{
        font-size: 2.2rem; font-weight: 850; line-height: 1.25; margin: 0; padding-top: 2px;
        overflow-wrap: anywhere; word-break: break-word;
      }
      .app-subtitle{
        color: var(--muted); margin: 0; margin-top: 6px; font-size: 1.02rem;
        overflow-wrap: anywhere; word-break: break-word;
      }
      .chip{
        flex: 0 0 auto; max-width: 100%; white-space: nowrap;
        display:inline-flex; align-items:center; gap:8px;
        border: 1px solid var(--card-border);
        border-radius: 999px; padding: 8px 12px; background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,.04);
        font-size: .92rem; color: var(--muted);
      }
      .chip b { color: #111827; font-weight: 750; }

      .card{
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 16px 16px;
        background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,.04);
      }
      .muted{ color: var(--muted); }

      .soft{
        border: 1px solid var(--card-border);
        background: var(--bg-soft);
        border-radius: 14px;
        padding: 12px 14px;
      }

      .kpis{
        display:grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
      }
      .kpi{
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 12px 14px;
        background: var(--bg-soft);
      }
      .kpi .label{ color: var(--muted); font-size: .86rem; }
      .kpi .value{ font-size: 1.55rem; font-weight: 850; margin-top: 2px; }
      .kpi .sub{ color: var(--muted); font-size: .86rem; margin-top: 2px; }

      .hr{ height:1px; background: rgba(0,0,0,.06); margin: 12px 0; }

      .stButton>button{ border-radius: 12px; padding: 0.60rem 1rem; font-weight: 750; }
      .stNumberInput input, .stTextInput input{ border-radius: 12px !important; }
      .stDataFrame{ border-radius: 14px; overflow: hidden; }

      section[data-testid="stSidebar"] .block-container{ padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Files
# -----------------------------
MODEL_FILENAME = "mortality_xgboost_pipeline.joblib"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# -----------------------------
# Feature ranges (from your screenshot)
# If value is outside -> show "not valid" message and block prediction
# -----------------------------
FEATURE_RANGES = {
    "GCS_max": (0.0, 8.0),
    "GCS_mean": (0.0, 8.0),
    "Lactate_min": (0.0, 25.0),
    "Lactate_max": (0.0, 35.0),
    "Lactate_mean": (0.0, 30.0),
    "BUN_min": (0.0, 200.0),
    "BUN_mean": (0.0, 220.0),
    "Bilirubin_max": (0.0, 85.0),
    "Bilirubin_mean": (0.0, 75.0),
    "AG_MEAN": (0.0, 45.0),
    "AG_MAX": (0.0, 55.0),
    "AG_MEDIAN": (0.0, 50.0),
    "AG_MIN": (0.0, 40.0),
    "AG_STD": (0.0, 20.0),
    "SYSBP_MIN": (0.0, 160.0),
    "SYSBP_MEAN": (0.0, 190.0),
    "SYSBP_STD": (0.0, 55.0),
    "DIASBP_MIN": (0.0, 90.0),
    "DIASBP_MEAN": (0.0, 115.0),
    "AGE": (0.0, 95.0),
    "RR_MEAN": (0.0, 45.0),
    "RR_STD": (0.0, 15.0),
    "RR_MAX": (0.0, 65.0),
    "TEMP_STD": (0.0, 5.0),
    "TEMP_MIN": (0.0, 45.0),
    "HR_MEAN": (0.0, 160.0),
    "HR_MAX": (0.0, 250.0),
    "age_adj_comorbidity_score": (0.0, 65.0),
}

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

imputer = pipeline.named_steps["imputer"]
xgb_model = pipeline.named_steps["model"]

# -----------------------------
# Group features (FIXED: AGE will not go to Labs)
# -----------------------------
def group_features(cols):
    groups = {"Vitals": [], "Labs": [], "Scores / Comorbidity": [], "Other": []}

    score_keys = ("GCS", "SOFA", "SAPS", "OASIS", "COMORB", "AGE", "COMORBIDITY", "SCORE")
    vitals_keys = ("HR", "RR", "SBP", "DBP", "MBP", "SYSBP", "DIASBP", "SPO2", "TEMP", "O2", "RESP")

    labs_tokens = {
        "BUN", "CREAT", "WBC", "HGB", "HCT", "PLT", "SOD", "POT", "CHL", "GLU",
        "BILI", "ALT", "AST", "ALB", "LACT", "LACTATE", "BILIRUBIN"
    }

    def is_anion_gap(name: str) -> bool:
        u = name.upper()
        return (u == "AG") or ("AG_" in u) or ("_AG" in u)  # NOT AGE

    for c in cols:
        u = c.upper()

        if any(k in u for k in score_keys):
            groups["Scores / Comorbidity"].append(c)
        elif any(k in u for k in vitals_keys):
            groups["Vitals"].append(c)
        else:
            token_hit = any(tok in u for tok in labs_tokens)
            if token_hit or is_anion_gap(c):
                groups["Labs"].append(c)
            else:
                groups["Other"].append(c)

    return {k: v for k, v in groups.items() if v}

feature_groups = group_features(feature_cols)

# -----------------------------
# SHAP explainer (cached)
# -----------------------------
@st.cache_resource
def build_shap_explainer(_xgb_model):
    return shap.TreeExplainer(_xgb_model)

explainer = build_shap_explainer(xgb_model)

def compute_shap_values_single_row(X_user_df: pd.DataFrame) -> np.ndarray:
    X_imp = imputer.transform(X_user_df[feature_cols])
    sv = explainer.shap_values(X_imp)

    if isinstance(sv, list) and len(sv) == 2:
        sv = sv[1]  # class 1

    sv = np.array(sv)
    if sv.ndim == 2:
        return sv[0]
    return sv

# -----------------------------
# Validation helpers
# -----------------------------
def validate_ranges(X_user_df: pd.DataFrame):
    """
    Return list of (feature, val, lo, hi) for invalid values.
    Ignore NaN (missing).
    """
    invalid = []
    row = X_user_df.iloc[0]
    for feat, (lo, hi) in FEATURE_RANGES.items():
        if feat in row.index:
            v = row[feat]
            if pd.isna(v):
                continue
            try:
                v = float(v)
            except Exception:
                invalid.append((feat, v, lo, hi))
                continue
            if (v < lo) or (v > hi):
                invalid.append((feat, v, lo, hi))
    return invalid

# -----------------------------
# Other helpers
# -----------------------------
def risk_band(p: float):
    if p < 0.30:
        return "Low", "🟢"
    if p < 0.70:
      return "Moderate", "🟡"
    return "High", "🔴"

def interpret(prob: float):
    band, icon = risk_band(prob)
    if band == "Low":
        return icon, band, "Lower risk band based on the model output."
    if band == "Moderate":
        return icon, band, "Moderate risk band based on the model output."
    return icon, band, "Higher risk band based on the model output."

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
# Header
# -----------------------------
st.markdown(
    """
    <div class="topbar">
      <div class="title-wrap">
        <div class="app-title">Mortality Risk Predictor</div>
        <div class="app-subtitle">
          XGBoost pipeline (Median Imputation + XGBoost).
          Patient-level probability + SHAP explanation.
        </div>
      </div>
      <div class="chip">Model: <b>XGBoost</b> • Output: <b>Probability</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar inputs (search + groups + missing + RANGE LIMITS)
# -----------------------------
with st.sidebar:
    st.header("Patient Inputs")

    threshold = st.slider(
        "Classification threshold",
        0.05,
        0.95,
        default_threshold,
        0.01
    )

    st.markdown("**Find a feature**")
    q = st.text_input(
        "Search",
        value="",
        placeholder="Type to filter (e.g., lactate, gcs, bun)",
        label_visibility="collapsed"
    )

    st.markdown("**Show groups**")
    selected_groups = st.multiselect(
        "Feature groups",
        options=list(feature_groups.keys()),
        default=list(feature_groups.keys()),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Tip: Mark a feature as Missing if you don’t know it. The pipeline will handle it.")

    input_data = {}
    missing_cols = []

    for g in selected_groups:
        cols_in_g = feature_groups.get(g, [])

        if q.strip():
            ql = q.strip().lower()
            cols_in_g = [c for c in cols_in_g if ql in c.lower()]

        if not cols_in_g:
            continue

        with st.expander(g, expanded=(g in ["Vitals", "Labs"])):
            for c in cols_in_g:
                colA, colB = st.columns([0.42, 0.58])

                with colA:
                    is_missing = st.checkbox(f"{c} missing", value=False, key=f"miss_{c}")

                # apply range if we have it
                lo_hi = FEATURE_RANGES.get(c, None)
                min_v = float(lo_hi[0]) if lo_hi else None
                max_v = float(lo_hi[1]) if lo_hi else None

                with colB:
                    kwargs = dict(
                        value=0.0,
                        step=0.1,
                        format="%.3f",
                        key=f"val_{c}",
                        disabled=is_missing
                    )
                    if min_v is not None:
                        kwargs["min_value"] = min_v
                    if max_v is not None:
                        kwargs["max_value"] = max_v

                    val = st.number_input(c, **kwargs)

                if is_missing:
                    input_data[c] = np.nan
                    missing_cols.append(c)
                else:
                    input_data[c] = float(val)

    # Any feature not shown -> NaN (pipeline imputes)
    for c in feature_cols:
        if c not in input_data:
            input_data[c] = np.nan

    st.markdown("---")
    run_pred = st.button("Predict Risk", use_container_width=True)

# Build input DF in correct order
X_user = pd.DataFrame([input_data])[feature_cols]

# -----------------------------
# Tabs
# -----------------------------
tab_pred, tab_shap, tab_about = st.tabs([
    "📈 Risk Prediction",
    "🧠 Model Explainability",
    "📄 About the Model"
])

# -----------------------------
# Prediction tab
# -----------------------------
with tab_pred:
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Patient snapshot")

        with st.expander("View input table (all features)", expanded=False):
            st.dataframe(X_user, use_container_width=True, hide_index=True)

        if len(missing_cols) > 0:
            st.markdown(
                f"<div class='soft muted'><b>Marked as missing:</b> {', '.join(missing_cols[:20])}"
                + (" ..." if len(missing_cols) > 20 else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='soft muted'>No features were marked as missing.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Risk output")

        if run_pred:
            # --------- RANGE VALIDATION (BLOCK prediction if invalid) ----------
            invalid = validate_ranges(X_user)
            if invalid:
                # show one clear message + details
                st.error("Some inputs are out of valid range. Please check your input values.")
                for feat, v, lo, hi in invalid[:12]:
                    st.warning(f"❌ {feat} is not valid: {v} (valid range: {lo} to {hi})")
                if len(invalid) > 12:
                    st.info(f"...and {len(invalid) - 12} more invalid fields.")
            else:
                prob = float(pipeline.predict_proba(X_user)[0, 1])
                pred = int(prob >= threshold)

                st.session_state.last_X_user = X_user
                st.session_state.last_prob = prob
                st.session_state.last_pred = pred

        if st.session_state.last_prob is None:
            st.markdown('<div class="muted">Click <b>Predict Risk</b> to generate results.</div>', unsafe_allow_html=True)
        else:
            prob = st.session_state.last_prob
            pred = st.session_state.last_pred
            icon, band, msg = interpret(prob)

            st.markdown(
                f"""
                <div class="kpis">
                  <div class="kpi">
                    <div class="label">Predicted probability</div>
                    <div class="value">{prob:.3f}</div>
                    <div class="sub">P(mortality)</div>
                  </div>
                  <div class="kpi">
                    <div class="label">Risk band</div>
                    <div class="value">{icon} {band}</div>
                    <div class="sub">Rule: &lt;0.30 low, &lt;0.70 moderate</div>
                  </div>
                  <div class="kpi">
                    <div class="label">Class</div>
                    <div class="value">{pred}</div>
                    <div class="sub">Threshold = {threshold:.2f}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            st.markdown("#### Risk gauge")
            st.progress(min(max(prob, 0.0), 1.0))

            if band == "Low":
                st.success(msg)
            elif band == "Moderate":
                st.warning(msg)
            else:
                st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# SHAP tab
# -----------------------------
with tab_shap:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Patient-level explanation (SHAP)")
    st.markdown(
        '<div class="muted">Positive SHAP values increase predicted risk; negative values decrease it. '
        "Explanations are computed for the last predicted patient.</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        explain_btn = st.button("Explain prediction", use_container_width=True)
    with c2:
        hide_btn = st.button("Hide explanation", use_container_width=True)
    with c3:
        top_k = st.slider("Top features to show", 5, 30, 12, 1)

    if hide_btn:
        st.session_state.show_shap = False
    if explain_btn:
        st.session_state.show_shap = True

    if st.session_state.last_X_user is None:
        st.info("Run a prediction first in the **Risk Prediction** tab, then come back here.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        if not st.session_state.show_shap:
            st.caption("Click **Explain prediction** to compute SHAP.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            X_to_explain = st.session_state.last_X_user

            with st.spinner("Computing SHAP explanation..."):
                sv = compute_shap_values_single_row(X_to_explain)
                df_shap = pd.DataFrame({"feature": feature_cols, "shap_value": sv})
                df_shap["abs"] = df_shap["shap_value"].abs()
                df_top = df_shap.sort_values("abs", ascending=False).head(top_k)

            l, r = st.columns([1.15, 1])

            with l:
                st.markdown("#### Top contributors")
                st.dataframe(df_top[["feature", "shap_value"]], use_container_width=True, hide_index=True)

            with r:
                st.markdown("#### Feature impact chart")
                df_plot = df_top.copy()
                fig_height = max(5, 0.35 * len(df_plot) + 1.5)

                fig = plt.figure(figsize=(10, fig_height))
                plt.barh(df_plot["feature"][::-1], df_plot["shap_value"][::-1])
                plt.axvline(0, linewidth=1)
                plt.xlabel("SHAP value  ( + increases risk )")
                plt.ylabel("")
                plt.title("Top feature impacts for this patient", pad=12)
                plt.tight_layout()
                plt.subplots_adjust(left=0.35)
                st.pyplot(fig)

            st.caption("SHAP is computed on demand using TreeExplainer for XGBoost.")
            st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Advanced view: Waterfall plot (single patient)", expanded=False):
    try:
        X_to_explain = st.session_state.last_X_user
        if X_to_explain is None:
            st.info("Run a prediction first, then open this section.")
        else:
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]

            X_imp = imputer.transform(X_to_explain[feature_cols])
            sv = compute_shap_values_single_row(X_to_explain)

            exp = shap.Explanation(
                values=sv,
                base_values=base_value,
                data=X_imp[0],
                feature_names=feature_cols,
            )

            fig2 = plt.figure(figsize=(10, 7))
            shap.plots.waterfall(exp, max_display=20, show=False)
            plt.tight_layout()
            st.pyplot(fig2)

    except Exception as e:
        st.warning(f"Could not render waterfall plot. Details: {e}")

# -----------------------------
# About tab
# -----------------------------
with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### About this app")
    st.markdown(
        """
- **Model:** XGBoost classifier  
- **Preprocessing:** Median imputation (inside the pipeline)  
- **Inputs:** Clinical numeric features only  
- **Output:** Probability of in-hospital mortality + binary class based on threshold  
- **Explainability:** SHAP values (patient-level feature contributions)  
- **Input validation:** Each feature has a valid range; out-of-range values are blocked.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


