import os
import io
import re
import joblib
import pandas as pd
import streamlit as st
from typing import List, Optional, Tuple, Dict

from huggingface_hub import hf_hub_download, HfApi

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Predictive Maintenance — Engine Health",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Predictive Maintenance — Engine Health")
st.write("Predict whether an engine is **Healthy** or **At Risk** based on sensor readings.")

# ---------------------------
# Hugging Face settings
# ---------------------------
# You can point to either:
# - a Model Hub repo (repo_type="model")
# - or a Space repo (repo_type="space") where you uploaded artifacts/
MODEL_REPO = os.getenv("MODEL_REPO", "Yashwanthsairam/predictive-maintenance-mlops")
MODEL_REPO_TYPE = os.getenv("MODEL_REPO_TYPE", "space")  # "model" or "space"
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")  # where model files live in the repo

# Optional (only required if repo is private)
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

api = HfApi(token=HF_TOKEN if HF_TOKEN else None)

# ---------------------------
# Helpers
# ---------------------------
def _norm_col(s: str) -> str:
    """Normalize a column name for matching: lower, keep alnum, collapse separators."""
    s = (s or "").strip().lower()
    s = re.sub(r"[%\(\)\[\]\{\}]", "", s)
    s = re.sub(r"[\s\-\/]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _discover_model_candidates(repo_id: str, repo_type: str, artifacts_dir: str) -> List[str]:
    """
    List repo files and return likely model artifact candidates.
    Priority: joblib/pkl under artifacts folder.
    """
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    rfiles = [f.rfilename if hasattr(f, "rfilename") else str(f) for f in files]

    # Focus inside artifacts dir first
    in_artifacts = [f for f in rfiles if f.startswith(artifacts_dir.rstrip("/") + "/")]

    # Candidate extensions
    exts = (".joblib", ".pkl", ".pickle")
    candidates = [f for f in in_artifacts if f.lower().endswith(exts)]

    # If nothing in artifacts, fall back to root scan
    if not candidates:
        candidates = [f for f in rfiles if f.lower().endswith(exts)]

    # Prefer “best” names if present
    preferred_order = []
    for kw in ["best", "xgboost", "model"]:
        preferred_order.extend([f for f in candidates if kw in f.lower()])

    # de-dup while preserving order
    out = []
    for f in (preferred_order + candidates):
        if f not in out:
            out.append(f)
    return out

@st.cache_resource(show_spinner=True)
def load_model_auto(repo_id: str, repo_type: str, artifacts_dir: str) -> Tuple[object, str]:
    """
    Auto-discover and load the model file from HF repo.
    Returns (model, chosen_filename_in_repo).
    """
    candidates = _discover_model_candidates(repo_id, repo_type, artifacts_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No model files found in repo '{repo_id}' (type={repo_type}). "
            f"Expected something like '{artifacts_dir}/model.pkl' or '*.joblib'."
        )

    last_err = None
    for fname in candidates:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type=repo_type,
                token=HF_TOKEN if HF_TOKEN else None,
            )
            model = joblib.load(local_path)
            return model, fname
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to download/load any candidate model file. Last error: {last_err}")

def get_expected_features(model) -> Optional[List[str]]:
    """
    Try to extract feature names the model expects (sklearn style).
    """
    feats = getattr(model, "feature_names_in_", None)
    if feats is None:
        return None
    return list(feats)

def build_synonym_map(expected: List[str]) -> Dict[str, List[str]]:
    """
    Provide a small synonym list for common naming variants.
    We normalize everything before matching, so these are just extra hints.
    """
    # If your model uses different casing/spelling, add more here.
    # Key: expected normalized, Values: possible input normalized.
    # This is intentionally conservative.
    syn = {
        _norm_col("Engine rpm"): [_norm_col("Engine_rpm"), _norm_col("Engine RPM"), _norm_col("RPM"), _norm_col("Engine rpm")],
        _norm_col("Lub oil pressure"): [_norm_col("Lub_oil_pressure"), _norm_col("Lub oil pressure"), _norm_col("Lube_oil_pressure"), _norm_col("Lub_Oil_Pressure")],
        _norm_col("Fuel pressure"): [_norm_col("Fuel_pressure"), _norm_col("Fuel pressure"), _norm_col("Fuel_Pressure")],
        _norm_col("Coolant pressure"): [_norm_col("Coolant_pressure"), _norm_col("Coolant pressure"), _norm_col("Coolant_Pressure")],
        _norm_col("lub oil temp"): [_norm_col("Lub_oil_temp"), _norm_col("Lub oil temp"), _norm_col("Lub_Oil_Temp"), _norm_col("Lub oil temperature")],
        _norm_col("Coolant temp"): [_norm_col("Coolant_temp"), _norm_col("Coolant temp"), _norm_col("Coolant_Temp"), _norm_col("Coolant temperature")],
    }
    # Only keep entries that actually exist in expected (normalized)
    exp_norm = {_norm_col(x) for x in expected}
    return {k: v for k, v in syn.items() if k in exp_norm}

def align_df_to_model(df_in: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """
    Make input DF match model expected columns:
    - normalize column names
    - map synonyms
    - reorder columns
    - error if required columns missing
    """
    df = df_in.copy()

    # Build normalized lookup of incoming columns
    incoming_cols = list(df.columns)
    incoming_norm_map = {_norm_col(c): c for c in incoming_cols}

    expected_norm = [_norm_col(c) for c in expected_features]
    syn_map = build_synonym_map(expected_features)

    rename_map = {}

    for exp_col, exp_n in zip(expected_features, expected_norm):
        # Direct normalized match
        if exp_n in incoming_norm_map:
            rename_map[incoming_norm_map[exp_n]] = exp_col
            continue

        # Synonym match
        matched = False
        for alt_n in syn_map.get(exp_n, []):
            if alt_n in incoming_norm_map:
                rename_map[incoming_norm_map[alt_n]] = exp_col
                matched = True
                break
        if matched:
            continue

    # Apply renames
    df = df.rename(columns=rename_map)

    # Check missing
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise ValueError(
            "Input is missing required columns for this model.\n\n"
            f"Expected: {expected_features}\n"
            f"Got: {list(df_in.columns)}\n"
            f"Missing: {missing}\n\n"
            "Tip: Upload a CSV with the same columns used during training, "
            "or update the UI labels/mapping."
        )

    # Keep only expected + in correct order
    df = df[expected_features].copy()
    return df

def predict_df(model, df_raw: pd.DataFrame) -> pd.DataFrame:
    expected = get_expected_features(model)
    df = df_raw.copy()

    if expected:
        df = align_df_to_model(df, expected)

    preds = model.predict(df)
    out = df_raw.copy()

    # probability is optional
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(df)[:, 1]
            out["failure_probability"] = proba
        except Exception:
            out["failure_probability"] = None

    out["failure_prediction"] = preds
    return out

# ---------------------------
# Load model (auto)
# ---------------------------
with st.spinner("Loading model (auto-discovery) from Hugging Face…"):
    model, chosen_file = load_model_auto(MODEL_REPO, MODEL_REPO_TYPE, ARTIFACTS_DIR)
st.success(f"Model loaded from **{MODEL_REPO}** (`{chosen_file}`)")

expected_features = get_expected_features(model)
if expected_features:
    st.caption(f"Detected model features: {expected_features}")
else:
    st.warning(
        "Could not read model.feature_names_in_. "
        "If you see feature mismatch errors, re-save the model as a sklearn Pipeline or "
        "ensure the training preserved feature names."
    )

# ---------------------------
# UI: Single prediction (dynamic if possible)
# ---------------------------
st.subheader("Single Prediction")

if expected_features:
    # Build numeric inputs dynamically
    vals = {}
    cols = st.columns(2)
    for i, feat in enumerate(expected_features):
        with cols[i % 2]:
            vals[feat] = st.number_input(feat, value=0.0)

    input_df = pd.DataFrame([vals])
else:
    st.info("Single prediction UI is limited because model feature names were not detected.")
    input_df = pd.DataFrame([])

st.markdown("#### Input Preview")
st.dataframe(input_df, use_container_width=True)

if st.button("🔮 Predict Engine Health", type="primary", disabled=input_df.empty):
    try:
        result = predict_df(model, input_df)
        pred = int(result.loc[0, "failure_prediction"])
        prob = result.loc[0, "failure_probability"] if "failure_probability" in result.columns else None

        status = "⚠️ At Risk" if pred == 1 else "✅ Healthy"
        st.subheader("Prediction Result")
        if prob is not None and pd.notna(prob):
            st.success(f"{status} — Failure Probability: **{float(prob):.3f}**")
        else:
            st.success(f"{status}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------
# Batch prediction
# ---------------------------
st.subheader("Batch Prediction — Upload CSV (same schema, no target)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded and st.button("Run Batch Prediction"):
    try:
        batch_df = pd.read_csv(io.BytesIO(uploaded.read()))
        res = predict_df(model, batch_df)

        st.success("Batch prediction completed.")
        st.dataframe(res.head(50), use_container_width=True)

        st.download_button(
            "⬇️ Download Predictions",
            data=res.to_csv(index=False),
            file_name="engine_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

st.caption(
    "If you see feature mismatch errors, it means the CSV/UI columns still don't match the model's training features. "
    "This app tries to auto-align columns, but you may need to adjust mapping for your exact feature names."
)
