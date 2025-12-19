import io
import requests
import joblib
import pandas as pd
import streamlit as st

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Predictive Maintenance — Engine Health",
    page_icon="⚙️",
    layout="centered"
)

# ---------------------------
# Hugging Face SPACE config (Option 1)
# ---------------------------
HF_SPACE_REPO_ID = "Yashwanthsairam/predictive-maintenance-mlops"
BASE_URL = "https://huggingface.co/spaces/{repo}/resolve/main".format(
    repo=HF_SPACE_REPO_ID
)

MODEL_URL   = "{base}/artifacts/model.pkl".format(base=BASE_URL)
METRICS_URL = "{base}/artifacts/metrics.json".format(base=BASE_URL)

# ---------------------------
# Model loader (cached, URL-based)
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_model_from_url(url: str):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return joblib.load(io.BytesIO(resp.content))

# ---------------------------
# UI
# ---------------------------
st.title("⚙️ Predictive Maintenance — Engine Health")
st.write(
    "Predict whether an engine is **Healthy** or **At Risk** "
    "based on sensor readings."
)

# ---------------------------
# Input form
# ---------------------------
st.subheader("Engine Sensor Inputs")

c1, c2 = st.columns(2)

with c1:
    rpm = st.number_input("RPM", 0, 10000, 2500)
    coolant_temp = st.number_input("Coolant Temperature (°C)", -50, 200, 90)
    oil_pressure = st.number_input("Oil Pressure (psi)", 0.0, 200.0, 55.0)
    vibration = st.number_input("Vibration Level", 0.0, 100.0, 12.5)

with c2:
    fuel_rate = st.number_input("Fuel Consumption Rate", 0.0, 100.0, 15.0)
    engine_load = st.slider("Engine Load (%)", 0, 100, 65)
    ambient_temp = st.number_input("Ambient Temperature (°C)", -50, 60, 30)
    runtime_hours = st.number_input("Engine Runtime Hours", 0, 100000, 1500)

# ---------------------------
# Build input DataFrame
# ---------------------------
input_df = pd.DataFrame([{
    "RPM": rpm,
    "Coolant_Temperature": coolant_temp,
    "Oil_Pressure": oil_pressure,
    "Vibration": vibration,
    "Fuel_Rate": fuel_rate,
    "Engine_Load": engine_load,
    "Ambient_Temperature": ambient_temp,
    "Runtime_Hours": runtime_hours
}])

st.markdown("#### Input Preview")
st.dataframe(input_df, use_container_width=True)

# ---------------------------
# Load model
# ---------------------------
with st.spinner("Loading model from Hugging Face Space…"):
    model = load_model_from_url(MODEL_URL)
    st.success("Model loaded successfully from Space")

# ---------------------------
# Prediction helper
# ---------------------------
def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(df)
    proba = model.predict_proba(df)[:, 1]

    out = df.copy()
    out["failure_probability"] = proba
    out["failure_prediction"] = preds
    return out

# ---------------------------
# Actions
# ---------------------------
a, b = st.columns(2)

with a:
    if st.button("🔮 Predict Engine Health"):
        try:
            result = predict_df(input_df)
            pred = int(result.loc[0, "failure_prediction"])
            prob = float(result.loc[0, "failure_probability"])

            status = "⚠️ At Risk" if pred == 1 else "✅ Healthy"
            st.subheader("Prediction Result")
            st.success("{status} — Failure Probability: **{p:.3f}**".format(
                status=status, p=prob
            ))
        except Exception as e:
            st.error("Prediction failed: {e}".format(e=e))

with b:
    uploaded = st.file_uploader(
        "📦 Batch Prediction — Upload CSV (same schema, no target)",
        type=["csv"]
    )

    if uploaded and st.button("Run Batch Prediction"):
        try:
            batch_df = pd.read_csv(io.BytesIO(uploaded.read()))
            res = predict_df(batch_df)

            st.success("Batch prediction completed.")
            st.dataframe(res.head(50), use_container_width=True)

            st.download_button(
                "⬇️ Download Predictions",
                data=res.to_csv(index=False),
                file_name="engine_predictions.csv"
            )
        except Exception as e:
            st.error("Batch prediction failed: {e}".format(e=e))

# ---------------------------
# Footer
# ---------------------------
st.caption(
    "Model is loaded directly from the same Hugging Face Space "
    "using URL-based artifacts."
)
