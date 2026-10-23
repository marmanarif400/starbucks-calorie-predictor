
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Starbucks Calorie Predictor",
    page_icon="☕",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "starbucks_decision_tree.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

st.title("☕ Starbucks Calorie Predictor")
st.write(
    "Upload beverage feature data to estimate calories "
    "using a trained Decision Tree regression model."
)

try:
    artifact = load_model()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

st.success("Model loaded successfully!")
st.write("**Model:**", artifact.get("model_type", "Decision Tree Regressor"))

uploaded_file = st.file_uploader(
    "Upload a CSV file containing model input features",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    missing = [
        column for column in feature_columns
        if column not in data.columns
    ]

    if missing:
        st.error(f"Missing required columns: {missing}")
    elif st.button("Predict Calories"):
        try:
            predictions = model.predict(data[feature_columns])

            results = data.copy()
            results["Predicted Calories"] = predictions.round(2)

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download predictions",
                results.to_csv(index=False).encode("utf-8"),
                file_name="starbucks_predictions.csv",
                mime="text/csv"
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")
