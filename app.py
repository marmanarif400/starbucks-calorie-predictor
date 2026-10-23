
import pickle
from pathlib import Path

import numpy as np
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


def get_input_details(model):
    category_options = {}
    numeric_defaults = {}
    categorical_columns = []

    preprocessor = model.named_steps["preprocessor"]

    for name, transformer, columns in preprocessor.transformers_:
        columns = list(columns) if not isinstance(columns, str) else [columns]

        if name == "categorical":
            categorical_columns.extend(columns)

            encoder = transformer.named_steps["encoder"]

            for column, categories in zip(
                columns, encoder.categories_
            ):
                options = [
                    str(value)
                    for value in categories
                    if pd.notna(value)
                ]
                if options:
                    category_options[column] = options

        elif name == "numeric":
            imputer = transformer.named_steps["imputer"]

            for column, value in zip(
                columns, imputer.statistics_
            ):
                if pd.notna(value) and np.isfinite(value):
                    numeric_defaults[column] = max(
                        0.0, float(value)
                    )

    return (
        category_options,
        numeric_defaults,
        categorical_columns
    )


st.title("☕ Starbucks Calorie Predictor")

st.write(
    "Enter beverage details and nutritional information "
    "to estimate calories using a trained Decision Tree model."
)

try:
    artifact = load_model()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    (
        category_options,
        numeric_defaults,
        categorical_columns
    ) = get_input_details(model)

except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

st.success("Model loaded successfully!")

st.caption(
    "Enter actual nutritional values when available. "
    "The numerical fields start with typical training-data medians "
    "as defaults; predictions using unchanged defaults are only estimates."
)

user_inputs = {}

with st.form("calorie_prediction_form"):
    st.subheader("1. Beverage details")

    for column in categorical_columns:
        options = category_options.get(column, [])

        if options:
            user_inputs[column] = st.selectbox(
                column.replace("_", " "),
                options=options
            )
        else:
            user_inputs[column] = st.text_input(
                column.replace("_", " ")
            )

    st.subheader("2. Nutritional information")

    st.write(
        "Edit the values below using the beverage's nutrition label "
        "where possible."
    )

    numeric_columns = [
        column for column in feature_columns
        if column not in categorical_columns
    ]

    left_col, right_col = st.columns(2)

    for index, column in enumerate(numeric_columns):
        default_value = numeric_defaults.get(column, 0.0)

        with (left_col if index % 2 == 0 else right_col):
            user_inputs[column] = st.number_input(
                column,
                min_value=0.0,
                value=float(default_value),
                step=0.5,
                format="%.2f"
            )

    submitted = st.form_submit_button(
        "Predict Calories",
        type="primary",
        use_container_width=True
    )

if submitted:
    try:
        input_data = pd.DataFrame(
            [user_inputs],
            columns=feature_columns
        )

        prediction = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("Prediction result")

        st.metric(
            label="Estimated calories",
            value=f"{prediction:.1f} kcal"
        )

        st.info(
            "This is a machine-learning estimate, not an official "
            "Starbucks nutrition value."
        )

        with st.expander("View the values submitted"):
            display_data = input_data.T.reset_index()
            display_data.columns = ["Feature", "Input value"]
            st.dataframe(display_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
