import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Starbucks Calories Prediction App
# Place app.py, decision_tree_model.joblib, and
# simple_imputer.joblib in the same GitHub repository folder.
# ------------------------------------------------------------

st.set_page_config(
    page_title="Starbucks Calories Predictor",
    page_icon="☕",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "decision_tree_model.joblib"
IMPUTER_PATH = BASE_DIR / "simple_imputer.joblib"

FEATURES = [
    "Total Fat (g)",
    "Trans Fat (g)",
    "Saturated Fat (g)",
    "Sodium (mg)",
    "Total Carbohydrates (g)",
    "Cholesterol (mg)",
    "Dietary Fibre (g)",
    "Sugars (g)",
    "Protein (g)",
    "Vitamin A (% DV)",
    "Vitamin C (% DV)",
    "Calcium (% DV)",
    "Iron (% DV)",
    "Caffeine (mg)",
]

@st.cache_resource
def load_artifacts():
    """Load the trained model and fitted imputer once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Upload decision_tree_model.joblib to your repository."
        )
    if not IMPUTER_PATH.exists():
        raise FileNotFoundError(
            f"Imputer file not found: {IMPUTER_PATH.name}. "
            "Upload simple_imputer.joblib to your repository."
        )
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, imputer

# Note: decorator must start at the beginning of the line.

st.title("☕ Starbucks Calories Predictor")
st.write(
    "Enter the nutritional values of a Starbucks beverage to estimate its calories "
    "using your trained Decision Tree Regressor."
)

try:
    model, imputer = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "Make sure app.py, decision_tree_model.joblib, and "
        "simple_imputer.joblib are in the same GitHub repository folder."
    )
    st.stop()

labels = {
    "Total Fat (g)": "Total Fat (g)",
    "Trans Fat (g)": "Trans Fat (g)",
    "Saturated Fat (g)": "Saturated Fat (g)",
    "Sodium (mg)": "Sodium (mg)",
    "Total Carbohydrates (g)": "Total Carbohydrates (g)",
    "Cholesterol (mg)": "Cholesterol (mg)",
    "Dietary Fibre (g)": "Dietary Fibre (g)",
    "Sugars (g)": "Sugars (g)",
    "Protein (g)": "Protein (g)",
    "Vitamin A (% DV)": "Vitamin A (% DV)",
    "Vitamin C (% DV)": "Vitamin C (% DV)",
    "Calcium (% DV)": "Calcium (% DV)",
    "Iron (% DV)": "Iron (% DV)",
    "Caffeine (mg)": "Caffeine (mg)",
}

# Sensible initial values; users can change them to match a beverage.
defaults = {
    "Total Fat (g)": 5.0,
    "Trans Fat (g)": 0.0,
    "Saturated Fat (g)": 3.0,
    "Sodium (mg)": 100.0,
    "Total Carbohydrates (g)": 30.0,
    "Cholesterol (mg)": 15.0,
    "Dietary Fibre (g)": 0.0,
    "Sugars (g)": 25.0,
    "Protein (g)": 5.0,
    "Vitamin A (% DV)": 0.0,
    "Vitamin C (% DV)": 0.0,
    "Calcium (% DV)": 10.0,
    "Iron (% DV)": 0.0,
    "Caffeine (mg)": 75.0,
}

with st.form("calorie_prediction_form"):
    st.subheader("Beverage nutrition values")
    st.caption("Enter values in the units shown. Values must be zero or greater.")

    input_values = {}
    left, right = st.columns(2)

    for index, feature in enumerate(FEATURES):
        column = left if index % 2 == 0 else right
        with column:
            input_values[feature] = st.number_input(
                labels[feature],
                min_value=0.0,
                value=float(defaults[feature]),
                step=1.0,
                format="%.2f",
                key=f"input_{index}",
            )

    submitted = st.form_submit_button(
        "Predict Calories", type="primary", use_container_width=True
    )

if submitted:
    try:
        input_df = pd.DataFrame([input_values], columns=FEATURES)
        input_imputed = imputer.transform(input_df)
        prediction = float(model.predict(input_imputed)[0])

        st.success("Prediction completed!")
        st.metric("Estimated Calories", f"{prediction:.1f} kcal")
        st.caption(
            "This is a machine-learning estimate, not an official Starbucks nutrition value."
        )
        with st.expander("Review entered values"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Model: DecisionTreeRegressor · Trained Starbucks nutrition dataset")
