"""
Sri Lankan Used Car Price Predictor (Streamlit app)

Uses model.pkl (the best model from the notebook: preprocessing + model).
The model predicts log(price), so the result is converted back with np.expm1.
"""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

AD_YEAR = 2023   # latest ad year in the data (car_age = AD_YEAR - year of manufacture)

st.set_page_config(page_title="Sri Lanka Used Car Price Predictor", page_icon="🚗")


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_options():
    with open("app_options.json") as f:
        return json.load(f)


model = load_model()
options = load_options()

FEATURES = ["brand", "model", "car_age", "mileage_km", "engine_cc", "condition",
            "transmission", "fuel_type", "body_type", "district", "seller_type"]


@st.cache_resource
def load_explainer():
    est = model.named_steps["model"] if isinstance(model, Pipeline) else model
    return shap.TreeExplainer(est)


def explain(car):
    """SHAP values for one car: one value per original feature, in log(price)."""
    explainer = load_explainer()
    if isinstance(model, Pipeline):
        pre = model.named_steps["preprocess"]
        est = model.named_steps["model"]
        X_pre = pre.transform(car)
        if not isinstance(est, XGBRegressor) and hasattr(X_pre, "toarray"):
            X_pre = X_pre.toarray()     # tree models from scikit-learn need dense data
        values = explainer.shap_values(X_pre)
        values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)

        # Add up the one-hot columns of each original feature
        names = pre.get_feature_names_out()
        grouped = np.zeros(len(FEATURES))
        for j, f in enumerate(FEATURES):
            cols = np.array([n == "num__" + f or n.startswith("cat__" + f + "_") for n in names])
            grouped[j] = values[0, cols].sum()
    else:
        grouped = np.asarray(explainer.shap_values(car))[0]   # CatBoost

    base_value = float(np.ravel(explainer.expected_value)[0])
    return grouped, base_value

st.title("🚗 Sri Lanka Used Car Price Predictor")
st.info(
    "Prices are based on ikman.lk car ads from **Sept 2022 – Apr 2023** "
    "(during the vehicle import ban). Today's prices may be different."
)

# ---------- Inputs ----------
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", options["brands"],
                         index=options["brands"].index("Toyota"))
    car_model = st.selectbox("Model", options["models_by_brand"][brand])
    year = st.number_input("Year of manufacture", min_value=1990, max_value=2024,
                           value=2015, step=1)
    mileage_km = st.number_input("Mileage (km)", min_value=0, max_value=500_000,
                                 value=80_000, step=1_000)
    engine_cc = st.number_input("Engine capacity (cc)", min_value=600, max_value=6000,
                                value=1500, step=50)

with col2:
    condition = st.selectbox("Condition", options["condition"])
    transmission = st.selectbox("Transmission", options["transmission"])
    fuel_type = st.selectbox("Fuel type", options["fuel_type"])
    body_type = st.selectbox("Body type", options["body_type"])
    district = st.selectbox("District", options["district"])
    seller_type = st.selectbox("Seller type", options["seller_type"])

# ---------- Prediction ----------
if st.button("Predict price", type="primary"):
    car = pd.DataFrame([{
        "brand": brand,
        "model": car_model,
        "car_age": max(AD_YEAR - int(year), 0),
        "mileage_km": int(mileage_km),
        "engine_cc": int(engine_cc),
        "condition": condition,
        "transmission": transmission,
        "fuel_type": fuel_type,
        "body_type": body_type,
        "district": district,
        "seller_type": seller_type,
    }])

    log_price = model.predict(car)[0]
    price = float(np.expm1(log_price))

    st.success(f"Estimated price: **Rs {price:,.0f}**")
    st.caption(f"Car age used by the model: {car['car_age'][0]} years "
               f"({AD_YEAR} − {int(year)})")

    # ---------- Why this price? (SHAP) ----------
    st.subheader("Why this price?")
    values, base_value = explain(car[FEATURES])

    car_exp = shap.Explanation(values=values, base_values=base_value,
                               data=car[FEATURES].iloc[0].values, feature_names=FEATURES)
    shap.plots.waterfall(car_exp, max_display=11, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    top = int(np.argmax(np.abs(values)))
    effect = (np.exp(values[top]) - 1) * 100
    direction = "up" if values[top] > 0 else "down"
    st.write(f"**Biggest effect:** {FEATURES[top]} = {car[FEATURES[top]][0]} → "
             f"price {direction} about {abs(effect):.0f}%.")
    st.caption("Red bars push the price up. Blue bars push it down. "
               "The chart starts at the average price E[f(X)] and ends at this car's prediction f(x). "
               "Values are in log(price).")

st.caption("Model: XGBoost (best model from the notebook). "
           "Scope: cars from 1990–2024, Rs 0.5M – 100M, petrol / diesel / hybrid.")
