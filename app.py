"""
app.py  —  Rain outlook dashboard (UI + backend)

Run:  streamlit run app.py

Backend : loads the two pre-trained models (pickle files) and runs inference.
UI      : input boxes -> "Predict" -> both models' results + performance tables.
"""

import gzip
import pickle
import random
from pathlib import Path

import pandas as pd
import streamlit as st

from features import (DISTRICTS, CLASS_NAMES, FEATURE_LABELS,
                      build_feature_row)

BASE = Path(__file__).parent
COLORS = {"Dry": "#C8913A", "Moderate": "#5E8C61", "Heavy": "#2F4E7A"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
PARTS = ["1st (days 1–10)", "2nd (days 11–20)", "3rd (day 21 – end)"]
MODEL_NAMES = ["Random Forest", "XGBoost"]

st.set_page_config(page_title="Rain outlook · Sri Lanka", page_icon="🌧️",
                   layout="centered")

# ------------------------------------------------------------------
# Small custom style (fonts, result cards, probability bar)
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, input, label {
    font-family: 'Manrope', system-ui, sans-serif !important;
}
h1 { font-weight: 800 !important; letter-spacing: -0.02em; line-height: 1.1 !important; }
h2, h3 { font-weight: 800 !important; letter-spacing: -0.01em; }
.lead { font-size: 1.05rem; color: #3D5566; max-width: 62ch; margin-bottom: 0.5rem; }
.card {
    background: #FFFFFF; border-radius: 14px; padding: 1.1rem 1.2rem;
    border-left: 8px solid var(--c);
}
.card .model { font-size: 0.9rem; color: #5A6E7B; margin: 0; }
.card .cls   { font-size: 2rem; font-weight: 800; color: var(--c); margin: 0.1rem 0; }
.card .conf  { font-size: 0.95rem; margin: 0 0 0.7rem 0; }
.bar { display: flex; height: 14px; border-radius: 7px; overflow: hidden; }
.bar span { display: block; height: 100%; }
.legend { display: flex; gap: 0.9rem; font-size: 0.8rem; color: #5A6E7B; margin-top: 0.4rem; flex-wrap: wrap; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }
.note { background: #FFFFFF; border-radius: 10px; padding: 0.7rem 1rem; margin-top: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Backend: load models and saved data once (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_backend():
    # Pickle files, compressed with gzip (the RF file is large)
    with gzip.open(BASE / "models" / "rf_model.pkl.gz", "rb") as f:
        rf = pickle.load(f)
    with gzip.open(BASE / "models" / "xgb_model.pkl.gz", "rb") as f:
        xgb = pickle.load(f)
    with open(BASE / "models" / "artifacts.pkl", "rb") as f:
        art = pickle.load(f)
    return {"Random Forest": rf, "XGBoost": xgb}, art


models, art = load_backend()
normals = art["normals"]
code_by_name = {v: k for k, v in DISTRICTS.items()}

# ------------------------------------------------------------------
# Session state defaults (so the "real example" button can fill inputs)
# ------------------------------------------------------------------
defaults = {"district": "Colombo", "month": "May", "part": PARTS[2],
            "r1": 50.0, "r2": 50.0, "r3": 50.0, "r9": 300.0, "actual": None}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


def fill_real_example():
    """Pick a random real row from the test years (2019-2026) and fill the inputs."""
    ex = art["examples"].iloc[random.randrange(len(art["examples"]))]
    dek = int(ex["dekad_of_year"])
    st.session_state.update({
        "district": DISTRICTS[ex["PCODE"]],
        "month": MONTHS[(dek - 1) // 3],
        "part": PARTS[(dek - 1) % 3],
        "r1": round(float(ex["lag1"]), 1), "r2": round(float(ex["lag2"]), 1),
        "r3": round(float(ex["lag3"]), 1), "r9": round(float(ex["r3h_lag1"]), 1),
        "actual": (CLASS_NAMES[int(ex["label"])], float(ex["rfh"]),
                   ex["date"].strftime("%d %b %Y")),
        "year": int(ex["date"].year), "nodata": False,
    })


def clear_actual():
    st.session_state["actual"] = None


# Years with data in the test period (2019-2026)
EX = art["examples"]
YEARS = sorted(EX["date"].dt.year.unique().tolist())
st.session_state.setdefault("year", YEARS[-2])
st.session_state.setdefault("nodata", False)


def autofill():
    """Fill the 4 rain fields from the dataset for the chosen district, year and dekad."""
    code = code_by_name[st.session_state["district"]]
    dek = (MONTHS.index(st.session_state["month"]) * 3
           + PARTS.index(st.session_state["part"]) + 1)
    match = EX[(EX["PCODE"] == code) & (EX["date"].dt.year == st.session_state["year"])
               & (EX["dekad_of_year"] == dek)]
    if match.empty:                       # no data for this date (e.g. after June 2026)
        st.session_state.update({"nodata": True, "actual": None})
        return
    ex = match.iloc[0]
    st.session_state.update({
        "nodata": False,
        "r1": round(float(ex["lag1"]), 1), "r2": round(float(ex["lag2"]), 1),
        "r3": round(float(ex["lag3"]), 1), "r9": round(float(ex["r3h_lag1"]), 1),
        "actual": (CLASS_NAMES[int(ex["label"])], float(ex["rfh"]),
                   ex["date"].strftime("%d %b %Y")),
    })


if "filled_once" not in st.session_state:     # fill once when the app opens
    st.session_state["filled_once"] = True
    autofill()


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("Rain outlook for the next 10 days")
st.markdown(
    "<p class='lead'>Choose a district and a date. "
    "Two trained models, Random Forest and XGBoost, predict whether the next "
    "10 days will be <b style='color:#C8913A'>Dry</b> (under 10 mm), "
    "<b style='color:#5E8C61'>Moderate</b> (10–100 mm) or "
    "<b style='color:#2F4E7A'>Heavy</b> (100 mm or more).</p>",
    unsafe_allow_html=True)

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
st.subheader("Choose a place and time")
st.button("Fill with a real example from 2019–2026", on_click=fill_real_example,
          help="Loads real values from the test years, so you can compare the "
               "prediction with what really happened.")

c1, c2, c3, c4 = st.columns([1.2, 0.8, 1, 1.3])
c1.selectbox("District", sorted(DISTRICTS.values()), key="district", on_change=autofill)
c2.selectbox("Year", YEARS, key="year", on_change=autofill)
c3.selectbox("Month to predict", MONTHS, key="month", on_change=autofill)
c4.selectbox("Part of the month", PARTS, key="part", on_change=autofill)

pcode = code_by_name[st.session_state["district"]]
target_dekad = MONTHS.index(st.session_state["month"]) * 3 + PARTS.index(st.session_state["part"]) + 1
st.caption(f"Normal rain in {st.session_state['district']} for this period: "
           f"**{normals[(pcode, target_dekad)][0]:.0f} mm** (long-term average).")

if st.session_state["nodata"]:
    st.warning("The dataset has no rain values for this date. "
               "Type the recent rain below, or choose another date.")

with st.expander("Recent rain (filled from the dataset, you can change it)",
                 expanded=st.session_state["nodata"]):
    st.markdown("**Rain in the last 3 dekads (mm)**")
    r1c, r2c, r3c = st.columns(3)
    r1c.number_input("Last 10 days", 0.0, 2000.0, step=1.0, key="r1", on_change=clear_actual)
    r2c.number_input("10–20 days ago", 0.0, 2000.0, step=1.0, key="r2", on_change=clear_actual)
    r3c.number_input("20–30 days ago", 0.0, 2000.0, step=1.0, key="r3", on_change=clear_actual)
    st.number_input("Total rain in the last 3 months (mm)", 0.0, 6000.0, step=5.0, key="r9",
                    on_change=clear_actual,
                    help="Sum of the last 9 dekads. It includes the 3 values above.")

predict = st.button("Predict rain level", type="primary", width="stretch")

# ------------------------------------------------------------------
# Inference + result
# ------------------------------------------------------------------
if predict:
    r1, r2, r3, r9 = (st.session_state[k] for k in ["r1", "r2", "r3", "r9"])
    if r9 < r1 + r2 + r3:
        st.error(f"The 3-month total ({r9:.0f} mm) is smaller than the last 3 dekads "
                 f"together ({r1 + r2 + r3:.0f} mm). Make the 3-month total at least that much.")
    else:
        x = build_feature_row(pcode, target_dekad, r1, r2, r3, r9, normals)
        st.subheader("Result")
        cols = st.columns(2)
        preds = {}
        for col, name in zip(cols, MODEL_NAMES):
            proba = models[name].predict_proba(x)[0]
            cls = CLASS_NAMES[proba.argmax()]
            preds[name] = cls
            bar = "".join(f"<span style='width:{p * 100:.1f}%;background:{COLORS[c]}'></span>"
                          for c, p in zip(CLASS_NAMES, proba))
            legend = "".join(f"<span><i style='background:{COLORS[c]}'></i>{c} {p:.0%}</span>"
                             for c, p in zip(CLASS_NAMES, proba))
            col.markdown(
                f"<div class='card' style='--c:{COLORS[cls]}'>"
                f"<p class='model'>{name}</p><p class='cls'>{cls}</p>"
                f"<p class='conf'>{proba.max():.0%} confident</p>"
                f"<div class='bar'>{bar}</div><div class='legend'>{legend}</div></div>",
                unsafe_allow_html=True)

        if preds["Random Forest"] == preds["XGBoost"]:
            msg = f"✅ Both models agree: <b>{preds['XGBoost']}</b>."
        else:
            msg = "⚠️ The models disagree. Rain is close to a class limit, so treat this as uncertain."
        actual = st.session_state["actual"]
        if actual:
            msg += (f"<br>📍 What really happened ({actual[2]}): "
                    f"<b style='color:{COLORS[actual[0]]}'>{actual[0]}</b>, {actual[1]:.0f} mm.")
        st.markdown(f"<div class='note'>{msg}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Model performance (fixed, from the test years)
# ------------------------------------------------------------------
st.divider()
st.subheader("How well the models performed")
st.caption(f"Trained on {art['train_period'][0][:4]}–{art['train_period'][1][:4]} "
           f"({art['n_train']:,} rows). Tested on unseen years "
           f"{art['test_period'][0][:4]}–{art['test_period'][1][:4]} ({art['n_test']:,} rows).")

met = pd.DataFrame(art["metrics"])[["Random Forest", "XGBoost", "Normal rainfall rule",
                                    "Majority class"]]
st.dataframe(met.style.format("{:.3f}").highlight_max(
    axis=1, subset=["Random Forest", "XGBoost"], color="#D7E6D8"),
    width="stretch")
st.caption("Macro scores give each class equal weight. 'Normal rainfall rule' and "
           "'Majority class' are simple rules without ML, shown for comparison.")

with st.expander("Confusion matrices (rows = real class, columns = predicted class)"):
    m1, m2 = st.columns(2)
    for col, name in zip([m1, m2], MODEL_NAMES):
        col.markdown(f"**{name}**")
        cm = pd.DataFrame(art["confusion"][name], index=CLASS_NAMES, columns=CLASS_NAMES)
        col.dataframe(cm, width="stretch")

with st.expander("Which inputs matter most"):
    imp = pd.DataFrame(art["importance"]).rename(index=FEATURE_LABELS)
    imp = imp.sort_values("Random Forest", ascending=False)
    st.bar_chart(imp, horizontal=True, stack=False, color=["#5E8C61", "#2F4E7A"], height=420)

with st.expander("Tuned model settings"):
    for name in MODEL_NAMES:
        p = {k: (round(v, 3) if isinstance(v, float) else v)
             for k, v in art["best_params"][name].items()}
        st.markdown(f"**{name}**")
        st.json(p, expanded=True)

st.caption("Data: Sri Lanka Rainfall Indicators at Subnational Level (HDX, CHIRPS).")
