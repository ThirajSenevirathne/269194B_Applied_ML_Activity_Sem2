"""
features.py
-----------
Shared code for the rainfall project.

Both the training notebook and the Streamlit app import this file.
This makes sure the features are calculated in EXACTLY the same way
during training and during inference (prediction).
If the two ways were different, the model would get wrong inputs.
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. District codes -> names
#    The file uses the Department of Census and Statistics order.
#    (Checked with the n_pixels column: district size matches.)
# ------------------------------------------------------------------
DISTRICTS = {
    "LK11": "Colombo",      "LK12": "Gampaha",     "LK13": "Kalutara",
    "LK21": "Kandy",        "LK22": "Matale",      "LK23": "Nuwara Eliya",
    "LK31": "Galle",        "LK32": "Matara",      "LK33": "Hambantota",
    "LK41": "Jaffna",       "LK42": "Mannar",      "LK43": "Vavuniya",
    "LK44": "Mullaitivu",   "LK45": "Kilinochchi",
    "LK51": "Batticaloa",   "LK52": "Ampara",      "LK53": "Trincomalee",
    "LK61": "Kurunegala",   "LK62": "Puttalam",
    "LK71": "Anuradhapura", "LK72": "Polonnaruwa",
    "LK81": "Badulla",      "LK82": "Monaragala",
    "LK91": "Ratnapura",    "LK92": "Kegalle",
}
# The model needs numbers, so each district code gets a number (0..24).
DISTRICT_TO_NUM = {code: i for i, code in enumerate(sorted(DISTRICTS))}

# ------------------------------------------------------------------
# 2. Label (what we predict): rain level of the next dekad
#    Dry      : rfh <  10 mm   (less than about 1 mm per day)
#    Moderate : 10 mm <= rfh < 100 mm
#    Heavy    : rfh >= 100 mm  (about 10 mm or more per day)
# ------------------------------------------------------------------
CLASS_NAMES = ["Dry", "Moderate", "Heavy"]
DRY_LIMIT = 10.0
HEAVY_LIMIT = 100.0


def rain_class(rfh):
    """Turn rainfall in mm into a class number: 0=Dry, 1=Moderate, 2=Heavy."""
    rfh = np.asarray(rfh, dtype=float)
    return np.where(rfh < DRY_LIMIT, 0, np.where(rfh >= HEAVY_LIMIT, 2, 1))


# ------------------------------------------------------------------
# 3. The features (model inputs). All of them are known BEFORE the
#    dekad we want to predict, so there is no data leakage.
# ------------------------------------------------------------------
FEATURES = [
    "district_num",   # which district (0..24)
    "dekad_of_year",  # 1..36  (which 10-day period of the year)
    "rfh_avg",        # long-term normal rainfall for this district + dekad
    "lag1",           # rainfall of the last dekad        (t-1)
    "lag2",           # rainfall 2 dekads ago             (t-2)
    "lag3",           # rainfall 3 dekads ago             (t-3)
    "r1h_lag1",       # total rainfall of the last 1 month  (3 dekads)
    "r3h_lag1",       # total rainfall of the last 3 months (9 dekads)
    "rfq_lag1",       # last dekad's anomaly        (% of normal)
    "r1q_lag1",       # last 1-month anomaly        (% of normal)
    "r3q_lag1",       # last 3-month anomaly        (% of normal)
]

# Friendly names for charts in the app
FEATURE_LABELS = {
    "district_num": "District",
    "dekad_of_year": "Time of year (dekad)",
    "rfh_avg": "Normal rainfall for this dekad",
    "lag1": "Rain, last dekad",
    "lag2": "Rain, 2 dekads ago",
    "lag3": "Rain, 3 dekads ago",
    "r1h_lag1": "Rain, last 1 month",
    "r3h_lag1": "Rain, last 3 months",
    "rfq_lag1": "Last dekad vs normal (%)",
    "r1q_lag1": "Last month vs normal (%)",
    "r3q_lag1": "Last 3 months vs normal (%)",
}


def dekad_of_year(dates):
    """1..36. Day 1 -> 1st dekad, day 11 -> 2nd, day 21 -> 3rd."""
    dates = pd.to_datetime(dates)
    return (dates.dt.month - 1) * 3 + dates.dt.day // 10 + 1


def anomaly(value, normal):
    """
    Rainfall anomaly in %, using the same formula as the dataset:
        (value + 5) / (normal + 5) * 100
    We checked this formula against the rfq, r1q and r3q columns.
    The '+5' avoids dividing by zero in very dry periods.
    """
    return (value + 5.0) / (normal + 5.0) * 100.0


def previous_dekad(dek):
    """Dekad number before 'dek' (dekad 1 comes after dekad 36)."""
    return 36 if dek == 1 else dek - 1


# ------------------------------------------------------------------
# 4. Build ONE feature row from simple user inputs (used by the app)
# ------------------------------------------------------------------
def build_feature_row(pcode, target_dekad, rain_t1, rain_t2, rain_t3,
                      rain_3months, normals):
    """
    pcode        : district code, e.g. "LK11"
    target_dekad : 1..36, the dekad we want to predict
    rain_t1..t3  : rainfall (mm) of the last 3 dekads
    rain_3months : total rainfall (mm) of the last 9 dekads (3 months)
    normals      : dict {(pcode, dekad): (rfh_avg, r1h_avg, r3h_avg)}
                   (long-term averages saved from the dataset)
    """
    prev = previous_dekad(target_dekad)
    rfh_avg_t = normals[(pcode, target_dekad)][0]          # normal for t
    rfh_avg_p, r1h_avg_p, r3h_avg_p = normals[(pcode, prev)]  # normals for t-1

    r1h = rain_t1 + rain_t2 + rain_t3       # 1 month = last 3 dekads
    r3h = rain_3months                      # 3 months = last 9 dekads

    row = {
        "district_num": DISTRICT_TO_NUM[pcode],
        "dekad_of_year": target_dekad,
        "rfh_avg": rfh_avg_t,
        "lag1": rain_t1,
        "lag2": rain_t2,
        "lag3": rain_t3,
        "r1h_lag1": r1h,
        "r3h_lag1": r3h,
        "rfq_lag1": anomaly(rain_t1, rfh_avg_p),
        "r1q_lag1": anomaly(r1h, r1h_avg_p),
        "r3q_lag1": anomaly(r3h, r3h_avg_p),
    }
    return pd.DataFrame([row], columns=FEATURES)
