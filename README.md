# Rain Outlook for Sri Lanka: Random Forest vs XGBoost


The app predicts the **rain level of the next dekad (next 10 days)** for any of the
**25 districts** of Sri Lanka, using two pre-trained models: **Random Forest** and **XGBoost**.

| Class | Rule |
|---|---|
| Dry | 10-day rainfall below 10 mm |
| Moderate | 10 mm to below 100 mm |
| Heavy | 100 mm or more |

## Dataset

*Sri Lanka: Rainfall Indicators at Subnational Level*, Humanitarian Data Exchange (HDX):
https://data.humdata.org/dataset/lka-rainfall-subnational

CHIRPS satellite rainfall with station data, averaged per district, one row every dekad,
1981 to 2026. Only district rows (`adm_level = 2`) and `final` values are used.

## Files

| File | Purpose |
|---|---|
| `rainfall_model_training.ipynb` | Data cleaning, feature engineering, tuning, RF vs XGBoost comparison, saving the models |
| `features.py` | Shared code: district names, class rules, feature calculation (used by notebook **and** app) |
| `app.py` | Streamlit app: UI + backend (loads the pickle files and runs inference) |
| `models/rf_model.pkl.gz` | Trained Random Forest (pickle, gzip-compressed) |
| `models/xgb_model.pkl.gz` | Trained XGBoost (pickle, gzip-compressed) |
| `models/artifacts.pkl` | Normal rainfall tables, metrics, confusion matrices, feature importance, test examples |
| `data/lka-rainfall-subnat-full.xlsx` | The dataset |
| `.streamlit/config.toml` | App colour theme |
| `requirements.txt` | Python libraries (versions match the saved models) |

## How to run the app

Python 3.12 is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501

## How to use the app

1. Choose the **district**, the **month** and the **part of the month** to predict.
2. Enter the rain of the **last 3 dekads** and the **total rain of the last 3 months** (mm).
   Or click **"Fill with a real example from 2019–2026"** to load real values. The app then also
   shows what really happened, so the prediction can be checked.
3. Click **Predict rain level**. Both models run, and the app shows each model's class,
   confidence, and whether the two models agree.
4. The **How well the models performed** section shows test results, confusion matrices,
   feature importance and the tuned settings.


## Method summary

- **No data leakage:** `r1h`, `r3h`, `rfq`, `r1q`, `r3q` of a row are calculated from that row's own
  rainfall, so only their values from the **previous dekad** are used. Lag features
  (`lag1`, `lag2`, `lag3`) are created with `groupby("PCODE").shift()`.
- **Time-based split:** train 1981–2018, test 2019–2026.
- **Tuning:** `RandomizedSearchCV` with `TimeSeriesSplit` (3 folds), scoring = macro F1.
- **Class imbalance:** class weights (RF) and balanced sample weights (XGBoost).
- **Baselines:** majority class and the "normal rainfall rule" (class of the long-term average).

## Deploy online (Streamlit Community Cloud, free)

1. Create a **public GitHub repository** and upload all files in this folder
   (keep the same folder structure, including `.streamlit/` and `models/`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository, branch `main`, and main file `app.py`.
4. Under **Advanced settings**, choose **Python 3.12**, then click **Deploy**.
5. Share the app link.
