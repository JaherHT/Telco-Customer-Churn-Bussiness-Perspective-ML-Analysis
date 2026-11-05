"""Helpers for the churn app. Everything here only LOADS and USES the saved model.

There is no training code here. The SHAP and LIME explainers only read summary numbers
(means, spreads) from the background sample; they do not train a model.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DATA = ROOT / "data"
RAW_CSV = DATA / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

THRESHOLD = 0.5          # label threshold, same as the notebook
RISK_LOW_MAX = 0.30      # below this: low risk. Display convention, fixed in advance, not tuned.
RISK_HIGH_MIN = 0.50     # at or above this: high risk (same as the label threshold)
SEED = 42

LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Has a partner",
    "Dependents": "Has dependents", "tenure": "Tenure (months)", "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines", "InternetService": "Internet service",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "Contract": "Contract type", "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method", "MonthlyCharges": "Monthly charges ($)",
    "TotalCharges": "Total charges ($)",
}


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def load_metadata() -> dict:
    return json.loads((ART / "model_metadata.json").read_text())


def load_pipeline():
    """Load the fitted pipeline. Never fitted here."""
    return joblib.load(ART / "churn_pipeline.joblib")


def load_raw() -> pd.DataFrame:
    """Raw dataset, cleaned like the notebook: TotalCharges is text, blank becomes 0."""
    df = pd.read_csv(RAW_CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
    return df


def load_csv(name: str, index_name: str | None = None) -> pd.DataFrame:
    """Read an artifact CSV. The notebook writes some tables with an unnamed first (index) column;
    pass index_name to give that column a proper name."""
    df = pd.read_csv(ART / name)
    if index_name and df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: index_name})
    return df


def display_value(col: str, value) -> str:
    if col == "SeniorCitizen":
        return "Yes" if int(value) == 1 else "No"
    if col == "tenure":
        return f"{int(value)} months"
    if col in ("MonthlyCharges", "TotalCharges"):
        return f"${float(value):,.2f}"
    return str(value)


# ---------------------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------------------
def build_row(values: dict, meta: dict, dtypes: pd.Series) -> pd.DataFrame:
    """One-row DataFrame with the raw column names, in the order given by model_metadata.json."""
    cols = meta["raw_features"]
    row = pd.DataFrame([{c: values[c] for c in cols}], columns=cols)
    return row.astype({c: dtypes[c] for c in cols})


def predict_proba(pipeline, row: pd.DataFrame) -> float:
    return float(pipeline.predict_proba(row)[0, 1])


def risk_band(p: float) -> str:
    if p >= RISK_HIGH_MIN:
        return "High"
    if p >= RISK_LOW_MAX:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------------------------
def feature_info(pipeline, meta: dict):
    """Encoded column names, and the original variable each encoded column comes from."""
    prep = pipeline.named_steps["prep"]
    num_cols, cat_cols = meta["numeric_features"], meta["categorical_features"]
    feature_names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    levels = prep.named_transformers_["cat"].categories_
    source = num_cols + [c for c, lv in zip(cat_cols, levels) for _ in lv]
    assert len(source) == len(feature_names)
    return feature_names, source


def transform(pipeline, df: pd.DataFrame, meta: dict) -> np.ndarray:
    return np.asarray(pipeline.named_steps["prep"].transform(df[meta["raw_features"]]), dtype=float)


def make_shap_explainer(pipeline, background: pd.DataFrame, meta: dict):
    import shap
    bg_t = transform(pipeline, background, meta)
    # max_samples keeps all background rows; shap would otherwise silently cut them to 100.
    masker = shap.maskers.Independent(bg_t, max_samples=len(bg_t))
    return shap.LinearExplainer(pipeline.named_steps["clf"], masker)


def shap_for_row(pipeline, explainer, row: pd.DataFrame, meta: dict):
    """Exact SHAP values in log-odds, summed per original variable. Positive means toward churn."""
    feature_names, source = feature_info(pipeline, meta)
    x_t = transform(pipeline, row, meta)
    sv = explainer.shap_values(x_t)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    per_feature = pd.Series(sv[0], index=feature_names)
    grouped = per_feature.groupby(np.array(source), sort=False).sum()
    base = float(np.ravel(explainer.expected_value)[-1])
    return grouped, base, per_feature


def lime_for_row(pipeline, background: pd.DataFrame, row: pd.DataFrame, meta: dict, k: int = 8):
    """LIME weights for the churn class, set up like the notebook. Returns a list of (label, weight)."""
    from lime.lime_tabular import LimeTabularExplainer
    feature_names, source = feature_info(pipeline, meta)
    n_num = len(meta["numeric_features"])
    bg_t = transform(pipeline, background, meta)
    explainer = LimeTabularExplainer(
        training_data=bg_t,
        feature_names=feature_names,
        class_names=["No churn", "Churn"],
        categorical_features=list(range(n_num, len(feature_names))),
        categorical_names={i: ["0", "1"] for i in range(n_num, len(feature_names))},
        mode="classification",
        random_state=SEED,
    )
    x_t = transform(pipeline, row, meta)[0]
    clf = pipeline.named_steps["clf"]
    exp = explainer.explain_instance(x_t, clf.predict_proba, num_features=k)
    out = []
    for idx, w in exp.as_map()[1]:
        var = source[idx]
        if idx < n_num:
            label = f"{LABELS[var]} = {display_value(var, row.iloc[0][var])}"
        else:
            level = feature_names[idx].split(var + "_", 1)[1]
            present = x_t[idx] >= 0.5
            shown = display_value(var, level)
            label = f"{LABELS[var]}: {shown}" if present else f"{LABELS[var]}: not {shown}"
        out.append((label, float(w)))
    return out


def reasons_sentence(p: float, grouped: pd.Series, row: pd.DataFrame, top: int = 3) -> str:
    """One plain sentence built from the largest grouped SHAP contributions."""
    best = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(top)
    parts = []
    for var, val in best.items():
        direction = "pushes toward leaving" if val > 0 else "pushes toward staying"
        parts.append(f"{LABELS[var].replace(' ($)', '').lower()} of {display_value(var, row.iloc[0][var])} ({direction})"
                     if var in ("tenure", "MonthlyCharges", "TotalCharges")
                     else f"{LABELS[var].lower()}: {display_value(var, row.iloc[0][var])} ({direction})")
    joined = f"{parts[0]}, {parts[1]} and {parts[2]}"
    return f"The model estimates a {p * 100:.0f}% chance this customer will leave. The main reasons are {joined}."


# ---------------------------------------------------------------------------------------------
# Business
# ---------------------------------------------------------------------------------------------
def net_per_customer(churn, monthly, retention_rate, value_months, offer_cost):
    """Churned: retention_rate x MonthlyCharges x value_months - offer_cost. Otherwise: -offer_cost."""
    churn = np.asarray(churn, dtype=float)
    monthly = np.asarray(monthly, dtype=float)
    return churn * retention_rate * monthly * value_months - offer_cost


def targeting(test: pd.DataFrame, share: float, retention_rate: float, value_months: float, offer_cost: float):
    """Contact the top `share` of customers by churn_probability; compare with a random list of equal size."""
    proba = test["churn_probability"].to_numpy()
    churn = test["Churn"].to_numpy()
    monthly = test["MonthlyCharges"].to_numpy()
    net = net_per_customer(churn, monthly, retention_rate, value_months, offer_cost)
    order = np.argsort(-proba, kind="stable")
    n = len(test)
    k = int(min(n, max(1, round(share * n))))
    top = order[:k]
    return {
        "n": n, "k": k,
        "model_net": float(net[top].sum()),
        "random_net": float(k * net.mean()),                     # expected value of a random list
        "model_retained": float(retention_rate * churn[top].sum()),
        "random_retained": float(retention_rate * k * churn.mean()),
        "model_churners": int(churn[top].sum()),
        "cutoff": float(proba[top[-1]]),
        "curve_share": np.arange(1, n + 1) / n,
        "curve_model": np.cumsum(net[order]),
        "curve_random": np.arange(1, n + 1) * net.mean(),
    }


def segment_table(test: pd.DataFrame, retention_rate: float, value_months: float, offer_cost: float) -> pd.DataFrame:
    p = test["churn_probability"]
    band = pd.cut(p, [-np.inf, RISK_LOW_MAX, RISK_HIGH_MIN, np.inf], labels=["Low", "Medium", "High"], right=False)
    net = pd.Series(net_per_customer(test["Churn"], test["MonthlyCharges"], retention_rate, value_months, offer_cost),
                    index=test.index)
    rows = []
    for name, rule in (("High", f"probability ≥ {RISK_HIGH_MIN:.2f}"),
                       ("Medium", f"{RISK_LOW_MAX:.2f} to {RISK_HIGH_MIN:.2f}"),
                       ("Low", f"probability < {RISK_LOW_MAX:.2f}")):
        m = band == name
        rows.append({
            "Segment": name, "Rule": rule, "Customers": int(m.sum()),
            "Share of customers": float(m.mean()),
            "Actual churn rate": float(test.loc[m, "Churn"].mean()) if m.any() else float("nan"),
            "Net value if all contacted (hypothetical, $)": float(net[m].sum()),
            "Net value per customer (hypothetical, $)": float(net[m].mean()) if m.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def confusion_from_predictions(test: pd.DataFrame, threshold: float = THRESHOLD):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                                 roc_auc_score)
    y = test["Churn"].to_numpy()
    p = test["churn_probability"].to_numpy()
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    metrics = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
               "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p)}
    return cm, metrics
