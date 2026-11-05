"""Acceptance checks. Run from the project folder:  python tests/acceptance_check.py

Drives the real app.py through Streamlit's AppTest harness. Prints PASS or FAIL per check.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import churn_lib as L  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def py(v):
    return v.item() if hasattr(v, "item") else v


# 1. App starts and renders without exceptions -------------------------------------------------
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
check("app runs without exception", not at.exception, str([e.value for e in at.exception]))

# 2. Four sections with exact names ------------------------------------------------------------
labels = [t.label for t in at.tabs]
want = ["Exploratory Data Analysis", "Model Performance", "Prediction & Explainability", "Business Application"]
check("four section names exact", labels == want, str(labels))

# 3. No training code in the app ---------------------------------------------------------------
bad = re.compile(r"\.fit\(|\.fit_transform\(|GridSearch|RandomizedSearch|optuna|cross_val|cross_validate|partial_fit",
                 re.I)
hits = []
for f in ("app.py", "churn_lib.py", "ui.py"):
    for i, line in enumerate((ROOT / f).read_text(encoding="utf8").splitlines(), 1):
        if bad.search(line):
            hits.append(f"{f}:{i}: {line.strip()}")
check("no fit/GridSearch/Optuna/CV code in app.py or churn_lib.py", not hits, "; ".join(hits))

# 4. EDA: 3 KPI-type cards present, filters respond --------------------------------------------
raw = L.load_raw()
n_all = len(raw)
shown = {m.label: m.value for m in at.metric}
check("EDA KPI cards present", all(k in shown for k in ("Customers", "Churn rate", "Average monthly charge")))
check("EDA customer count equals dataset size by default", shown["Customers"] == f"{n_all:,}", shown["Customers"])
at.multiselect[0].set_value(["Month-to-month"]).run()
shown2 = {m.label: m.value for m in at.metric}
d = raw[raw["Contract"] == "Month-to-month"]
check("EDA filter updates KPIs", shown2["Customers"] == f"{len(d):,}"
      and shown2["Churn rate"] == f"{(d['Churn'] == 'Yes').mean() * 100:.1f}%", str(shown2["Customers"]))
at.multiselect[0].set_value(sorted(raw["Contract"].unique())).run()

# 5. Probability via the real form matches test_predictions.csv within 1e-6 ---------------------
test = pd.read_csv(ROOT / "artifacts" / "test_predictions.csv")
rng = np.random.default_rng(0)
idx = [0, 1, 2] + [int(i) for i in rng.choice(len(test), 2, replace=False)]
meta = L.load_metadata()
errs = []
for i in idx:
    r = test.iloc[i]
    for col in meta["categorical_features"]:
        at.selectbox(key=f"in_{col}").set_value(py(r[col]))
    at.slider(key="in_tenure").set_value(int(r["tenure"]))
    at.slider(key="in_MonthlyCharges").set_value(float(r["MonthlyCharges"]))
    at.number_input(key="in_TotalCharges").set_value(float(r["TotalCharges"]))
    [b for b in at.button if b.label == "Estimate churn risk"][0].click()
    at.run()
    errs.append(abs(at.session_state["last_proba"] - float(r["churn_probability"])))
check("app probability == churn_probability for 5 test rows (<1e-6)", max(errs) < 1e-6, f"max abs err {max(errs):.2e}")
check("no exception after form submits", not at.exception)

# 6. Result text, SHAP and LIME present for the last row ---------------------------------------
text = " ".join(m.value for m in at.markdown)
check("plain-language sentence present",
      re.search(r"The model estimates a \d+% chance this customer will leave\. The main reasons are", text) is not None)

# 7. SHAP is exact: baseline + sum of contributions == model log-odds ---------------------------
pipe = L.load_pipeline()
bg = pd.read_csv(ROOT / "artifacts" / "background_sample.csv")
expl = L.make_shap_explainer(pipe, bg, meta)
row = L.build_row({c: py(test.iloc[3][c]) for c in meta["raw_features"]}, meta, raw[meta["raw_features"]].dtypes)
grouped, base, per_feature = L.shap_for_row(pipe, expl, row, meta)
logit = float(pipe.decision_function(row)[0])
check("SHAP base + grouped sum == model log-odds", abs(base + grouped.sum() - logit) < 1e-6,
      f"{base + grouped.sum():.6f} vs {logit:.6f}")
check("SHAP groups the 46 encoded columns into 19 variables", len(per_feature) == 46 and len(grouped) == 19)
lime = L.lime_for_row(pipe, bg, row, meta, k=8)
check("LIME returns 8 weights", len(lime) == 8)

# 7b. Artifacts agree with the saved model ----------------------------------------------------------
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402
fresh = pipe.predict_proba(test[meta["raw_features"]])[:, 1]
check("test_predictions.csv probabilities == saved pipeline output (<1e-9)",
      np.abs(fresh - test["churn_probability"]).max() < 1e-9)
_pred = (test["churn_probability"] >= 0.5).astype(int)
_m = {"accuracy": accuracy_score(test["Churn"], _pred), "precision": precision_score(test["Churn"], _pred),
      "recall": recall_score(test["Churn"], _pred), "f1": f1_score(test["Churn"], _pred),
      "roc_auc": roc_auc_score(test["Churn"], test["churn_probability"])}
check("test metrics recomputed from test_predictions.csv == model_metadata.json",
      all(abs(_m[k] - v) < 1e-9 for k, v in meta["test_metrics"].items()))
check("background sample has 300 rows and the 19 raw columns",
      len(bg) == 300 and list(bg.columns) == meta["raw_features"])

# 8. Business section ---------------------------------------------------------------------------
tp = L.load_csv("test_predictions.csv") if hasattr(L, "load_csv") else test
a = meta["business_assumptions_hypothetical"]
r = L.targeting(tp, meta["business_contact_share"], a["retention_rate"], a["value_months"], a["offer_cost"])
k = r["k"]
srt = tp.sort_values("churn_probability", ascending=False, kind="stable").head(k)
manual = float((srt["Churn"] * a["retention_rate"] * srt["MonthlyCharges"] * a["value_months"] - a["offer_cost"]).sum())
check("model-list net value matches manual calculation", abs(manual - r["model_net"]) < 1e-6, f"{manual:.2f}")
seg = L.segment_table(tp, a["retention_rate"], a["value_months"], a["offer_cost"])
check("segments cover every test customer", int(seg["Customers"].sum()) == len(tp))
shown_b = {m.label: m.value for m in at.metric}
want_k = f"{round(meta['business_contact_share'] * len(tp)):,}"
check("Business default contacts round(business_contact_share x n) customers",
      shown_b.get("Customers contacted") == want_k, f"{shown_b.get('Customers contacted')} vs {want_k}")
labels_b = {m.label for m in at.metric}
check("Business KPI cards present", {"Customers contacted", "Expected retained customers (hypothetical)",
                                     "Expected net value (hypothetical)"} <= labels_b)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
