"""Telco customer churn: a Streamlit app that loads a model trained in a notebook.

The app only LOADS and USES the saved pipeline. It never trains, tunes or refits anything.
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import churn_lib as L
import ui

st.set_page_config(page_title="Telco Customer Churn", layout="wide")
ui.inject_css()

STAY, CHURN, AMBER, SOFT = ui.STAY, ui.CHURN, ui.AMBER, ui.SOFT
STAY_CHURN_MAP = {"No": STAY, "Yes": CHURN}


# --------------------------------------------------------------------------------------------
# Cached loaders. The pipeline is loaded once and never fitted.
# --------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the saved model...")
def get_pipeline():
    return L.load_pipeline()


@st.cache_data(show_spinner=False)
def get_meta():
    return L.load_metadata()


@st.cache_data(show_spinner=False)
def get_raw():
    return L.load_raw()


@st.cache_data(show_spinner=False)
def get_csv(name, index_name=None):
    return L.load_csv(name, index_name)


@st.cache_resource(show_spinner="Preparing the SHAP explainer...")
def get_shap_explainer():
    return L.make_shap_explainer(get_pipeline(), get_csv("background_sample.csv"), get_meta())


@st.cache_data(show_spinner="Computing LIME weights...")
def get_lime(values_key: tuple):
    values = dict(values_key)
    row = L.build_row(values, get_meta(), get_raw()[get_meta()["raw_features"]].dtypes)
    return L.lime_for_row(get_pipeline(), get_csv("background_sample.csv"), row, get_meta(), k=8)


def money(x: float, signed: bool = False) -> str:
    sign = "-" if x < 0 else "+" if signed else ""
    return f"{sign}${abs(x):,.0f}"


def bar_h(labels, values, xtitle, height=None):
    """Horizontal bars. Terracotta pushes toward churn, teal pushes away."""
    colors = [CHURN if v > 0 else STAY for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                           hovertemplate="%{y}<br>%{x:+.3f}<extra></extra>"))
    fig.update_layout(xaxis_title=xtitle, yaxis=dict(autorange="reversed"), margin=dict(l=0, r=10, t=10, b=0),
                      showlegend=False)
    fig.add_vline(x=0, line_color=SOFT, line_width=1)
    return ui.style_fig(fig, height or max(260, 26 * len(labels) + 60))


# --------------------------------------------------------------------------------------------
# Section A: Exploratory Data Analysis
# --------------------------------------------------------------------------------------------
def section_eda():
    raw = get_raw()
    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1.2])
    contracts = sorted(raw["Contract"].unique())
    internet = sorted(raw["InternetService"].unique())
    payments = sorted(raw["PaymentMethod"].unique())
    f_contract = c1.multiselect("Contract", contracts, default=contracts)
    f_internet = c2.multiselect("Internet service", internet, default=internet)
    f_payment = c3.multiselect("Payment method", payments, default=payments)
    f_tenure = c4.slider("Tenure (months)", 0, 72, (0, 72))

    d = raw[raw["Contract"].isin(f_contract) & raw["InternetService"].isin(f_internet)
            & raw["PaymentMethod"].isin(f_payment) & raw["tenure"].between(*f_tenure)].copy()
    if d.empty:
        st.warning("No customers match these filters.")
        return
    d["churned"] = (d["Churn"] == "Yes").astype(int)
    all_rate = (raw["Churn"] == "Yes").mean()

    k1, k2, k3 = st.columns(3)
    k1.metric("Customers", f"{len(d):,}", delta=f"of {len(raw):,}", delta_color="off", delta_arrow="off", border=True)
    k2.metric("Churn rate", f"{d['churned'].mean() * 100:.1f}%",
              delta=f"{(d['churned'].mean() - all_rate) * 100:+.1f} pts vs all", delta_color="off",
              delta_arrow="off", border=True)
    k3.metric("Average monthly charge", f"${d['MonthlyCharges'].mean():,.2f}",
              delta=f"{d['MonthlyCharges'].mean() - raw['MonthlyCharges'].mean():+.2f} vs all", delta_color="off",
              delta_arrow="off", border=True)

    left, right = st.columns(2)
    by_contract = d.groupby("Contract").agg(rate=("churned", "mean")).reset_index()
    by_contract["rate"] *= 100
    fig = px.bar(by_contract, x="Contract", y="rate", text=by_contract["rate"].map("{:.1f}%".format),
                 color_discrete_sequence=[CHURN], labels={"rate": "Churn rate (%)", "Contract": ""})
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_layout(title="Churn rate by contract", yaxis_range=[0, max(60, by_contract["rate"].max() * 1.15)],
                      margin=dict(t=44, b=0, l=0, r=0), bargap=0.45, yaxis_title="Churn rate (%)")
    left.plotly_chart(ui.style_fig(fig))

    fig = px.histogram(d, x="tenure", color="Churn", nbins=36, barmode="overlay", opacity=0.7,
                       histnorm="percent", color_discrete_map=STAY_CHURN_MAP,
                       labels={"tenure": "Tenure (months)", "Churn": "Left?"})
    fig.update_layout(title="Tenure by churn", yaxis_title="Share of group (%)", margin=dict(t=44, b=0, l=0, r=0))
    right.plotly_chart(ui.style_fig(fig))

    fig = px.violin(d, x="Churn", y="MonthlyCharges", color="Churn", box=True, points=False,
                    color_discrete_map=STAY_CHURN_MAP, labels={"MonthlyCharges": "Monthly charges ($)", "Churn": "Left?"})
    fig.update_layout(title="Monthly charges by churn", showlegend=False, margin=dict(t=44, b=0, l=0, r=0))
    st.plotly_chart(ui.style_fig(fig))


# --------------------------------------------------------------------------------------------
# Section B: Model Performance
# --------------------------------------------------------------------------------------------
def section_performance():
    meta = get_meta()

    st.subheader("Baseline models (5-fold CV)")
    base = get_csv("baseline_table.csv", "model").set_index("model").drop(columns=["f1 std"])
    st.dataframe(base, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in base.columns})

    st.subheader("Search strategies")
    table = get_csv("hyperparameter_search.csv")
    st.dataframe(table[["strategy", "configs evaluated", "time (s)", "best CV F1"]].rename(columns={"strategy": "Strategy"}),
                 hide_index=True, column_config={"time (s)": st.column_config.NumberColumn(format="%.1f"),
                                                 "best CV F1": st.column_config.NumberColumn(format="%.4f")})
    if table["best CV F1"].max() - table["best CV F1"].min() < 5e-4:
        st.caption(f"All four strategies reach the same CV F1 ({table['best CV F1'].max():.4f}); they differ in time.")
    a, b = st.columns(2)
    fig = px.bar(table, x="strategy", y="best CV F1", text=table["best CV F1"].map("{:.4f}".format),
                 color_discrete_sequence=[STAY], labels={"strategy": "", "best CV F1": ""})
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_layout(title="Best CV F1", yaxis_range=[0, 1], margin=dict(t=44, b=0, l=0, r=0), bargap=0.45)
    a.plotly_chart(ui.style_fig(fig))
    fig = px.bar(table, x="strategy", y="time (s)", text=table["time (s)"].map("{:.1f}".format),
                 color_discrete_sequence=[AMBER], labels={"strategy": "", "time (s)": ""})
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_layout(title="Search time (s)", margin=dict(t=44, b=0, l=0, r=0), bargap=0.45)
    b.plotly_chart(ui.style_fig(fig))

    st.subheader("Pareto front: fewer features vs F1")
    pareto = get_csv("pareto_front.csv")
    p = meta["params"]
    sel = pareto[np.isclose(pareto["C"], p["C"]) & np.isclose(pareto["l1_ratio"], p["l1_ratio"])
                 & (pareto["class_weight"].astype(str) == str(p["class_weight"]))].sort_values("trial")
    fig = go.Figure()
    if not sel.empty:   # drawn first and larger, so the front dots stay visible on top of the star
        s = sel.iloc[0]
        fig.add_trace(go.Scatter(x=[s["features"]], y=[s["f1"]], mode="markers", name="Selected model",
                                 marker=dict(size=22, color=CHURN, symbol="star", line=dict(width=1, color="#FFFFFF")),
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pareto["features"], y=pareto["f1"], mode="markers", name="Pareto front",
                             marker=dict(size=9, color=STAY, line=dict(width=1, color="#FFFFFF")),
                             customdata=pareto[["trial", "C", "l1_ratio", "class_weight"]],
                             hovertemplate="trial %{customdata[0]}<br>features %{x:.0f}<br>CV F1 %{y:.4f}"
                                           "<br>C=%{customdata[1]}, l1_ratio=%{customdata[2]}, "
                                           "class_weight=%{customdata[3]}<extra></extra>"))
    fig.update_layout(xaxis_title="Features used", yaxis_title="CV F1",
                      xaxis_range=[pareto["features"].min() - 6, pareto["features"].max() + 6],
                      yaxis_range=[pareto["f1"].min() - 0.01, pareto["f1"].max() + 0.01],
                      margin=dict(t=10, b=40, l=0, r=0))
    st.plotly_chart(ui.style_fig(fig))
    st.caption("Up and left is better. The star is the selected model.")

    st.subheader("Test set results")
    tm = meta["test_metrics"]
    cols = st.columns(5)
    for col, (k, name) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                                     ("f1", "F1"), ("roc_auc", "ROC-AUC")]):
        col.metric(name, f"{tm[k]:.3f}", border=True)

    st.subheader("Confusion matrix (test set, threshold 0.5)")
    test = get_csv("test_predictions.csv")
    cm, mt = L.confusion_from_predictions(test)
    left, right = st.columns([1.1, 1])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, "#FFFFFF"], [1, "#9CC9C4"]],
                    x=["Predicted stays", "Predicted leaves"], y=["Actually stayed", "Actually left"])
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=0, l=0, r=0))
    left.plotly_chart(ui.style_fig(fig, 300))
    match = all(abs(mt[k] - tm[k]) < 1e-9 for k in tm)
    (right.success if match else right.warning)(
        "Matches the notebook's stored test metrics." if match else
        "Differs from the notebook's stored test metrics. Check the files.")


# --------------------------------------------------------------------------------------------
# Section C: Prediction & Explainability
# --------------------------------------------------------------------------------------------
def section_prediction():
    meta, raw, pipe = get_meta(), get_raw(), get_pipeline()
    dtypes = raw[meta["raw_features"]].dtypes

    def opts(col):
        return sorted(raw[col].unique().tolist())

    defaults = {c: raw[c].mode().iloc[0] for c in meta["categorical_features"]}
    defaults.update({"tenure": int(raw["tenure"].median()), "MonthlyCharges": float(raw["MonthlyCharges"].median()),
                     "TotalCharges": float(round(raw["TotalCharges"].median(), 2))})
    if "customer" not in st.session_state:
        st.session_state["customer"] = {c: (v.item() if hasattr(v, "item") else v) for c, v in defaults.items()}

    result = st.container()

    def sel(col, container):
        o = opts(col)
        kw = {"format_func": lambda v: L.display_value(col, v)} if col == "SeniorCitizen" else {}
        return container.selectbox(L.LABELS[col], o, index=o.index(defaults[col]), key=f"in_{col}", **kw)

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        v = {}
        c1.markdown("**About**")
        for col in ("gender", "SeniorCitizen", "Partner", "Dependents"):
            v[col] = sel(col, c1)
        c2.markdown("**Account**")
        v["tenure"] = c2.slider("Tenure (months)", 0, 72, defaults["tenure"], key="in_tenure")
        v["Contract"] = sel("Contract", c2)
        v["PaperlessBilling"] = sel("PaperlessBilling", c2)
        v["PaymentMethod"] = sel("PaymentMethod", c2)
        v["MonthlyCharges"] = c2.slider("Monthly charges ($)", 18.25, 118.75, defaults["MonthlyCharges"], step=0.05,
                                        key="in_MonthlyCharges")
        v["TotalCharges"] = c2.number_input("Total charges ($)", min_value=0.0, value=defaults["TotalCharges"],
                                            step=10.0, key="in_TotalCharges")
        c3.markdown("**Services**")
        for col in ("PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"):
            v[col] = sel(col, c3)
        submitted = st.form_submit_button("Estimate churn risk", type="primary")
    if submitted:
        st.session_state["customer"] = {c: (x.item() if hasattr(x, "item") else x) for c, x in v.items()}

    values = st.session_state["customer"]
    row = L.build_row(values, meta, dtypes)
    proba = L.predict_proba(pipe, row)
    st.session_state["last_proba"] = proba          # read by the acceptance test
    band = L.risk_band(proba)
    grouped, base, per_feature = L.shap_for_row(pipe, get_shap_explainer(), row, meta)

    with result:
        st.subheader("Result")
        st.markdown(ui.verdict(L.reasons_sentence(proba, grouped, row)), unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Chance of leaving", f"{proba * 100:.1f}%", border=True)
        m2.metric(f"Prediction (cutoff {L.THRESHOLD:.0%})", "Leaves" if proba >= L.THRESHOLD else "Stays", border=True)
        with m3.container(border=True):
            st.caption(f"Risk band  ·  low under {L.RISK_LOW_MAX:.0%}, high from {L.RISK_HIGH_MIN:.0%}")
            st.markdown(ui.risk_pill(band, big=True), unsafe_allow_html=True)

    st.subheader("Why? SHAP")
    st.caption("Log-odds. Terracotta raises the chance of leaving; teal lowers it.")
    g = grouped.reindex(grouped.abs().sort_values(ascending=False).index)
    labels = [f"{L.LABELS[c]} = {L.display_value(c, row.iloc[0][c])}" for c in g.index]
    st.plotly_chart(bar_h(labels, g.to_numpy(), "Contribution (log-odds)", height=520))

    st.subheader("Cross-check: LIME")
    lime = get_lime(tuple((c, values[c]) for c in meta["raw_features"]))
    lime = sorted(lime, key=lambda t: abs(t[1]), reverse=True)
    st.plotly_chart(bar_h([t[0] for t in lime], [t[1] for t in lime], "Weight toward churn", height=340))


# --------------------------------------------------------------------------------------------
# Section D: Business Application
# --------------------------------------------------------------------------------------------
def section_business():
    meta, test = get_meta(), get_csv("test_predictions.csv")
    a = meta["business_assumptions_hypothetical"]

    st.subheader("Hypothetical assumptions")
    c1, c2, c3 = st.columns(3)
    rr = c1.slider("Retention rate (hypothetical)", 0.0, 1.0, float(a["retention_rate"]), 0.01,
                   help="Share of contacted churners the offer keeps.")
    vm = c2.slider("Value months (hypothetical)", 1, 24, int(a["value_months"]),
                   help="Months of monthly charges kept for each saved customer.")
    oc = c3.slider("Offer cost, $ (hypothetical)", 0.0, 100.0, float(a["offer_cost"]), 0.5,
                   help="Cost of contacting one customer.")

    default_share = float(meta["business_contact_share"])
    share_pct = st.slider("Share contacted (%)", 1.0, 100.0, default_share * 100, 0.1, format="%.1f%%",
                          help=f"Highest predicted risk first. Default {default_share:.1%} comes from development data in the notebook.")
    share = share_pct / 100

    r = L.targeting(test, share, rr, vm, oc)
    st.subheader("Who to contact")
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers contacted", f"{r['k']:,}", delta=f"of {r['n']:,}", delta_color="off", delta_arrow="off", border=True)
    k2.metric("Expected retained customers (hypothetical)", f"{r['model_retained']:.1f}",
              delta=f"{r['model_retained'] - r['random_retained']:+.1f} vs random", delta_color="off",
              delta_arrow="off", border=True)
    k3.metric("Expected net value (hypothetical)", money(r["model_net"]),
              delta=f"{money(r['model_net'] - r['random_net'], signed=True)} vs random", delta_color="off",
              delta_arrow="off", border=True)

    comp = pd.DataFrame({
        "List": ["Model (highest risk first)", "Random (same size)"],
        "Contacted": [r["k"], r["k"]],
        "Retained (hypothetical)": [r["model_retained"], r["random_retained"]],
        "Net value (hypothetical)": [money(r["model_net"]), money(r["random_net"])],
    })
    st.dataframe(comp, hide_index=True, column_config={
        "Retained (hypothetical)": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Net value per customer: retention rate × monthly charges × value months − offer cost if the customer churned; "
               "− offer cost otherwise.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r["curve_share"], y=r["curve_model"], name="Model ranking",
                             line=dict(color=STAY, width=2.5)))
    fig.add_trace(go.Scatter(x=r["curve_share"], y=r["curve_random"], name="Random",
                             line=dict(color=SOFT, dash="dash", width=1.5)))
    fig.add_vline(x=r["k"] / r["n"], line_color=CHURN, line_dash="dot")
    fig.update_layout(title="Net value by share contacted (hypothetical)", xaxis_title="Share contacted",
                      yaxis_title="Net value ($)", xaxis_tickformat=".0%", margin=dict(t=44, b=40, l=0, r=0))
    st.plotly_chart(ui.style_fig(fig))

    st.subheader("Risk segments (test set)")
    seg = L.segment_table(test, rr, vm, oc)
    seg_show = seg[["Segment", "Customers", "Actual churn rate", "Net value per customer (hypothetical, $)"]].copy()
    seg_show["Net value per customer (hypothetical, $)"] = seg_show["Net value per customer (hypothetical, $)"].map(
        lambda x: f"{'-' if x < 0 else '+'}${abs(x):,.2f}")
    st.dataframe(seg_show, hide_index=True, column_config={
        "Actual churn rate": st.column_config.NumberColumn(format="percent")})

    by = seg.set_index("Segment")
    cards = st.columns(3)
    texts = {
        "High": "Worth testing an offer.",
        "Medium": ("Test a small, low-cost offer first." if by.loc["Medium", "Net value per customer (hypothetical, $)"] > 0
                   else "An offer likely costs more than it returns."),
        "Low": ("No offer needed." if by.loc["Low", "Net value per customer (hypothetical, $)"] <= 0
                else "Lowest priority."),
    }
    for col, name in zip(cards, ["High", "Medium", "Low"]):
        s = by.loc[name]
        per = s["Net value per customer (hypothetical, $)"]
        stat = f"{int(s['Customers']):,} customers · {s['Actual churn rate'] * 100:.1f}% churned"
        value = f"{'-' if per < 0 else '+'}${abs(per):,.2f} per customer (hypothetical)"
        col.markdown(ui.segment_card(name, stat, texts[name], value), unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------
st.title("Telco Customer Churn")
st.caption("A saved logistic-regression model. This app loads it and never trains it. Money figures are hypothetical.")
tabs = st.tabs(["Exploratory Data Analysis", "Model Performance", "Prediction & Explainability", "Business Application"])
with tabs[0]:
    section_eda()
with tabs[1]:
    section_performance()
with tabs[2]:
    section_prediction()
with tabs[3]:
    section_business()
