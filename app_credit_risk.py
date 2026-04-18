import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Default Risk", page_icon="🏦", layout="centered")

# we load our saved artifacts (model, encoders, etc.) once and cache them for future use
@st.cache_resource
def load_artifacts():
    return {
        "lgbm":     joblib.load("lgbm_model.pkl"),
        "meta":     joblib.load("meta_model.pkl"),
        "imputer":  joblib.load("imputer.pkl"),
        "scaler":   joblib.load("scaler.pkl"),
        "cols":     joblib.load("feature_columns.pkl"),
    }

art = load_artifacts()

st.title("Home Credit Default Risk")
st.caption("Loan applicant risk assessment: LightGBM + Logistic Regression stacking model")

with st.form("applicant"):
    st.subheader("Applicant details")
    col1, col2 = st.columns(2)

    with col1:
        income      = st.number_input("Annual income", 10000, 1000000, 135000, step=5000)
        credit      = st.number_input("Loan amount",   10000, 2000000, 500000, step=10000)
        annuity     = st.number_input("Annual annuity", 5000,  200000,  25000, step=1000)
        age_days    = st.slider("Age (years)", 18, 70, 35)
        employed_days = st.slider("Years employed", 0, 40, 5)

    with col2:
        gender      = st.selectbox("Gender", ["M", "F"])
        car         = st.selectbox("Owns a car?", ["Y", "N"])
        realty      = st.selectbox("Owns property?", ["Y", "N"])
        children    = st.number_input("Number of children", 0, 10, 0)
        family_size = st.number_input("Family members", 1, 10, 2)

    submitted = st.form_submit_button("Assess risk", use_container_width=True)

if submitted:
    # building a row with all expected features  *defaulting unknowns to 0
    row = dict.fromkeys(art["cols"], 0)

    row["AMT_INCOME_TOTAL"]  = income
    row["AMT_CREDIT"]        = credit
    row["AMT_ANNUITY"]       = annuity
    row["DAYS_BIRTH"]        = -(age_days * 365)
    row["DAYS_EMPLOYED"]     = -(employed_days * 365)
    row["CNT_CHILDREN"]      = children
    row["CNT_FAM_MEMBERS"]   = family_size
    row["CODE_GENDER"]       = 1 if gender == "M" else 0
    row["FLAG_OWN_CAR"]      = 1 if car == "Y" else 0
    row["FLAG_OWN_REALTY"]   = 1 if realty == "Y" else 0

    # our engineered features  *same as training
    row["CREDIT_INCOME_RATIO"]  = credit / (income + 1)
    row["ANNUITY_INCOME_RATIO"] = annuity / (income + 1)

    X_input = pd.DataFrame([row])[art["cols"]]

    # LGBM
    lgbm_prob = art["lgbm"].predict_proba(X_input)[0][1]

    # LogReg  *needs impute + scale
    X_imp    = art["imputer"].transform(X_input)
    X_scaled = art["scaler"].transform(X_imp)
    logreg_prob = art["meta"].predict_proba(
        np.column_stack([[lgbm_prob], [art["meta"].predict_proba(
            np.column_stack([[lgbm_prob], [0]]))[0][1]]]))[0][1]

    # Simple final: use LGBM directly (most reliable with partial input)
    final_prob = lgbm_prob

    if final_prob < 0.25:
        tier, color, desc = "Low risk", "green", "Applicant shows strong repayment indicators"
    elif final_prob < 0.50:
        tier, color, desc = "Medium risk", "orange", "Some risk factors present — review recommended"
    else:
        tier, color, desc = "High risk", "red", "Significant default probability — caution advised"

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Default probability", f"{final_prob:.1%}")
    c2.metric("Risk tier", tier)
    c3.metric("Credit / Income ratio", f"{row['CREDIT_INCOME_RATIO']:.2f}")

    st.markdown(f"### :{color}[{tier}]")
    st.caption(desc)

    st.divider()
    st.markdown("**What drove this prediction**")
    feat_imp = pd.Series(
        art["lgbm"].feature_importances_,
        index=art["cols"]
    ).nlargest(8).reset_index()
    feat_imp.columns = ["Feature", "Importance"]
    st.bar_chart(feat_imp.set_index("Feature"))