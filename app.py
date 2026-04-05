import streamlit as st

st.title("Credit Risk Assessment")
st.subheader("Enter applicant details below")

with st.form("applicant_form"):
    AMT_INCOME_TOTAL = st.number_input("Annual Income", min_value=0.0, value=150000.0)
    AMT_CREDIT = st.number_input("Credit Amount", min_value=0.0, value=500000.0)
    AMT_ANNUITY = st.number_input("Annuity Amount", min_value=0.0, value=25000.0)
    CODE_GENDER = st.selectbox("Gender", ["M", "F"])
    submitted = st.form_submit_button("Assess Risk")

if submitted:
    import requests
    import numpy as np
    import pandas as pd
    
    X_sample = pd.read_csv("application_train.csv").head(1)
    sample = X_sample.drop(columns=["SK_ID_CURR", "TARGET"]).to_dict(orient="records")[0]
    sample = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in sample.items()}
    
    sample["AMT_INCOME_TOTAL"] = AMT_INCOME_TOTAL
    sample["AMT_CREDIT"] = AMT_CREDIT
    sample["AMT_ANNUITY"] = AMT_ANNUITY
    sample["CODE_GENDER"] = CODE_GENDER
    
    response = requests.post("http://127.0.0.1:8000/predict", json=sample)
    #st.write(response.status_code)
    #st.write(response.text)
    prob = response.json()["default_probability"]
    
    st.metric("Default Probability", f"{prob:.1%}")
    st.progress(prob)
    if prob > 0.5:
        st.error("HIGH RISK applicant")
    else:
        st.success("LOW RISK applicant")