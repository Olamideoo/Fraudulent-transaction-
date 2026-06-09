import streamlit as st
import pandas as pd
import joblib

# --- 1. OPTIMIZED LOADING ---
# Using cache_resource makes the app much faster
@st.cache_resource
def load_assets():
    model = joblib.load("rf_honest_fraud_model.joblib")
    feature_names = joblib.load("model_feature_names.joblib")
    return model, feature_names

model, feature_names = load_assets()

st.title("🛡️ Fraud Detection System")
st.markdown("---")

# --- 2. INPUT SECTION ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Transaction")
    amount_src = st.number_input("Transaction Amount", min_value=0.0, value=500.0)
    amount_usd = st.number_input("Amount (USD)", min_value=0.0, value=500.0)
    fee = st.number_input("Fee", min_value=0.0, value=5.0)
    exchange_rate = st.number_input("Exchange Rate", min_value=0.0, value=1.0)
    
    st.subheader("👤 Account & Device")
    account_age_days = st.number_input("Account Age (Days)", min_value=0, value=365)
    device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.8)
    corridor_risk = st.slider("Corridor Risk", 0.0, 1.0, 0.1)

with col2:
    st.subheader("🕒 Time & Location")
    hour = st.slider("Transaction Hour", 0, 23, 12)
    day_of_week = st.slider("Day of Week", 0, 6, 3)
    
    st.subheader("🌐 Context")
    new_device = st.checkbox("New Device")
    location_mismatch = st.checkbox("Location Mismatch")
    is_weekend = st.checkbox("Weekend Transaction")

    # Categoricals
    home_country = st.selectbox("Home Country", ["CA", "UK", "US", "UNKNOWN"])
    source_currency = st.selectbox("Source Currency", ["CAD", "GBP", "USD"])
    dest_currency = st.selectbox("Destination Currency", ["CAD", "CNY", "EUR", "GBP", "INR", "MXN", "NGN", "PHP", "USD"])
    channel = st.selectbox("Channel", ["ATM", "MOBILE", "WEB", "UNKNOWN"])
    ip_country = st.selectbox("IP Country", ["CA", "UK", "US", "unknown"])
    kyc_tier = st.selectbox("KYC Tier", ["ENHANCED", "LOW", "STANDARD", "unknown"])

st.markdown("---")

# --- 3. PREDICTION LOGIC ---
if st.button("Predict Fraud Risk", use_container_width=True):
    # Prepare the input matching the training columns
    raw_input = pd.DataFrame([{
        "amount_src": amount_src,
        "amount_usd": amount_usd,
        "fee": fee,
        "exchange_rate_src_to_dest": exchange_rate,
        "new_device": new_device,
        "location_mismatch": location_mismatch,
        "account_age_days": account_age_days,
        "device_trust_score": device_trust_score,
        "corridor_risk": corridor_risk,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "home_country": home_country,
        "source_currency": source_currency,
        "dest_currency": dest_currency,
        "channel": channel,
        "ip_country": ip_country,
        "kyc_tier": kyc_tier
    }])

    # Encode using the same logic as training
    encoded = pd.get_dummies(raw_input)

    # Align columns with the model's expected features
    encoded = encoded.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(encoded)[0]
    probability = model.predict_proba(encoded)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"### 🚨 Fraud Detected\n\n**Risk Score: {probability:.2%}**")
    else:
        st.success(f"### ✅ Legitimate Transaction\n\n**Risk Score: {probability:.2%}**")
    
    st.progress(probability)

