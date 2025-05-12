import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Ensure 'model.pkl' exists in the 'models' folder.")

st.title("🔧 Predictive Maintenance - AI Dashboard")
st.markdown("Use the controls on the left to enter machine parameters and predict failures.")



# Sidebar Inputs
with st.sidebar:
    st.header("🛠️ Machine Parameters")

    product_type = st.selectbox("Product Type", ['L', 'M', 'H'])
    air_temp = st.slider("Air Temperature (°C)", 10.0, 100.0, 25.0)
    process_temp = st.slider("Process Temperature (°C)", 20.0, 120.0, 50.0)
    rot_speed = st.slider("Rotational Speed (rpm)", 500.0, 2000.0, 1550.0)
    torque = st.slider("Torque (Nm)", 0.0, 300.0, 40.0)
    tool_wear = st.slider("Tool Wear (min)", 0.0, 300.0, 0.0)

    TWF = st.number_input("Tool Wear Failure (TWF)", value=0.0)
    HDF = st.number_input("Heat Dissipation Failure (HDF)", value=0.0)
    PWF = st.number_input("Power Failure (PWF)", value=0.0)
    OSF = st.number_input("Overstrain Failure (OSF)", value=0.0)
    RNF = st.number_input("Random Failure (RNF)", value=0.0)

type_dict = {'L': 1, 'M': 2, 'H': 0}
type_encoded = type_dict[product_type]

input_df = pd.DataFrame([{
    'Type': type_encoded,
    'Air temperature [K]': air_temp + 273.15,
    'Process temperature [K]': process_temp + 273.15,
    'Rotational speed [rpm]': rot_speed,
    'Torque [Nm]': torque,
    'Tool wear [min]': tool_wear,
    'TWF': TWF,
    'HDF': HDF,
    'PWF': PWF,
    'OSF': OSF,
    'RNF': RNF
}])

# Prediction
prediction = model.predict(input_df)[0]
probas = model.predict_proba(input_df)[0]
failure_prob = round(probas[1] * 100, 2)
normal_prob = round(probas[0] * 100, 2)

# Metrics
st.subheader("🔍 Prediction Overview")
col1, col2 = st.columns(2)
col1.metric("🟢 No Failure", f"{normal_prob}%", delta=f"{normal_prob - 50:.2f}")
col2.metric("🔴 Failure", f"{failure_prob}%", delta=f"{failure_prob - 50:.2f}")

if prediction == 1:
    st.error(f"⚠️ Machine Failure Likely (Confidence: {failure_prob}%)")
else:
    st.success(f"✅ Machine Operating Normally (Confidence: {normal_prob}%)")

# Expandable Input Summary
with st.expander("📈 Input Feature Summary"):
    fig, ax = plt.subplots()
    ax.barh(input_df.columns, input_df.values[0], color='skyblue')
    ax.set_xlabel("Value")
    ax.set_title("Input Features Overview")
    st.pyplot(fig)

# Suggestions
with st.expander("💡 Suggestions"):
    suggestions = []

    if torque > 120:
        suggestions.append("🔧 High Torque: Check for misalignment or shaft overload.")
    if tool_wear > 150:
        suggestions.append("🛠️ Excessive Tool Wear: Replace the tool and inspect cutting surfaces.")
    if OSF > 50:
        suggestions.append("📦 Overstrain: Inspect load conditions and reduce mechanical stress.")
    if air_temp > 40:
        suggestions.append("🌡️ High Air Temperature: Improve ventilation or cooling systems.")
    if process_temp > 75:
        suggestions.append("🔥 High Process Temp: Verify sensors and check insulation.")
    if rot_speed > 1700:
        suggestions.append("⚙️ High Rotational Speed: Inspect bearings and rebalance the motor.")
    if HDF > 0:
        suggestions.append("💨 Heat Dissipation Failure: Clean cooling components or check thermal paste.")
    if PWF > 0:
        suggestions.append("🔌 Power Failure: Check electrical connections and input voltage.")
    if RNF > 0:
        suggestions.append("❗ Random Failure: Perform full diagnostic or sensor calibration.")

    if suggestions:
        for s in suggestions:
            st.warning(s)
    else:
        st.success("✅ All parameters within safe operating range.")

# Risk Flags
with st.expander("🚨 Risk Flags"):
    risk_flags = []

    if torque > 120:
        risk_flags.append("⚠️ High Torque")
    if tool_wear > 150:
        risk_flags.append("⚠️ Excessive Tool Wear")
    if OSF > 50:
        risk_flags.append("⚠️ Overstrain Risk")
    if air_temp > 40:
        risk_flags.append("⚠️ High Air Temperature")
    if process_temp > 75:
        risk_flags.append("⚠️ High Process Temperature")
    if rot_speed > 1700:
        risk_flags.append("⚠️ High Rotational Speed")
    if HDF > 0:
        risk_flags.append("⚠️ Heat Dissipation Issue")
    if PWF > 0:
        risk_flags.append("⚠️ Power Failure")
    if RNF > 0:
        risk_flags.append("⚠️ Random Failure")

    if risk_flags:
        for flag in risk_flags:
            st.warning(flag)
    else:
        st.success("✅ No critical risk indicators.")

# Logging Prediction
log_data = input_df.copy()
log_data["Prediction"] = "Failure" if prediction == 1 else "No Failure"
log_data["Failure_Prob"] = failure_prob
log_data["Normal_Prob"] = normal_prob
log_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_data["Risk_Flags"] = "; ".join(risk_flags) if risk_flags else "None"

log_file = "prediction_log.csv"
if os.path.exists(log_file):
    log_data.to_csv(log_file, mode='a', header=False, index=False)
else:
    log_data.to_csv(log_file, mode='w', header=True, index=False)

# Download log
if os.path.exists(log_file):
    with open(log_file, "rb") as f:
        st.download_button("📥 Download Prediction Log", f, file_name="prediction_log.csv")
