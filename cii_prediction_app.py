import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import base64  # add this import for base64 encoding

# --- Page Config ---
st.set_page_config(
    page_title="CII Prediction & Optimization",
    page_icon="logo_Nakakita.png",  # Replace with your favicon image path or a small icon file
    layout="wide"
)

# Function to convert image file to base64 encoding
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load your company logo as base64 - replace 'company_logo.png' with your logo file path
logo_base64 = get_base64_of_bin_file("Nakakita_Logo.png")

# Display logo left aligned with heading next to it
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 24px; margin-bottom: 20px; width: 100%;">
        <img src="data:image/png;base64,{logo_base64}" alt="Company Logo" style="height:70px; margin-right:32px;">
        <h1 style="margin:0; font-size:3.5rem; color:#333366; text-align:left; padding-left:0;">
            CII Prediction & Optimization
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Your existing code continues here...
# Load trained model and scaler
model = joblib.load('rf_final_model.joblib')
scaler = joblib.load('scaler.joblib')

# Final features used in training etc...
# (rest of your original code)


# Load trained model and scaler
model = joblib.load('rf_final_model.joblib')
scaler = joblib.load('scaler.joblib')

# Final features used in training
features = [
    'CO2_Emission_g',
    'Distance_NM',
    'Avg_Speed',
    'Avg_Wind_Speed',
    'Avg_CppPitch',
    'Avg_Heel',
    'Avg_Trim',
    'Avg_Draft'
]

GT = 14052  # Gross tonnage
FUEL_DENSITY = 0.991
CO2_FACTOR = 3.114

# Rating function
def assign_rating(cii):
    if cii < 17.5:
        return 'A'
    elif cii < 19.20:
        return 'B'
    elif cii < 21.4:
        return 'C'
    elif cii < 23.70:
        return 'D'
    else:
        return 'E'

# Suggestions function
def generate_suggestion(row):
    suggestions = []
    if row['Avg_Trim'] > 1.0:
        suggestions.append("Reduce trim to improve fuel efficiency")
    if row['Avg_Heel'] > 0.5:
        suggestions.append("Balance ballast to reduce heel")
    if row['Avg_Wind_Speed'] > 12:
        suggestions.append("Avoid sailing during high wind")
    if row['Avg_CppPitch'] < 15:
        suggestions.append("Increase CPP pitch for propulsion efficiency")
    if row['Avg_Speed'] < 17:
        suggestions.append("Maintain optimal cruising speed")
    return "; ".join(suggestions) if suggestions else "Performance is within expected range"

# Preprocessing for CSV uploads
def preprocess_csv(df):
    # Ensure Time is datetime
    df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M', errors='coerce')

    # Convert numeric columns and forward fill
    for col in ['FO_ME_Cons', 'FO_GE_Cons', 'Ship_Speed', 'CppPitch', 'Wind_Speed', 'HEEL', 'Fore_Draft', 'Aft_Draft']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill')

    # Drop rows with missing critical values
    df = df.dropna(subset=['FO_ME_Cons', 'FO_GE_Cons', 'Ship_Speed', 'CppPitch', 'Wind_Speed', 'HEEL', 'Fore_Draft', 'Aft_Draft'])

    # Remove idle/port states
    df = df[df['Ship_Speed'] > 0.5]

    # Calculate fuel consumption per row
    fuel_me = df['FO_ME_Cons'].diff().clip(lower=0)
    fuel_ge = df['FO_GE_Cons'].diff().clip(lower=0)
    df['Fuel_Liters'] = fuel_me + fuel_ge

    # Calculate Trim and Avg Draft
    df['Trim'] = df['Aft_Draft'] - df['Fore_Draft']
    df['Avg_Draft'] = (df['Aft_Draft'] + df['Fore_Draft']) / 2

    # Calculate CO2 emissions and distance
    fuel_kg = df['Fuel_Liters'] * FUEL_DENSITY
    df['CO2_Emission_g'] = fuel_kg * 1000 * CO2_FACTOR
    df['Distance_NM'] = df['Ship_Speed'] / 60  # per minute

    # Aggregate daily
    df['Day'] = df['Time'].dt.to_period('D')
    daily_df = df.groupby('Day').agg({
        'CO2_Emission_g': 'sum',
        'Distance_NM': 'sum',
        'Ship_Speed': 'mean',
        'Wind_Speed': 'mean',
        'CppPitch': 'mean',
        'HEEL': 'mean',
        'Trim': 'mean',
        'Avg_Draft': 'mean'
    }).reset_index()

    daily_df.rename(columns={
        'Ship_Speed': 'Avg_Speed',
        'Wind_Speed': 'Avg_Wind_Speed',
        'CppPitch': 'Avg_CppPitch',
        'HEEL': 'Avg_Heel',
        'Trim': 'Avg_Trim'
    }, inplace=True)

    return daily_df

# Streamlit UI
st.title("CII Prediction & Optimization")

# Manual Entry
st.header("Manual Entry (Daily Aggregated Values)")
with st.form("manual_entry"):
    inputs = {}
    for col in features:
        inputs[col] = st.number_input(f"{col}", min_value=0.0)
    submitted = st.form_submit_button("Predict")
    if submitted:
        df_input = pd.DataFrame([inputs])
        df_scaled = scaler.transform(df_input[features])
        pred_cii = model.predict(df_scaled)[0]
        rating = assign_rating(pred_cii)
        suggestion = generate_suggestion(inputs)
        st.markdown(f"**Predicted CII:** {pred_cii:.2f}")
        st.markdown(f"**Rating:** {rating}")
        st.markdown(f"**Suggestions:** {suggestion}")

# CSV Upload
st.header("CSV Upload (Raw Ship Logs)")
csv_file = st.file_uploader("Upload CSV file", type=["csv"])
if csv_file:
    df = pd.read_csv(csv_file)
    required_cols = ['Time', 'FO_ME_Cons', 'FO_GE_Cons', 'Ship_Speed', 'CppPitch', 'Wind_Speed', 'HEEL', 'Fore_Draft', 'Aft_Draft']

    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV missing required columns. Required: {required_cols}")
    else:
        daily_df = preprocess_csv(df)
        df_scaled = scaler.transform(daily_df[features])
        daily_df['Predicted_CII'] = model.predict(df_scaled)
        daily_df['Relative_Rating'] = daily_df['Predicted_CII'].apply(assign_rating)
        daily_df['Optimization_Suggestions'] = daily_df.apply(generate_suggestion, axis=1)

        st.dataframe(daily_df.round(2))
        st.download_button(
            "Download Results",
            daily_df.to_csv(index=False),
            "cii_prediction_results.csv",
            "text/csv"
        )
