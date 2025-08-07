# Streamlit app to simulate a CII dashboard with real-time & predictive CII

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

# Load trained model and scaler
model = joblib.load('model1_rf.pkl')  # Your trained Random Forest model
scaler = joblib.load('scaler.pkl')      # Corresponding scaler

# Features used in the model
features = ['CO2_Emission_g', 'Distance_NM', 'Avg_Speed', 'Avg_Wind_Speed', 
            'Avg_CppPitch', 'Avg_Heel', 'Avg_Trim', 'Avg_Draft']

# Helper functions
def calculate_real_time_cii(fuel_g, distance_nm, gt):
    emission_factor = 3.114
    co2_emission = fuel_g * emission_factor
    return co2_emission / (gt * distance_nm) if distance_nm > 0 else np.nan

def assign_rating(cii):
    if cii < 2:
        return 'A'
    elif cii < 3:
        return 'B'
    elif cii < 4.5:
        return 'C'
    elif cii < 6.5:
        return 'D'
    else:
        return 'E'

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

# Streamlit UI setup
st.set_page_config(page_title="Live CII Dashboard", layout="wide")
st.title("📊 Real-Time CII Monitoring Dashboard")
st.markdown("Simulated live updates with real-time & annual prediction")

# Load and preprocess sample data
live_data = pd.read_csv("sample_data_year.csv")
live_data['Time'] = pd.to_datetime(live_data['Time'], format='%d-%m-%Y %H:%M')
live_data = live_data.sort_values('Time').reset_index(drop=True)

cols_to_clean = ['FO_ME_Cons', 'FO_GE_Cons', 'Ship_Speed', 'CppPitch', 'Wind_Speed',
                 'HEEL', 'Fore_Draft', 'Aft_Draft']

for col in cols_to_clean:
    live_data[col] = pd.to_numeric(live_data[col], errors='coerce')

# Compute fuel consumption properly
fuel_me = live_data['FO_ME_Cons'].diff().clip(lower=0)
fuel_ge = (-live_data['FO_GE_Cons'].diff()).clip(lower=0)
live_data['Fuel_Liters'] = fuel_me + fuel_ge
live_data.fillna(method='ffill', inplace=True)
live_data.dropna(inplace=True)

# Additional feature engineering
live_data['Trim'] = live_data['Aft_Draft'] - live_data['Fore_Draft']
live_data['Avg_Draft'] = (live_data['Fore_Draft'] + live_data['Aft_Draft']) / 2
live_data['CO2_Emission_g'] = live_data['Fuel_Liters'] * 0.991 * 1000 * 3.114
live_data['Distance_NM'] = live_data['Ship_Speed'] / 60

# Set GT
gt = 14052

# Simulate ticker updates
predicted_cii_values = []
live_cii_values = []
time_points = []

for i in range(30):  # Simulate first 30 rows for demo
    row = live_data.iloc[i]

    # Real-time CII
    fuel = row['Fuel_Liters'] * 0.991
    dist = row['Distance_NM']
    real_time_cii = calculate_real_time_cii(fuel, dist, gt)

    # Predictive Annual CII from data up to now
    temp_df = live_data.iloc[:i+1].copy()
    daily_avg = temp_df[features].mean().to_frame().T
    daily_scaled = scaler.transform(daily_avg)
    predicted_annual_cii = model.predict(daily_scaled)[0]

    # Store for plotting
    live_cii_values.append(real_time_cii)
    predicted_cii_values.append(predicted_annual_cii)
    time_points.append(row['Time'])

    # Show ticker
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Real-Time CII", value=f"{real_time_cii:.2f}", delta=assign_rating(real_time_cii))
    with col2:
        st.metric(label="Predicted Annual CII", value=f"{predicted_annual_cii:.2f}", delta=assign_rating(predicted_annual_cii))

    # Suggestion
    suggestion = generate_suggestion(row)
    st.markdown(f"**Suggestion:** {suggestion}")

    # Graph
    st.line_chart(pd.DataFrame({
        'Live CII': live_cii_values,
        'Predicted CII': predicted_cii_values
    }, index=time_points))
   
    time.sleep(1)
    st.stop()