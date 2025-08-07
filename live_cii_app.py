import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import joblib
import time


# --- App Config ---
st.set_page_config(page_title="Live CII Dashboard", layout="wide")
st.markdown(
    """
    <style>
        html, body, .block-container, .main, #root, [data-testid="stAppViewContainer"] {
            width: 100vw !important;
            height: 100vh !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 20px !important;
        }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            overflow: auto !important;
            max-width: 100vw !important;
            position: relative;
        }
        .stPlotlyChart {
            height: 100% !important;
            min-height: 400px !important;
            max-height: 60vh !important;
        }
        html, body {
            overflow-y: hidden !important;
            width: 100vw !important;
            height: 100vh !important;
        }
        h2, .css-10trblm {
            font-size: 2rem !important;
            margin-top: 20px !important;
            margin-bottom: 20px !important;
            overflow: visible !important;
            white-space: normal !important;
        }
        .element-container {
            width: 100vw !important;
        }
        .fuel-limit-box {
            font-size: 1.2rem !important;
            padding: 7px 15px 7px 15px;
            background: #f5f6fa;
            border-radius: 7px;
            margin-left: 15px;
            margin-top: 7px;
            display: inline-block;
            color: #333366;
            font-family: monospace;
        }
        .overlay {
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background-color: rgba(255,255,255,0.8);
            backdrop-filter: blur(5px);
            z-index: 9999;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .overlay-text {
            font-size: 2rem;
            font-weight: bold;
            color: #333366;
            margin-bottom: 20px;
        }
        .overlay-stats {
            font-size: 1.2rem;
            color: #555555;
            max-width: 90vw;
            overflow-y: auto;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 8px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
            width: 60vw;
            max-height: 60vh;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Constants ---
FUEL_DENSITY = 0.991  # kg/L
CO2_FACTOR = 3.114    # g CO2/g fuel
GT = 14052            # Gross Tonnage
CO2_LIMIT_TONS = 47000  # Target CO2 annual limit in tons


# --- Load model and scaler ---
rf_model = joblib.load("rf_final_model.joblib")
scaler = joblib.load("scaler.joblib")


# --- Util: Outlier removal ---
def is_cii_outlier(previous_cii, new_cii, max_val=100):
    if len(previous_cii) == 0:
        return False
    rolling_median = np.median(previous_cii[-10:])
    if new_cii < 0 or new_cii > max_val:
        return True
    if rolling_median > 0 and new_cii > 3 * rolling_median:
        return True
    return False


def calculate_cii_1min(fuel_liters, ship_speed):
    if ship_speed > 0:
        co2 = fuel_liters * FUEL_DENSITY * 1000 * CO2_FACTOR
        distance_nm = ship_speed / 60
        return co2 / (GT * distance_nm) if distance_nm > 0 else np.nan
    else:
        return np.nan


def assign_rating(predicted_cii):
    if predicted_cii <= 17.75:
        return 'A'
    elif predicted_cii <= 19.20:
        return 'B'
    elif predicted_cii <= 21.45:
        return 'C'
    elif predicted_cii <= 23.70:
        return 'D'
    else:
        return 'E'


# Improved suggestion logic based on predicted CII and thresholds
def generate_suggestion(row, predicted_cii):
    suggestions = []

    # Threshold for when suggestions are relevant (worse performance)
    if predicted_cii > 18:  # example threshold for worse CII grades B or lower
        if row['Trim'] > 1.0:
            suggestions.append("Reduce trim to improve fuel efficiency")
        if row['HEEL'] > 0.5:
            suggestions.append("Balance ballast to reduce heel")
        if row['Wind_Speed'] > 12:
            suggestions.append("Avoid sailing during high wind")
        if row['CppPitch'] < 15:
            suggestions.append("Increase CPP pitch for propulsion efficiency")
        if row['Ship_Speed'] < 17:
            suggestions.append("Maintain optimal cruising speed")

    if not suggestions:
        return "Performance is within expected range"
    else:
        return "; ".join(suggestions)


# --- Session State Initialization ---
for v, default in [
    ('i', 1),
    ('cumulative_rows', []),
    ('instant_cii_vals', []),
    ('time_vals', []),
    ('cum_co2_g', 0),
    ('cum_fuel_liters', 0),
    ('running', False),
    ('range_selector', "All"),
    ('at_rest', False),
    ('data_loaded', False)
]:
    if v not in st.session_state:
        st.session_state[v] = default


st.markdown("## 🚢 Real-Time Instant CII Plot (Live Monitor)")


# --- Upload Section ---
if not st.session_state.data_loaded:
    st.info("Please upload your ship data CSV file to begin.")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded, low_memory=False)
            required_cols = {'FO_ME_Cons', 'Ship_Speed', 'CppPitch', 'Wind_Speed', 'HEEL', 'Fore_Draft', 'Aft_Draft', 'Time'}
            if not required_cols.issubset(data.columns):
                st.error("CSV file missing required columns.")
            else:
                data['Time'] = pd.to_datetime(data['Time'], format='%d-%m-%Y %H:%M')
                data = data.sort_values('Time').reset_index(drop=True)
                for col in ['FO_ME_Cons', 'Ship_Speed', 'CppPitch', 'Wind_Speed', 'HEEL', 'Fore_Draft', 'Aft_Draft']:
                    data[col] = pd.to_numeric(data[col], errors='coerce').ffill()
                data = data.dropna()
                data['Trim'] = data['Aft_Draft'] - data['Fore_Draft']
                data['Avg_Draft'] = (data['Fore_Draft'] + data['Aft_Draft']) / 2
                
                # Calculate fuel consumption differences safely
                data['Fuel_Liters'] = data['FO_ME_Cons'].diff().clip(lower=0)
                
                st.session_state.uploaded_data = data
                st.success("Data loaded. You can now start the simulation.")
                
                # Show Start Simulation button only when simulation is not started
                if not st.session_state.running:
                    if st.button("Start Simulation"):
                        st.session_state.data_loaded = True
                        st.session_state.i = 1
                        st.session_state.cumulative_rows = []
                        st.session_state.instant_cii_vals = []
                        st.session_state.time_vals = []
                        st.session_state.cum_co2_g = 0
                        st.session_state.cum_fuel_liters = 0
                        st.session_state.running = True
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
    st.stop()
else:
    data = st.session_state.uploaded_data


# --- Simulation Controls ---
col1, col2 = st.columns([1, 8])
with col1:
    if st.button("⏹ Stop Simulation" if st.session_state.running else "▶ Start Simulation"):
        st.session_state.running = not st.session_state.running
with col2:
    if st.session_state.running:
        st.session_state.range_selector = st.selectbox(
            "View Range", ["1 Hour", "1 Day", "1 Week", "1 Month", "All"], key="range"
        )


# --- Live Simulation ---
if st.session_state.running and st.session_state.i < len(data):
    row = data.iloc[st.session_state.i]
    fuel = row['Fuel_Liters']
    speed = row['Ship_Speed']

    co2_emission = fuel * FUEL_DENSITY * 1000 * CO2_FACTOR  # grams CO2
    distance_nm = speed / 60

    # Always update cumulative fuel (even at rest)
    st.session_state.cum_fuel_liters += fuel

    # Detect if ship is at rest
    if speed < 0.5:
        st.session_state.at_rest = True
    else:
        if st.session_state.at_rest:
            st.toast("Voyage started")
        st.session_state.at_rest = False

    if not st.session_state.at_rest:
        st.session_state.cumulative_rows.append({
            'CO2_Emission_g': co2_emission,
            'Distance_NM': distance_nm,
            'Avg_Speed': row['Ship_Speed'],
            'Avg_Wind_Speed': row['Wind_Speed'],
            'Avg_CppPitch': row['CppPitch'],
            'Avg_Heel': row['HEEL'],
            'Avg_Trim': row['Trim'],
            'Avg_Draft': row['Avg_Draft']
        })
        st.session_state.cum_co2_g += co2_emission

        cumulative_df = pd.DataFrame(st.session_state.cumulative_rows)
        processed_input = pd.DataFrame([{
            'CO2_Emission_g': cumulative_df['CO2_Emission_g'].sum(),
            'Distance_NM': cumulative_df['Distance_NM'].sum(),
            'Avg_Speed': cumulative_df['Avg_Speed'].mean(),
            'Avg_Wind_Speed': cumulative_df['Avg_Wind_Speed'].mean(),
            'Avg_CppPitch': cumulative_df['Avg_CppPitch'].mean(),
            'Avg_Heel': cumulative_df['Avg_Heel'].mean(),
            'Avg_Trim': cumulative_df['Avg_Trim'].mean(),
            'Avg_Draft': cumulative_df['Avg_Draft'].mean()
        }])

        scaled_input = scaler.transform(processed_input)
        predicted_cii = rf_model.predict(scaled_input)[0]

        # To reduce initial spike, use rolling mean for first 10 values for instant CII
        instant_cii_raw = calculate_cii_1min(fuel, speed)
        if len(st.session_state.instant_cii_vals) < 10:
            if np.isnan(instant_cii_raw):
                instant_cii = np.nan
            else:
                # Use simple smoothing by averaging last few valid points + new
                recent = [v for v in st.session_state.instant_cii_vals[-5:] if not np.isnan(v)]
                if recent:
                    instant_cii = (sum(recent) + instant_cii_raw) / (len(recent) + 1)
                else:
                    instant_cii = instant_cii_raw
        else:
            instant_cii = instant_cii_raw

        suggestion = generate_suggestion(row, predicted_cii)

        if not is_cii_outlier(st.session_state.instant_cii_vals, instant_cii, max_val=100):
            st.session_state.instant_cii_vals.append(instant_cii)
            st.session_state.time_vals.append(row['Time'])

    else:
        predicted_cii = st.session_state.instant_cii_vals[-1] if st.session_state.instant_cii_vals else np.nan
        instant_cii = predicted_cii
        suggestion = "Ship is at rest"

    df_plot = pd.DataFrame({
        "Time": st.session_state.time_vals,
        "Instant_CII": st.session_state.instant_cii_vals
    })

    if st.session_state.range_selector != "All" and not df_plot.empty:
        time_cutoff = {
            "1 Hour": pd.Timedelta(hours=1),
            "1 Day": pd.Timedelta(days=1),
            "1 Week": pd.Timedelta(weeks=1),
            "1 Month": pd.Timedelta(days=30)
        }[st.session_state.range_selector]
        latest_time = df_plot["Time"].iloc[-1]
        df_plot = df_plot[df_plot["Time"] >= latest_time - time_cutoff]

    # Display overlay when ship is at rest
    if st.session_state.at_rest:
        st.markdown(
            """
            <div class="overlay">
                <div class="overlay-text">The ship is currently at rest.</div>
                <div class="overlay-stats">
            """, unsafe_allow_html=True)
        cols = st.columns(4)
        cols[0].metric("🔹 Instant CII", f"{instant_cii:.2f}" if not np.isnan(instant_cii) else "N/A")
        cols[1].metric("🔸 Predicted CII", f"{predicted_cii:.2f}" if not np.isnan(predicted_cii) else "N/A")
        # Show rating under predicted CII
        rating = assign_rating(predicted_cii)
        cols[1].markdown(f"<span style='font-size:1.5rem;font-weight:bold;color:#333366;'>Rating: {rating}</span>", unsafe_allow_html=True)
        
        cum_fuel_tons = st.session_state.cum_fuel_liters * FUEL_DENSITY / 1000
        cols[2].metric("⛽ Cumulative Fuel Used (tons)", f"{cum_fuel_tons:.2f}")
        cum_co2_tons = st.session_state.cum_co2_g / 1_000_000
        cols[3].markdown(
            f"<div class='fuel-limit-box'>"
            f"<b>🌱 Cumulative CO₂ Emissions (tons)</b>: {cum_co2_tons:.2f}<br>"
            f"<span style='color:#888'>Annual Limit: {CO2_LIMIT_TONS:,} tons</span>"
            f"</div>", unsafe_allow_html=True
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

    else:
        cols = st.columns(5)
        cols[0].metric("🔹 Instant CII", f"{instant_cii:.2f}" if not np.isnan(instant_cii) else "N/A")
        cols[1].metric("🔸 Predicted CII", f"{predicted_cii:.2f}" if not np.isnan(predicted_cii) else "N/A")
        # Show rating under predicted CII
        rating = assign_rating(predicted_cii)
        cols[1].markdown(f"<span style='font-size:1.5rem;font-weight:bold;color:#333366;'>Rating: {rating}</span>", unsafe_allow_html=True)
        
        cum_fuel_tons = st.session_state.cum_fuel_liters * FUEL_DENSITY / 1000
        cols[2].metric("⛽ Cumulative Fuel Used (tons)", f"{cum_fuel_tons:.2f}")
        cum_co2_tons = st.session_state.cum_co2_g / 1_000_000
        cols[3].markdown(
            f"<div class='fuel-limit-box'>"
            f"<b>🌱 Cumulative CO₂ Emissions (tons)</b>: {cum_co2_tons:.2f}<br>"
            f"<span style='color:#888'>Annual Limit: {CO2_LIMIT_TONS:,} tons</span>"
            f"</div>", unsafe_allow_html=True
        )
        cols[4].success(f"💡 {suggestion}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['Time'],
            y=df_plot['Instant_CII'],
            mode='lines+markers',
            name='Instant CII',
            line=dict(color='royalblue', width=2),
            marker=dict(size=6),
            hovertemplate='Time: %{x|%Y-%m-%d %H:%M}<br>CII: %{y:.2f}<extra></extra>'
        ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="Time",
            yaxis_title="Instant CII",
            xaxis=dict(rangeslider_visible=False),
            height=360,
            template="plotly_white",
            showlegend=False,
            uirevision="stay",
            dragmode='pan'
        )
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False, automargin=True)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "doubleClick": "reset",
                "displaylogo": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                "responsive": True
            }
        )

    st.session_state.i += 1
    time.sleep(1)
    st.rerun()
