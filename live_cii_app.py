import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import joblib
import time
import base64
from io import BytesIO

# --- Page Config ---
st.set_page_config(
    page_title="Live CII Dashboard",
    page_icon="logo_Nakakita.png",  # favicon
    layout="wide"
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_of_bin_file("Nakakita_Logo.png")

# Logo left aligned, heading nearly centered but slightly more left for balance
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: flex-start; gap: 24px; margin-bottom: 20px; width: 100%;">
        <img src="data:image/png;base64,{logo_base64}" alt="Nakakita Logo" style="height:70px; margin-right:32px;">
        <div style="flex: 0 0 70%; max-width: 70%;">
            <h1 style="margin:0; font-size:3.5rem; color:#333366; text-align:left; padding-left:20px;">
                Live CII Dashboard
            </h1>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- App Config ---
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3rem !important;
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
        .cii-btn {
            margin-top: 20px;
            margin-bottom: 10px;
        }
        /* Style the download link as button */
        a.download-link {
            display: inline-block;
            background-color: #3178c6;
            color: white;
            padding: 10px 20px;
            font-weight: bold;
            border-radius: 7px;
            text-decoration: none;
            transition: background-color 0.3s;
        }
        a.download-link:hover {
            background-color: #235a8a;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Constants ---
FUEL_DENSITY = 0.991  # kg/L
CO2_FACTOR = 3.114  # g CO2/g fuel
GT = 14052  # Gross Tonnage
CO2_LIMIT_TONS = 47000  # Target CO2 annual limit in tons

# --- Load model and scaler ---
rf_model = joblib.load("rf_final_model.joblib")
scaler = joblib.load("scaler.joblib")


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


def generate_suggestion(row, predicted_cii):
    suggestions = []
    if predicted_cii > 18:
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


def get_cii_report_df(session_state, predicted_cii, suggestion, range_selector):
    # Aggregates
    cum_fuel_tons = session_state['cum_fuel_liters'] * FUEL_DENSITY / 1000
    cum_co2_tons = session_state['cum_co2_g'] / 1_000_000
    times = session_state['time_vals']
    instant_cii = session_state['instant_cii_vals']

    start_time = times[0] if len(times) else pd.NaT
    end_time = times[-1] if len(times) else pd.NaT
    duration_min = (
        (end_time - start_time).total_seconds() / 60.0
        if (isinstance(start_time, pd.Timestamp) and isinstance(end_time, pd.Timestamp))
        else np.nan
    )

    # Range stats
    min_cii = float(np.nanmin(instant_cii)) if len(instant_cii) else np.nan
    max_cii = float(np.nanmax(instant_cii)) if len(instant_cii) else np.nan
    mean_cii = float(np.nanmean(instant_cii)) if len(instant_cii) else np.nan
    std_cii = float(np.nanstd(instant_cii)) if len(instant_cii) else np.nan

    # Running averages from cumulative rows
    if session_state['cumulative_rows']:
        cr = pd.DataFrame(session_state['cumulative_rows'])
        avg_speed = float(cr['Avg_Speed'].mean())
        avg_wind = float(cr['Avg_Wind_Speed'].mean())
        avg_heel = float(cr['Avg_Heel'].mean())
        avg_trim = float(cr['Avg_Trim'].mean())
        avg_draft = float(cr['Avg_Draft'].mean())
    else:
        avg_speed = avg_wind = avg_heel = avg_trim = avg_draft = np.nan

    # Enriched CSV: repeat summary columns for every time row (simple single-file CSV)
    df = pd.DataFrame({
        "Time": times,
        "Instant CII": instant_cii,
        "Current Predicted CII": [predicted_cii] * len(times),
        "Rating": [assign_rating(predicted_cii)] * len(times),
        "Cumulative Fuel Used (tons)": [cum_fuel_tons] * len(times),
        "Cumulative CO2 Emissions (tons)": [cum_co2_tons] * len(times),
        "Annual CO2 Limit (tons)": [CO2_LIMIT_TONS] * len(times),
        "Avg Speed": [avg_speed] * len(times),
        "Avg Wind Speed": [avg_wind] * len(times),
        "Avg Heel": [avg_heel] * len(times),
        "Avg Trim": [avg_trim] * len(times),
        "Avg Draft": [avg_draft] * len(times),
        "Min Instant CII (range)": [min_cii] * len(times),
        "Max Instant CII (range)": [max_cii] * len(times),
        "Mean Instant CII (range)": [mean_cii] * len(times),
        "Std Instant CII (range)": [std_cii] * len(times),
        "Start Time (range)": [start_time] * len(times),
        "End Time (range)": [end_time] * len(times),
        "Duration (minutes)": [duration_min] * len(times),
        "Voyage Status": ["At Rest" if session_state['at_rest'] else "Underway"] * len(times),
        "Suggestion": [suggestion] * len(times),
        "Range Selector": [range_selector] * len(times)
    })
    return df


def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        towrite = BytesIO()
        object_to_download.to_csv(towrite, index=False)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
    elif isinstance(object_to_download, str):
        b64 = base64.b64encode(object_to_download.encode()).decode()
    else:
        towrite = BytesIO()
        towrite.write(object_to_download)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
    return f'<a class="download-link" href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


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
    ('data_loaded', False),
    ('uploaded_data', None),
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
                data['Fuel_Liters'] = data['FO_ME_Cons'].diff().clip(lower=0)
                st.session_state.uploaded_data = data
                st.success("Data loaded. You can now start the simulation.")
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
    co2_emission = fuel * FUEL_DENSITY * 1000 * CO2_FACTOR
    distance_nm = speed / 60
    st.session_state.cum_fuel_liters += fuel

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

        instant_cii_raw = calculate_cii_1min(fuel, speed)
        if len(st.session_state.instant_cii_vals) < 10:
            if np.isnan(instant_cii_raw):
                instant_cii = np.nan
            else:
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

    # ---------------------------
    # KPI STRIP — moved ABOVE plot
    # ---------------------------
    cols = st.columns(5)
    cols[0].metric("🔹 Instant CII", f"{instant_cii:.2f}" if not np.isnan(instant_cii) else "N/A")
    cols[1].metric("🔸 Predicted CII", f"{predicted_cii:.2f}" if not np.isnan(predicted_cii) else "N/A")
    cols[1].markdown(
        f"<span style='font-size:1.5rem;font-weight:bold;color:#333366;'>Rating: {assign_rating(predicted_cii)}</span>",
        unsafe_allow_html=True
    )
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

    # ---------------------------
    # Line Plot
    # ---------------------------
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

    # Single Download Button styled as a button
    cii_report_df = get_cii_report_df(
        st.session_state, predicted_cii, suggestion, st.session_state.range_selector
    )
    st.markdown('<div class="cii-btn"></div>', unsafe_allow_html=True)
    if st.button("⬇️ Download CII Report (CSV)"):
        tmp_download_link = download_link(cii_report_df, "cii_report_summary.csv", "Click here to download your CII Report")
        st.markdown(tmp_download_link, unsafe_allow_html=True)

    # --- Optional: At-rest overlay (unchanged visual, appears over chart) ---
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
        cols[1].markdown(f"<span style='font-size:1.5rem;font-weight:bold;color:#333366;'>Rating: {assign_rating(predicted_cii)}</span>", unsafe_allow_html=True)
        cols[2].metric("⛽ Cumulative Fuel Used (tons)", f"{cum_fuel_tons:.2f}")
        cols[3].markdown(
            f"<div class='fuel-limit-box'>"
            f"<b>🌱 Cumulative CO₂ Emissions (tons)</b>: {cum_co2_tons:.2f}<br>"
            f"<span style='color:#888'>Annual Limit: {CO2_LIMIT_TONS:,} tons</span>"
            f"</div>", unsafe_allow_html=True
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.session_state.i += 1
    time.sleep(1)
    st.rerun()

elif st.session_state.data_loaded:
    if st.session_state.instant_cii_vals:
        predicted_cii = st.session_state.instant_cii_vals[-1] if st.session_state.instant_cii_vals else np.nan
        rating = assign_rating(predicted_cii)
        suggestion = "No simulation currently running"
        cii_report_df = get_cii_report_df(
            st.session_state, predicted_cii, suggestion, st.session_state.range_selector
        )
        st.markdown('<div class="cii-btn"></div>', unsafe_allow_html=True)
        st.download_button(
    label="⬇️ Download CII Report (CSV)",
    data=cii_report_df.to_csv(index=False),
    file_name="cii_report_summary.csv",
    mime="text/csv"
)

