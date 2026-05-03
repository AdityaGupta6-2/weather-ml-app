import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WeatherML",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e0e6f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 25, 45, 0.95) !important;
    border-right: 1px solid rgba(100, 180, 255, 0.15);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,180,255,0.12);
    border-radius: 14px;
    padding: 16px;
}

/* Headers */
h1 { font-family: 'Space Mono', monospace !important; color: #7ecfff !important; }
h2, h3 { color: #c5deff !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a6bff, #0ea5e9);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div > select {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(100,180,255,0.2) !important;
    border-radius: 10px !important;
    color: #e0e6f0 !important;
}

/* Info boxes */
.stInfo {
    background: rgba(14, 165, 233, 0.1) !important;
    border: 1px solid rgba(14, 165, 233, 0.3) !important;
    border-radius: 12px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #8ba0c0;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(30, 107, 255, 0.3) !important;
    color: #7ecfff !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #0ea5e9 !important;
}

/* Divider */
hr { border-color: rgba(100,180,255,0.12) !important; }

/* Plotly charts */
.js-plotly-plot {
    border-radius: 14px;
    overflow: hidden;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(100,180,255,0.12);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  WEATHER CODE → ICON
# ─────────────────────────────────────────────
def get_icon(code):
    if pd.isna(code): return "🌡️"
    code = int(code)
    if code == 0: return "☀️"
    elif code <= 2: return "🌤️"
    elif code <= 3: return "☁️"
    elif code <= 49: return "🌫️"
    elif code <= 59: return "🌦️"
    elif code <= 69: return "🌧️"
    elif code <= 79: return "❄️"
    elif code <= 84: return "🌨️"
    else: return "⛈️"

# ─────────────────────────────────────────────
#  GEOCODING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def geocode(city):
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=10
        )
        data = r.json()
        if "results" not in data:
            return None
        loc = data["results"][0]
        return {
            "label": f"{loc['name']}, {loc.get('country', '')}",
            "lat": loc["latitude"],
            "lon": loc["longitude"]
        }
    except:
        return None

# ─────────────────────────────────────────────
#  FETCH HISTORICAL DATA (2 years)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_historical(lat, lon):
    end = datetime.now() - timedelta(days=5)
    start = end - timedelta(days=730)
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode",
                "timezone": "auto"
            }, timeout=20
        )
        data = r.json()
        if "daily" not in data:
            return None
        d = data["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(d["time"]),
            "temp_max": d["temperature_2m_max"],
            "temp_min": d["temperature_2m_min"],
            "rain": d["precipitation_sum"],
            "wind": d["windspeed_10m_max"],
            "code": d["weathercode"]
        }).dropna()
        return df
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return None

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(df, lag_days=14):
    df = df.copy().reset_index(drop=True)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    for col in ["temp_max", "temp_min", "rain", "wind"]:
        for lag in [1, 3, 7, 14]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_roll7"] = df[col].rolling(7).mean()
        df[f"{col}_roll14"] = df[col].rolling(14).mean()

    return df.dropna()

# ─────────────────────────────────────────────
#  TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
def train_model(df):
    feature_cols = [c for c in df.columns if c not in
                    ["date", "temp_max", "temp_min", "rain", "wind", "code", "day_of_year"]]
    target_cols = ["temp_max", "temp_min", "rain", "wind"]

    X = df[feature_cols].values
    y = df[target_cols].values

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    )
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    metrics = {}
    for i, col in enumerate(target_cols):
        metrics[col] = {
            "mae": round(mean_absolute_error(y_test[:, i], preds[:, i]), 2),
            "r2": round(r2_score(y_test[:, i], preds[:, i]), 3)
        }

    return model, scaler, feature_cols, target_cols, metrics, df[target_cols].values, preds

# ─────────────────────────────────────────────
#  PREDICT NEXT 30 DAYS (RECURSIVE)
# ─────────────────────────────────────────────
def predict_future(df, model, scaler, feature_cols, days=30):
    df_future = df.copy()
    predictions = []

    for i in range(days):
        df_feat = build_features(df_future)
        last = df_feat.iloc[[-1]]
        X = last[feature_cols].values
        X_s = scaler.transform(X)
        pred = model.predict(X_s)[0]

        next_date = df_future["date"].iloc[-1] + timedelta(days=1)
        new_row = pd.DataFrame({
            "date": [next_date],
            "temp_max": [pred[0]],
            "temp_min": [pred[1]],
            "rain": [max(0, pred[2])],
            "wind": [max(0, pred[3])],
            "code": [df_future["code"].iloc[-1]]
        })
        df_future = pd.concat([df_future, new_row], ignore_index=True)
        predictions.append({
            "date": next_date,
            "temp_max": round(pred[0], 1),
            "temp_min": round(pred[1], 1),
            "rain": round(max(0, pred[2]), 1),
            "wind": round(max(0, pred[3]), 1),
        })

    return pd.DataFrame(predictions)

# ─────────────────────────────────────────────
#  PLOTLY CHART STYLE
# ─────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="DM Sans", color="#c5deff"),
    xaxis=dict(gridcolor="rgba(100,180,255,0.08)", zeroline=False),
    yaxis=dict(gridcolor="rgba(100,180,255,0.08)", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(100,180,255,0.1)")
)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌦️ WeatherML")
    st.markdown("*Random Forest · 30-Day Forecast*")
    st.divider()

    city_input = st.text_input("📍 City", value="Delhi", placeholder="Enter city name...")
    run_btn = st.button("🚀 Run Prediction", use_container_width=True)

    st.divider()
    st.markdown("**Model Config**")
    st.markdown("- 🌲 Random Forest (200 trees)")
    st.markdown("- 📅 2 years of training data")
    st.markdown("- 🔄 14-day lag features")
    st.markdown("- 📈 Rolling averages (7 & 14 day)")
    st.divider()
    st.markdown("**Predicting**")
    st.markdown("- 🌡️ Max & Min Temperature")
    st.markdown("- 🌧️ Precipitation")
    st.markdown("- 💨 Wind Speed")
    st.markdown("- ⏳ 30 days ahead")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
st.markdown("# 🌦️ WeatherML Forecast")
st.markdown("Machine Learning weather prediction powered by Random Forest")
st.divider()

if not run_btn:
    st.info("👈 Enter a city and click **Run Prediction** to start")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌲 Model", "Random Forest")
    col2.metric("📅 Forecast", "30 Days")
    col3.metric("📊 Features", "~40 lag/roll features")
    st.stop()

# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────
city = city_input.strip()
if not city:
    st.warning("Please enter a city name.")
    st.stop()

with st.spinner("🔍 Locating city..."):
    loc = geocode(city)

if not loc:
    st.error("City not found. Try a different name.")
    st.stop()

st.success(f"📍 **{loc['label']}** — lat: {loc['lat']:.2f}, lon: {loc['lon']:.2f}")

with st.spinner("📡 Fetching 2 years of historical weather data..."):
    df_raw = fetch_historical(loc["lat"], loc["lon"])

if df_raw is None or len(df_raw) < 100:
    st.error("Not enough historical data available for this location.")
    st.stop()

with st.spinner("⚙️ Engineering features..."):
    df_feat = build_features(df_raw)

with st.spinner("🌲 Training Random Forest model..."):
    model, scaler, feature_cols, target_cols, metrics, y_test_all, preds_all = train_model(df_feat)

with st.spinner("🔮 Predicting next 30 days..."):
    df_pred = predict_future(df_feat, model, scaler, feature_cols, days=30)

# ─────────────────────────────────────────────
#  MODEL METRICS
# ─────────────────────────────────────────────
st.markdown("## 📊 Model Performance")
c1, c2, c3, c4 = st.columns(4)

labels = {"temp_max": "🌡️ Max Temp", "temp_min": "🌡️ Min Temp", "rain": "🌧️ Rain", "wind": "💨 Wind"}
units  = {"temp_max": "°C", "temp_min": "°C", "rain": "mm", "wind": "km/h"}
cols   = [c1, c2, c3, c4]

for i, (key, col) in enumerate(zip(target_cols, cols)):
    with col:
        st.metric(
            label=labels[key],
            value=f"MAE: {metrics[key]['mae']} {units[key]}",
            delta=f"R² = {metrics[key]['r2']}"
        )

st.divider()

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🗓️ 30-Day Forecast", "🌡️ Temperature", "🌧️ Rain & Wind", "📈 Model Accuracy"])

# ── TAB 1: FORECAST TABLE ────────────────────
with tab1:
    st.markdown("### 30-Day Forecast")

    # Quick cards for first 7 days
    st.markdown("**Next 7 Days**")
    cols7 = st.columns(7)
    for i, (_, row) in enumerate(df_pred.head(7).iterrows()):
        with cols7[i]:
            temp_avg = (row["temp_max"] + row["temp_min"]) / 2
            code_guess = 0 if row["rain"] < 1 else (61 if row["rain"] < 5 else 80)
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(100,180,255,0.15);
                        border-radius:12px;padding:12px;text-align:center;">
                <div style="font-size:11px;color:#8ba0c0;">{row['date'].strftime('%a %d')}</div>
                <div style="font-size:28px;">{get_icon(code_guess)}</div>
                <div style="font-size:14px;font-weight:700;color:#7ecfff;">{row['temp_max']}°</div>
                <div style="font-size:11px;color:#8ba0c0;">{row['temp_min']}°</div>
                <div style="font-size:10px;color:#38bdf8;">💧{row['rain']}mm</div>
                <div style="font-size:10px;color:#93c5fd;">🌬{row['wind']}km/h</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("**Full 30-Day Table**")
    display_df = df_pred.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df.columns = ["Date", "Max Temp (°C)", "Min Temp (°C)", "Rain (mm)", "Wind (km/h)"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── TAB 2: TEMPERATURE ───────────────────────
with tab2:
    st.markdown("### Temperature Forecast — Next 30 Days")

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(df_pred["date"]) + list(df_pred["date"])[::-1],
        y=list(df_pred["temp_max"] + 2) + list(df_pred["temp_min"] - 2)[::-1],
        fill="toself", fillcolor="rgba(14,165,233,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Range", showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_pred["date"], y=df_pred["temp_max"],
        mode="lines+markers", name="Max Temp",
        line=dict(color="#f97316", width=2.5),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=df_pred["date"], y=df_pred["temp_min"],
        mode="lines+markers", name="Min Temp",
        line=dict(color="#38bdf8", width=2.5),
        marker=dict(size=5)
    ))

    # Fill between
    fig.add_trace(go.Scatter(
        x=list(df_pred["date"]) + list(df_pred["date"])[::-1],
        y=list(df_pred["temp_max"]) + list(df_pred["temp_min"])[::-1],
        fill="toself", fillcolor="rgba(249,115,22,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="Day Range", showlegend=True
    ))

    fig.update_layout(title="Max & Min Temperature (°C)", **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Historical context
    st.markdown("### Historical Temperature (Last 60 Days)")
    df_hist60 = df_raw.tail(60)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_hist60["date"], y=df_hist60["temp_max"],
                              mode="lines", name="Historical Max", line=dict(color="#f97316", width=1.5)))
    fig2.add_trace(go.Scatter(x=df_hist60["date"], y=df_hist60["temp_min"],
                              mode="lines", name="Historical Min", line=dict(color="#38bdf8", width=1.5)))
    fig2.update_layout(title="Historical Temperature Context", **CHART_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

# ── TAB 3: RAIN & WIND ───────────────────────
with tab3:
    col_r, col_w = st.columns(2)

    with col_r:
        st.markdown("### 🌧️ Precipitation Forecast")
        fig_r = go.Figure()
        colors = ["#0ea5e9" if r < 5 else "#1d4ed8" if r < 20 else "#1e3a8a" for r in df_pred["rain"]]
        fig_r.add_trace(go.Bar(
            x=df_pred["date"], y=df_pred["rain"],
            marker_color=colors, name="Rain (mm)"
        ))
        fig_r.update_layout(title="Daily Precipitation (mm)", **CHART_LAYOUT)
        st.plotly_chart(fig_r, use_container_width=True)

        total_rain = df_pred["rain"].sum()
        rain_days = (df_pred["rain"] > 1).sum()
        st.metric("Total Predicted Rain", f"{round(total_rain, 1)} mm")
        st.metric("Rainy Days", f"{rain_days} / 30")

    with col_w:
        st.markdown("### 💨 Wind Speed Forecast")
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(
            x=df_pred["date"], y=df_pred["wind"],
            mode="lines+markers", fill="tozeroy",
            fillcolor="rgba(99,102,241,0.15)",
            line=dict(color="#818cf8", width=2.5),
            marker=dict(size=4), name="Wind (km/h)"
        ))
        fig_w.update_layout(title="Max Wind Speed (km/h)", **CHART_LAYOUT)
        st.plotly_chart(fig_w, use_container_width=True)

        avg_wind = df_pred["wind"].mean()
        max_wind = df_pred["wind"].max()
        st.metric("Avg Wind Speed", f"{round(avg_wind, 1)} km/h")
        st.metric("Peak Wind Speed", f"{round(max_wind, 1)} km/h")

# ── TAB 4: MODEL ACCURACY ────────────────────
with tab4:
    st.markdown("### 📈 Predicted vs Actual (Test Set)")

    split = int(len(df_feat) * 0.85)
    df_test = df_feat.iloc[split:].reset_index(drop=True)
    n = len(df_test)

    fig_acc = make_subplots(rows=2, cols=2,
                            subplot_titles=["Max Temp", "Min Temp", "Precipitation", "Wind Speed"])

    pairs = [
        (df_test["temp_max"].values, preds_all[:, 0], "#f97316", 1, 1),
        (df_test["temp_min"].values, preds_all[:, 1], "#38bdf8", 1, 2),
        (df_test["rain"].values,     preds_all[:, 2], "#0ea5e9", 2, 1),
        (df_test["wind"].values,     preds_all[:, 3], "#818cf8", 2, 2),
    ]

    for actual, pred, color, row, col in pairs:
        fig_acc.add_trace(go.Scatter(y=actual[:120], mode="lines", name="Actual",
                                     line=dict(color="rgba(255,255,255,0.4)", width=1.5),
                                     showlegend=(row == 1 and col == 1)), row=row, col=col)
        fig_acc.add_trace(go.Scatter(y=pred[:120], mode="lines", name="Predicted",
                                     line=dict(color=color, width=2),
                                     showlegend=(row == 1 and col == 1)), row=row, col=col)

    fig_acc.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(255,255,255,0.02)",
                          font=dict(family="DM Sans", color="#c5deff"),
                          margin=dict(l=10, r=10, t=50, b=10))
    fig_acc.update_xaxes(gridcolor="rgba(100,180,255,0.08)")
    fig_acc.update_yaxes(gridcolor="rgba(100,180,255,0.08)")
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("### Metrics Summary")
    metrics_df = pd.DataFrame([
        {"Variable": labels[k], "MAE": f"{metrics[k]['mae']} {units[k]}", "R² Score": metrics[k]['r2']}
        for k in target_cols
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        **Architecture**: Multi-Output Random Forest  
        - 200 decision trees per target variable  
        - Max depth: 12  
        - Input: ~40 engineered features  

        **Features Used**:
        - Sine/cosine encoded day-of-year (captures seasonality)
        - Lag features: 1, 3, 7, 14 days back
        - Rolling mean: 7-day and 14-day windows
        - Month of year

        **Training**: First 85% of data | **Test**: Last 15%  
        **Prediction method**: Recursive (each day's prediction feeds into the next)
        """)

st.divider()
st.markdown(
    "<div style='text-align:center;color:#4a6080;font-size:12px;'>WeatherML · Random Forest · Open-Meteo API · Built with Streamlit</div>",
    unsafe_allow_html=True
)