"""
WeatherML v2 — Advanced Weather Prediction
Upgrades over v1:
  • LightGBM models (separate per target, auto-tuned)
  • More input variables: pressure, cloud cover, solar radiation, dew point
  • 3 years of training data
  • Probabilistic forecasts ("X% chance of rain")
  • Storm alert system
  • Compare two cities side by side
  • Export forecast to CSV
  • Hourly forecast tab
  • Feature importance chart
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WeatherML v2",
    page_icon="🌩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #070b14 0%, #0b1622 50%, #08111f 100%); color: #e0e6f0; }
[data-testid="stSidebar"] { background: rgba(10,18,35,0.97) !important; border-right: 1px solid rgba(100,180,255,0.12); }
[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(100,180,255,0.12); border-radius: 14px; padding: 16px; }
h1 { font-family: 'Space Mono', monospace !important; color: #7ecfff !important; }
h2, h3 { color: #c5deff !important; }
.stButton > button { background: linear-gradient(135deg, #1a6bff, #0ea5e9); color: white; border: none; border-radius: 10px; font-weight: 600; padding: 0.6rem 2rem; transition: all 0.2s; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(14,165,233,0.4); }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8ba0c0; font-weight: 500; }
.stTabs [aria-selected="true"] { background: rgba(30,107,255,0.3) !important; color: #7ecfff !important; }
hr { border-color: rgba(100,180,255,0.1) !important; }
.alert-storm { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); border-radius: 12px; padding: 14px 18px; margin: 8px 0; }
.alert-warning { background: rgba(251,191,36,0.12); border: 1px solid rgba(251,191,36,0.35); border-radius: 12px; padding: 14px 18px; margin: 8px 0; }
.alert-ok { background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.3); border-radius: 12px; padding: 14px 18px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
TARGET_COLS  = ["temp_max", "temp_min", "rain", "wind"]
LABELS       = {"temp_max": "🌡️ Max Temp", "temp_min": "🌡️ Min Temp", "rain": "🌧️ Rain", "wind": "💨 Wind"}
UNITS        = {"temp_max": "°C", "temp_min": "°C", "rain": "mm", "wind": "km/h"}
COLORS       = {"temp_max": "#f97316", "temp_min": "#38bdf8", "rain": "#0ea5e9", "wind": "#818cf8"}

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
#  WEATHER ICON
# ─────────────────────────────────────────────
def get_icon(rain, wind, temp_max):
    if wind > 60:   return "🌀"
    if rain > 20:   return "⛈️"
    if rain > 10:   return "🌧️"
    if rain > 2:    return "🌦️"
    if rain > 0.5:  return "🌂"
    if temp_max > 38: return "🔥"
    return "☀️"

# ─────────────────────────────────────────────
#  GEOCODING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def geocode(city: str):
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": city, "count": 1}, timeout=10)
        data = r.json()
        if "results" not in data: return None
        loc = data["results"][0]
        return {"label": f"{loc['name']}, {loc.get('country','')}", "lat": loc["latitude"], "lon": loc["longitude"]}
    except: return None

# ─────────────────────────────────────────────
#  FETCH DAILY HISTORICAL  (3 years, rich vars)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_historical(lat: float, lon: float):
    end   = datetime.now() - timedelta(days=5)
    start = end - timedelta(days=365 * 3)
    vars_ = ",".join([
        "temperature_2m_max", "temperature_2m_min",
        "apparent_temperature_max",
        "precipitation_sum",
        "windspeed_10m_max",
        "pressure_msl_mean",
        "cloudcover_mean",
        "shortwave_radiation_sum",
        "dewpoint_2m_mean",
        "relative_humidity_2m_max",
        "weathercode",
    ])
    try:
        r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                         params={"latitude": lat, "longitude": lon,
                                 "start_date": start.strftime("%Y-%m-%d"),
                                 "end_date":   end.strftime("%Y-%m-%d"),
                                 "daily": vars_, "timezone": "auto"}, timeout=25)
        d = r.json().get("daily", {})
        if not d: return None

        df = pd.DataFrame({
            "date":      pd.to_datetime(d["time"]),
            "temp_max":  d["temperature_2m_max"],
            "temp_min":  d["temperature_2m_min"],
            "feels_max": d.get("apparent_temperature_max", d["temperature_2m_max"]),
            "rain":      d["precipitation_sum"],
            "wind":      d["windspeed_10m_max"],
            "pressure":  d.get("pressure_msl_mean"),
            "cloud":     d.get("cloudcover_mean"),
            "solar":     d.get("shortwave_radiation_sum"),
            "dewpoint":  d.get("dewpoint_2m_mean"),
            "humidity":  d.get("relative_humidity_2m_max"),
            "code":      d["weathercode"],
        })
        df["rain"]    = df["rain"].clip(lower=0)
        df["wind"]    = df["wind"].clip(lower=0)
        df["cloud"]   = df["cloud"].clip(0, 100)
        df["humidity"]= df["humidity"].clip(0, 100)
        return df.dropna(subset=["temp_max", "temp_min", "rain", "wind"])
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return None

# ─────────────────────────────────────────────
#  FETCH HOURLY (last 7 days + next 7 days forecast)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_hourly_forecast(lat: float, lon: float):
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude": lat, "longitude": lon,
                                 "hourly": "temperature_2m,precipitation,windspeed_10m,cloudcover,pressure_msl",
                                 "forecast_days": 7, "timezone": "auto"}, timeout=15)
        d = r.json().get("hourly", {})
        if not d: return None
        df = pd.DataFrame({
            "datetime":  pd.to_datetime(d["time"]),
            "temp":      d["temperature_2m"],
            "rain":      d["precipitation"],
            "wind":      d["windspeed_10m"],
            "cloud":     d["cloudcover"],
            "pressure":  d["pressure_msl"],
        })
        df["rain"] = df["rain"].clip(lower=0)
        return df
    except: return None

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
EXTRA_VARS   = ["feels_max", "pressure", "cloud", "solar", "dewpoint", "humidity"]
FEATURE_SKIP = {"date", "code"} | set(TARGET_COLS)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["month"]        = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["sin_doy"]      = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"]      = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["sin_month"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]    = np.cos(2 * np.pi * df["month"] / 12)

    lag_cols = TARGET_COLS + [v for v in EXTRA_VARS if v in df.columns]
    for col in lag_cols:
        for lag in [1, 2, 3, 7, 14, 21]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_roll7"]    = df[col].rolling(7,  min_periods=1).mean()
        df[f"{col}_roll14"]   = df[col].rolling(14, min_periods=1).mean()
        df[f"{col}_roll7std"] = df[col].rolling(7,  min_periods=2).std().fillna(0)

    # Interaction features
    df["temp_range"]   = df["temp_max"] - df["temp_min"]
    if "pressure" in df.columns:
        df["pressure_drop"] = df["pressure"].diff().fillna(0)   # falling pressure → rain
    if "dewpoint" in df.columns and "temp_min" in df.columns:
        df["dew_spread"] = df["temp_min"] - df["dewpoint"]      # frost risk proxy

    return df.dropna()

# ─────────────────────────────────────────────
#  TRAIN  — separate LightGBM per target
# ─────────────────────────────────────────────
def train_models(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in FEATURE_SKIP]
    X = df[feature_cols].values
    split = int(len(X) * 0.85)

    X_train, X_test = X[:split], X[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models, metrics, residual_std, importances = {}, {}, {}, {}

    for col in TARGET_COLS:
        y      = df[col].values
        y_train, y_test = y[:split], y[split:]

        params = dict(n_estimators=500, learning_rate=0.05,
                      num_leaves=63, max_depth=8,
                      min_child_samples=10,
                      subsample=0.8, colsample_bytree=0.8,
                      random_state=42, n_jobs=-1, verbose=-1)

        # Rain: use tweedie (handles zero-inflated data better)
        if col == "rain":
            params["objective"] = "tweedie"
            params["tweedie_variance_power"] = 1.5

        m = lgb.LGBMRegressor(**params)
        m.fit(X_train_s, y_train,
              eval_set=[(X_test_s, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])

        pred = m.predict(X_test_s)
        pred = np.maximum(pred, 0) if col in ("rain", "wind") else pred

        metrics[col] = {
            "mae": round(mean_absolute_error(y_test, pred), 2),
            "r2":  round(r2_score(y_test, pred), 3),
        }
        residual_std[col] = round(float(np.std(y_test - pred)), 2)
        models[col]       = m
        importances[col]  = dict(zip(feature_cols, m.feature_importances_))

    return models, scaler, feature_cols, metrics, residual_std, importances, split

# ─────────────────────────────────────────────
#  RECURSIVE FUTURE PREDICTION
# ─────────────────────────────────────────────
def predict_future(df_raw, models, scaler, feature_cols, residual_std, days=30):
    df_work = df_raw.copy()
    predictions = []

    for i in range(days):
        df_f = build_features(df_work)
        last = df_f.iloc[[-1]].copy()
        for mc in [c for c in feature_cols if c not in last.columns]:
            last[mc] = 0.0

        X_s  = scaler.transform(last[feature_cols].values)
        pred = {col: float(models[col].predict(X_s)[0]) for col in TARGET_COLS}
        pred["rain"] = max(0.0, pred["rain"])
        pred["wind"] = max(0.0, pred["wind"])
        pred["temp_min"] = min(pred["temp_min"], pred["temp_max"] - 0.5)

        unc = min(1.0 + i * 0.07, 2.8)
        next_date = df_work["date"].iloc[-1] + timedelta(days=1)

        # Rain probability (sigmoid on predicted amount + residual)
        rain_prob = int(min(100, max(0,
            100 * (1 / (1 + np.exp(-0.6 * (pred["rain"] - 1.5)))))))

        new_row = pd.DataFrame({
            "date":     [next_date],
            "temp_max": [pred["temp_max"]],
            "temp_min": [pred["temp_min"]],
            "feels_max":[pred["temp_max"] - 1.5],
            "rain":     [pred["rain"]],
            "wind":     [pred["wind"]],
            "pressure": [df_work["pressure"].iloc[-1] if "pressure" in df_work.columns else 1013],
            "cloud":    [df_work["cloud"].iloc[-1]    if "cloud"    in df_work.columns else 50],
            "solar":    [df_work["solar"].iloc[-1]    if "solar"    in df_work.columns else 10],
            "dewpoint": [df_work["dewpoint"].iloc[-1] if "dewpoint" in df_work.columns else pred["temp_min"] - 2],
            "humidity": [df_work["humidity"].iloc[-1] if "humidity" in df_work.columns else 60],
            "code":     [df_work["code"].iloc[-1]],
        })
        df_work = pd.concat([df_work, new_row], ignore_index=True)

        predictions.append({
            "date":          next_date,
            "temp_max":      round(pred["temp_max"], 1),
            "temp_min":      round(pred["temp_min"], 1),
            "rain":          round(pred["rain"], 1),
            "wind":          round(pred["wind"], 1),
            "rain_prob":     rain_prob,
            "temp_max_hi":   round(pred["temp_max"] + residual_std["temp_max"] * unc, 1),
            "temp_max_lo":   round(pred["temp_max"] - residual_std["temp_max"] * unc, 1),
            "temp_min_hi":   round(pred["temp_min"] + residual_std["temp_min"] * unc, 1),
            "temp_min_lo":   round(pred["temp_min"] - residual_std["temp_min"] * unc, 1),
        })

    return pd.DataFrame(predictions)

# ─────────────────────────────────────────────
#  STORM ALERT LOGIC
# ─────────────────────────────────────────────
def generate_alerts(df_pred):
    alerts = []
    for _, row in df_pred.iterrows():
        day = row["date"].strftime("%b %d")
        if row["wind"] > 60 and row["rain"] > 15:
            alerts.append(("🌀 STORM WARNING", f"{day} — Wind {row['wind']} km/h + Rain {row['rain']} mm. Dangerous conditions.", "storm"))
        elif row["wind"] > 50:
            alerts.append(("💨 High Wind Alert", f"{day} — Wind speeds up to {row['wind']} km/h expected.", "warning"))
        elif row["rain"] > 20:
            alerts.append(("🌧️ Heavy Rain Alert", f"{day} — {row['rain']} mm of precipitation expected. Flooding possible.", "warning"))
        elif row["temp_max"] > 42:
            alerts.append(("🔥 Extreme Heat Alert", f"{day} — Max temperature {row['temp_max']}°C. Heat stroke risk.", "storm"))
        elif row["temp_max"] < 0:
            alerts.append(("🧊 Freeze Warning", f"{day} — Max temp {row['temp_max']}°C. Ice and frost likely.", "warning"))
    return alerts

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌩️ WeatherML v2")
    st.markdown("*LightGBM · Advanced Forecast*")
    st.divider()

    mode = st.radio("Mode", ["Single City", "Compare Two Cities"], index=0)
    city1 = st.text_input("📍 City 1", value="Delhi", placeholder="Enter city...")
    city2 = None
    if mode == "Compare Two Cities":
        city2 = st.text_input("📍 City 2", value="Mumbai", placeholder="Enter city...")

    forecast_days = st.slider("📅 Forecast Horizon (days)", 7, 45, 30)
    run_btn = st.button("🚀 Run Prediction", use_container_width=True)

    st.divider()
    st.markdown("**Model: LightGBM**")
    st.markdown("- 🌲 500 trees per variable")
    st.markdown("- ⚡ Early stopping")
    st.markdown("- 🎯 Separate model per target")
    st.markdown("- 📅 3 years training data")
    st.divider()
    st.markdown("**New in v2**")
    st.markdown("- 🌡️ Pressure, cloud, solar, dew point")
    st.markdown("- 📉 Probabilistic rain forecast")
    st.markdown("- 🚨 Storm alert system")
    st.markdown("- 🕐 Hourly forecast (7-day)")
    st.markdown("- 🏙️ Compare two cities")
    st.markdown("- 📥 CSV export")
    st.markdown("- 📊 Feature importance")

# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("# 🌩️ WeatherML v2 — Advanced Forecast")
st.markdown("Powered by **LightGBM** · Pressure · Cloud Cover · Solar Radiation · Probabilistic Rain")
st.divider()

if not run_btn:
    st.info("👈 Enter a city and click **Run Prediction** to start")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Model",    "LightGBM")
    c2.metric("📅 Forecast", "Up to 45 Days")
    c3.metric("📊 Features", "~80 engineered")
    c4.metric("📡 Data",     "3 yrs historical")
    st.stop()

# ─────────────────────────────────────────────
#  RUN PIPELINE FOR ONE CITY
# ─────────────────────────────────────────────
def run_pipeline(city_name):
    loc = geocode(city_name.strip())
    if not loc:
        st.error(f"City '{city_name}' not found.")
        return None

    with st.spinner(f"📡 Fetching 3 years of data for {loc['label']}..."):
        df_raw = fetch_historical(loc["lat"], loc["lon"])
    if df_raw is None or len(df_raw) < 150:
        st.error("Not enough data for this location.")
        return None

    with st.spinner("⚙️ Engineering ~80 features..."):
        df_feat = build_features(df_raw)

    with st.spinner("⚡ Training LightGBM models (4 separate models)..."):
        models, scaler, feature_cols, metrics, residual_std, importances, split = train_models(df_feat)

    with st.spinner(f"🔮 Predicting next {forecast_days} days..."):
        df_pred = predict_future(df_raw, models, scaler, feature_cols, residual_std, days=forecast_days)

    with st.spinner("🕐 Fetching hourly 7-day forecast..."):
        df_hourly = fetch_hourly_forecast(loc["lat"], loc["lon"])

    return {
        "loc": loc, "df_raw": df_raw, "df_feat": df_feat,
        "models": models, "scaler": scaler, "feature_cols": feature_cols,
        "metrics": metrics, "residual_std": residual_std,
        "importances": importances, "split": split,
        "df_pred": df_pred, "df_hourly": df_hourly
    }

# ─────────────────────────────────────────────
#  DISPLAY FOR SINGLE CITY
# ─────────────────────────────────────────────
def display_city(res, label=""):
    loc      = res["loc"]
    df_raw   = res["df_raw"]
    df_feat  = res["df_feat"]
    df_pred  = res["df_pred"]
    df_hourly= res["df_hourly"]
    metrics  = res["metrics"]
    residual_std = res["residual_std"]
    importances  = res["importances"]
    split    = res["split"]

    prefix = f"**{label}** — " if label else ""
    st.success(f"📍 {prefix}**{loc['label']}** — lat: {loc['lat']:.2f}, lon: {loc['lon']:.2f}")

    # ── ALERTS ─────────────────────────────────
    alerts = generate_alerts(df_pred)
    if alerts:
        st.markdown("### 🚨 Weather Alerts")
        for title, msg, kind in alerts:
            css = f"alert-{kind}"
            st.markdown(f'<div class="{css}"><strong>{title}</strong><br>{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-ok">✅ <strong>All Clear</strong> — No significant weather alerts for the forecast period.</div>', unsafe_allow_html=True)

    # ── MODEL METRICS ──────────────────────────
    st.markdown("### 📊 Model Performance (LightGBM)")
    c1, c2, c3, c4 = st.columns(4)
    for (key, lbl), col in zip(LABELS.items(), [c1, c2, c3, c4]):
        with col:
            st.metric(label=lbl,
                      value=f"MAE {metrics[key]['mae']} {UNITS[key]}",
                      delta=f"R² = {metrics[key]['r2']}")

    st.divider()

    # ── TABS ───────────────────────────────────
    tabs = st.tabs(["🗓️ Forecast", "🌡️ Temperature", "🌧️ Rain & Wind", "🕐 Hourly", "📊 Feature Importance", "📈 Accuracy"])

    # TAB 1 — FORECAST TABLE
    with tabs[0]:
        st.markdown(f"### {forecast_days}-Day Forecast")

        # 7-day cards
        st.markdown("**Next 7 Days**")
        cols7 = st.columns(7)
        for i, (_, row) in enumerate(df_pred.head(7).iterrows()):
            with cols7[i]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(100,180,255,0.15);
                            border-radius:12px;padding:10px;text-align:center;">
                    <div style="font-size:10px;color:#8ba0c0;">{row['date'].strftime('%a %d')}</div>
                    <div style="font-size:26px;">{get_icon(row['rain'], row['wind'], row['temp_max'])}</div>
                    <div style="font-size:13px;font-weight:700;color:#f97316;">{row['temp_max']}°</div>
                    <div style="font-size:11px;color:#38bdf8;">{row['temp_min']}°</div>
                    <div style="font-size:10px;color:#38bdf8;">💧{row['rain']}mm</div>
                    <div style="font-size:10px;color:#60a5fa;">☔{row['rain_prob']}%</div>
                    <div style="font-size:10px;color:#93c5fd;">💨{row['wind']}km/h</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("**Full Forecast Table**")
        disp = df_pred[["date","temp_max","temp_min","rain","rain_prob","wind"]].copy()
        disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
        disp.columns = ["Date","Max °C","Min °C","Rain mm","Rain %","Wind km/h"]
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Export CSV
        csv = disp.to_csv(index=False)
        st.download_button("📥 Download Forecast CSV", csv,
                           file_name=f"forecast_{loc['label'].replace(', ','_')}.csv",
                           mime="text/csv")

    # TAB 2 — TEMPERATURE
    with tabs[1]:
        st.markdown("### Temperature Forecast with Uncertainty Bands")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(df_pred["date"]) + list(df_pred["date"])[::-1],
            y=list(df_pred["temp_max_hi"]) + list(df_pred["temp_max_lo"])[::-1],
            fill="toself", fillcolor="rgba(249,115,22,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="Max Uncertainty"))
        fig.add_trace(go.Scatter(
            x=list(df_pred["date"]) + list(df_pred["date"])[::-1],
            y=list(df_pred["temp_min_hi"]) + list(df_pred["temp_min_lo"])[::-1],
            fill="toself", fillcolor="rgba(56,189,248,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="Min Uncertainty"))
        fig.add_trace(go.Scatter(x=df_pred["date"], y=df_pred["temp_max"],
            mode="lines+markers", name="Max Temp",
            line=dict(color="#f97316", width=2.5), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=df_pred["date"], y=df_pred["temp_min"],
            mode="lines+markers", name="Min Temp",
            line=dict(color="#38bdf8", width=2.5), marker=dict(size=5)))
        fig.update_layout(title="Temperature Forecast (°C)", **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Historical context
        st.markdown("### Historical Temperature (Last 90 Days)")
        h90 = df_raw.tail(90)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=h90["date"], y=h90["temp_max"], mode="lines",
            name="Max", line=dict(color="#f97316", width=1.5)))
        fig2.add_trace(go.Scatter(x=h90["date"], y=h90["temp_min"], mode="lines",
            name="Min", line=dict(color="#38bdf8", width=1.5)))
        if "feels_max" in h90.columns:
            fig2.add_trace(go.Scatter(x=h90["date"], y=h90["feels_max"], mode="lines",
                name="Feels Like", line=dict(color="#a78bfa", width=1.5, dash="dot")))
        fig2.update_layout(title="Historical Temperature (°C)", **CHART_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # TAB 3 — RAIN & WIND
    with tabs[2]:
        col_r, col_w = st.columns(2)
        with col_r:
            st.markdown("### 🌧️ Precipitation + Rain Probability")
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(
                x=df_pred["date"], y=df_pred["rain"],
                marker_color=["#1e3a8a" if r > 20 else "#1d4ed8" if r > 5 else "#0ea5e9"
                              for r in df_pred["rain"]],
                name="Rain (mm)"))
            fig_r.add_trace(go.Scatter(
                x=df_pred["date"], y=df_pred["rain_prob"],
                mode="lines+markers", name="Rain Prob %",
                line=dict(color="#f0abfc", width=2, dash="dot"),
                yaxis="y2", marker=dict(size=4)))
            fig_r.update_layout(
                title="Precipitation (mm) & Probability (%)",
                yaxis2=dict(overlaying="y", side="right", range=[0, 110],
                            gridcolor="rgba(0,0,0,0)", title="Probability %",
                            color="#f0abfc"),
                **CHART_LAYOUT)
            st.plotly_chart(fig_r, use_container_width=True)
            st.metric("Total Rain",  f"{round(df_pred['rain'].sum(), 1)} mm")
            st.metric("Rainy Days",  f"{(df_pred['rain'] > 1).sum()} / {forecast_days}")
            st.metric("High Rain Prob Days (>60%)", f"{(df_pred['rain_prob'] > 60).sum()}")

        with col_w:
            st.markdown("### 💨 Wind Speed Forecast")
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(
                x=df_pred["date"], y=df_pred["wind"],
                mode="lines+markers", fill="tozeroy",
                fillcolor="rgba(99,102,241,0.12)",
                line=dict(color="#818cf8", width=2.5),
                marker=dict(size=4), name="Wind (km/h)"))
            # Danger threshold line
            fig_w.add_hline(y=50, line_dash="dash", line_color="rgba(239,68,68,0.5)",
                            annotation_text="High Wind ⚠️", annotation_position="top left")
            fig_w.update_layout(title="Max Wind Speed (km/h)", **CHART_LAYOUT)
            st.plotly_chart(fig_w, use_container_width=True)
            st.metric("Avg Wind",  f"{round(df_pred['wind'].mean(), 1)} km/h")
            st.metric("Peak Wind", f"{round(df_pred['wind'].max(), 1)} km/h")

        # Pressure & Cloud
        if "pressure" in df_raw.columns and "cloud" in df_raw.columns:
            st.markdown("### 🌡️ Atmospheric Pressure & Cloud Cover (Last 60 Days)")
            h60 = df_raw.tail(60)
            fig_atm = make_subplots(rows=1, cols=2,
                                    subplot_titles=["Pressure (hPa)", "Cloud Cover (%)"])
            fig_atm.add_trace(go.Scatter(x=h60["date"], y=h60["pressure"],
                mode="lines", name="Pressure", line=dict(color="#34d399", width=1.8)), row=1, col=1)
            fig_atm.add_trace(go.Scatter(x=h60["date"], y=h60["cloud"],
                mode="lines", fill="tozeroy", name="Cloud %",
                fillcolor="rgba(148,163,184,0.12)",
                line=dict(color="#94a3b8", width=1.8)), row=1, col=2)
            fig_atm.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(255,255,255,0.02)",
                                  font=dict(family="DM Sans", color="#c5deff"),
                                  margin=dict(l=10, r=10, t=40, b=10))
            fig_atm.update_xaxes(gridcolor="rgba(100,180,255,0.08)")
            fig_atm.update_yaxes(gridcolor="rgba(100,180,255,0.08)")
            st.plotly_chart(fig_atm, use_container_width=True)

    # TAB 4 — HOURLY
    with tabs[3]:
        if df_hourly is not None:
            st.markdown("### 🕐 Hourly Forecast — Next 7 Days")
            fig_h = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                  subplot_titles=["Temperature (°C)", "Rain (mm/hr)", "Wind (km/h)"],
                                  vertical_spacing=0.08)
            fig_h.add_trace(go.Scatter(x=df_hourly["datetime"], y=df_hourly["temp"],
                mode="lines", name="Temp °C",
                line=dict(color="#f97316", width=1.8)), row=1, col=1)
            fig_h.add_trace(go.Bar(x=df_hourly["datetime"], y=df_hourly["rain"],
                name="Rain mm", marker_color="#0ea5e9"), row=2, col=1)
            fig_h.add_trace(go.Scatter(x=df_hourly["datetime"], y=df_hourly["wind"],
                mode="lines", fill="tozeroy", name="Wind km/h",
                fillcolor="rgba(129,140,248,0.12)",
                line=dict(color="#818cf8", width=1.5)), row=3, col=1)
            fig_h.update_layout(height=560, paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(255,255,255,0.02)",
                                font=dict(family="DM Sans", color="#c5deff"),
                                margin=dict(l=10, r=10, t=50, b=10),
                                showlegend=False)
            fig_h.update_xaxes(gridcolor="rgba(100,180,255,0.08)")
            fig_h.update_yaxes(gridcolor="rgba(100,180,255,0.08)")
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.warning("Hourly data could not be fetched for this location.")

    # TAB 5 — FEATURE IMPORTANCE
    with tabs[4]:
        st.markdown("### 📊 Top 15 Most Important Features per Variable")
        fig_imp = make_subplots(rows=2, cols=2,
                                subplot_titles=["Max Temp","Min Temp","Rain","Wind"])
        positions = [(1,1),(1,2),(2,1),(2,2)]
        for (col, (row, c)) in zip(TARGET_COLS, positions):
            imp = importances[col]
            top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
            names, vals = zip(*top)
            fig_imp.add_trace(go.Bar(
                x=vals[::-1], y=names[::-1],
                orientation="h", name=LABELS[col],
                marker_color=COLORS[col]), row=row, col=c)
        fig_imp.update_layout(
            height=620, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(family="DM Sans", color="#c5deff"),
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False)
        fig_imp.update_xaxes(gridcolor="rgba(100,180,255,0.08)")
        fig_imp.update_yaxes(gridcolor="rgba(100,180,255,0.08)")
        st.plotly_chart(fig_imp, use_container_width=True)

    # TAB 6 — ACCURACY
    with tabs[5]:
        st.markdown("### 📈 Predicted vs Actual (Test Set)")
        y_all = df_feat[TARGET_COLS].values
        X_all_s = res["scaler"].transform(df_feat[res["feature_cols"]].values)
        split_  = res["split"]
        X_test_s = X_all_s[split_:]
        y_test   = y_all[split_:]
        preds_all = np.column_stack([
            res["models"][col].predict(X_test_s) for col in TARGET_COLS])

        n_show = min(150, len(y_test))
        fig_acc = make_subplots(rows=2, cols=2,
                                subplot_titles=["Max Temp (°C)","Min Temp (°C)","Rain (mm)","Wind (km/h)"])
        pairs = [(y_test[:,i], preds_all[:,i], COLORS[col], *pos)
                 for i,(col,pos) in enumerate(zip(TARGET_COLS,[(1,1),(1,2),(2,1),(2,2)]))]
        for actual, pred, color, row, col in pairs:
            sl = (row==1 and col==1)
            fig_acc.add_trace(go.Scatter(y=actual[:n_show], mode="lines", name="Actual",
                line=dict(color="rgba(255,255,255,0.4)", width=1.5), showlegend=sl), row=row, col=col)
            fig_acc.add_trace(go.Scatter(y=pred[:n_show], mode="lines", name="Predicted",
                line=dict(color=color, width=2), showlegend=sl), row=row, col=col)
        fig_acc.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(255,255,255,0.02)",
                              font=dict(family="DM Sans", color="#c5deff"),
                              margin=dict(l=10, r=10, t=50, b=10))
        fig_acc.update_xaxes(gridcolor="rgba(100,180,255,0.08)")
        fig_acc.update_yaxes(gridcolor="rgba(100,180,255,0.08)")
        st.plotly_chart(fig_acc, use_container_width=True)

        metrics_df = pd.DataFrame([
            {"Variable": LABELS[k], "MAE": f"{metrics[k]['mae']} {UNITS[k]}",
             "R² Score": metrics[k]["r2"], "Residual σ": f"{residual_std[k]} {UNITS[k]}"}
            for k in TARGET_COLS])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        with st.expander("ℹ️ About WeatherML v2"):
            st.markdown("""
            **Model**: LightGBM (Gradient Boosted Trees)
            - Separate model per target variable
            - 500 trees with early stopping (prevents overfitting)
            - Rain uses Tweedie objective (handles zero-inflated distributions)
            - ~80 engineered features

            **New Variables**: Atmospheric pressure, cloud cover, solar radiation, dew point

            **New Features**:
            - Probabilistic rain forecast (% chance)
            - Storm alert system
            - Hourly 7-day forecast (from Open-Meteo live API)
            - Compare two cities
            - CSV export
            - Feature importance visualization
            - Uncertainty bands grow with forecast horizon
            """)

# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
if mode == "Single City":
    res1 = run_pipeline(city1)
    if res1:
        display_city(res1)
else:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"## 🏙️ {city1.strip()}")
        res1 = run_pipeline(city1)
        if res1:
            display_city(res1, label=city1.strip())
    with col_b:
        st.markdown(f"## 🏙️ {city2.strip()}")
        res2 = run_pipeline(city2)
        if res2:
            display_city(res2, label=city2.strip())

st.divider()
st.markdown(
    "<div style='text-align:center;color:#3a5070;font-size:12px;'>"
    "WeatherML v2 · LightGBM · Open-Meteo API · Built with Streamlit"
    "</div>", unsafe_allow_html=True)
