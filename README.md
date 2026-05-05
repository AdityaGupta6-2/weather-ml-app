# 🌩️ WeatherML v2 — Advanced ML Weather Prediction App

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=for-the-badge&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-purple?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> An end-to-end machine learning weather forecasting web app — trained on 3 years of real meteorological data using **LightGBM**, deployed with **Streamlit**.

---

## 🚀 Live Demo

> Run locally — see instructions below

---

## 📸 Screenshots

> Add screenshots of your app here after running it!
> (Take a screenshot → drag into this README on GitHub)

---

## ✨ Features

- 🌡️ **30–45 day forecast** for any city in the world
- ⚡ **LightGBM model** — separate model per weather variable
- 📉 **Probabilistic rain forecast** — shows % chance of rain per day
- 🚨 **Storm alert system** — auto-detects dangerous weather days
- 🕐 **Hourly 7-day forecast** from live Open-Meteo API
- 🏙️ **Compare two cities** side by side
- 📊 **Feature importance** visualization per variable
- 📥 **Export forecast to CSV**
- 📈 **Uncertainty bands** that widen with forecast horizon (data-driven)
- 🌡️ Pressure, cloud cover, solar radiation, dew point as inputs

---

## 🧠 Machine Learning Details

### Model Architecture
| Variable | Model | Objective |
|----------|-------|-----------|
| Max Temperature | LightGBM | Regression |
| Min Temperature | LightGBM | Regression |
| Precipitation | LightGBM | Tweedie (zero-inflated) |
| Wind Speed | LightGBM | Regression |

### Feature Engineering (~80 features)
- **Lag features**: 1, 2, 3, 7, 14, 21 days for all variables
- **Rolling statistics**: 7-day and 14-day mean + std
- **Cyclical encoding**: sin/cos of day-of-year and month
- **Atmospheric**: pressure drop, dew point spread, heat index
- **Derived**: daily temperature range, feels-like temperature

### Model Performance (Delhi, India — example)
| Variable | MAE | R² Score |
|----------|-----|----------|
| Max Temperature | ~1.7°C | ~0.90 |
| Min Temperature | ~1.1°C | ~0.95 |
| Precipitation | ~1.3 mm | ~0.15 |
| Wind Speed | ~2.3 km/h | ~0.26 |

> Note: Rain and wind are inherently chaotic — low R² is expected even in professional models.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.9+** | Core language |
| **LightGBM** | ML model (gradient boosted trees) |
| **Scikit-learn** | Preprocessing, metrics |
| **Streamlit** | Web app framework |
| **Plotly** | Interactive charts |
| **Pandas / NumPy** | Data processing |
| **Open-Meteo API** | Free weather data (no API key needed) |

---

## 📦 Installation & Run

### 1. Clone the repository
```bash
git clone https://github.com/AdityaGupta-75/weather-prediction-ml.git
cd weather-prediction-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python -m streamlit run weather_prediction_app_v2.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 📁 Project Structure

```
weather-prediction-ml/
│
├── weather_prediction_app_v2.py   # Main Streamlit app
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🔄 How It Works

```
1. Enter any city name
        ↓
2. Geocode city → get lat/lon (Open-Meteo Geocoding API)
        ↓
3. Fetch 3 years of daily historical weather data
        ↓
4. Engineer ~80 features (lags, rolling stats, cyclical encoding)
        ↓
5. Train 4 separate LightGBM models (one per target variable)
        ↓
6. Recursively predict next 30–45 days
        ↓
7. Display interactive forecast with alerts + uncertainty bands
```

---

## 📡 Data Source

All weather data is sourced from **[Open-Meteo](https://open-meteo.com/)** — a free, open-source weather API with no API key required.

Variables used:
- Temperature (max, min, apparent)
- Precipitation sum
- Wind speed (max)
- Atmospheric pressure (mean)
- Cloud cover (mean)
- Solar radiation (shortwave sum)
- Dew point (mean)
- Relative humidity (max)

---

## ⚠️ Limitations

- Rain and wind predictions have low R² — weather chaos is real!
- Accuracy drops significantly beyond 7–10 days
- Cannot predict sudden events (cyclones, flash floods)
- Not a replacement for IMD / AccuWeather for critical decisions

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — free to use, modify and distribute.

---

## 👨‍💻 Author

**Aditya Gupta**  
GitHub: [@AdityaGupta-75](https://github.com/AdityaGupta6-2)

---

⭐ **If you found this useful, please give it a star!** ⭐
