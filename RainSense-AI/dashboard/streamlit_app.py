from __future__ import annotations

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics.weather_analytics import detect_weather_anomalies, monthly_forecast, rainfall_trends, seasonal_patterns
from src.config import FEATURE_COLUMNS, LATEST_MODEL_PATH, METRICS_PATH, MODEL_INFO_PATH
from src.explainability.insights import feature_importance, shap_summary
from src.mlops.tracking import read_json
from src.prediction.predictor import predict_rainfall
from src.preprocessing.data_loader import load_weather_data
from src.preprocessing.preprocessor import prepare_dataset
from src.training.train_models import train_all_models


st.set_page_config(page_title="RainSense AI", page_icon="🌧️", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
          linear-gradient(135deg, rgba(7, 18, 38, .94), rgba(8, 44, 55, .91)),
          url("https://images.unsplash.com/photo-1501691223387-dd0500403074?auto=format&fit=crop&w=1800&q=80");
        background-size: cover;
        color: #f7fbff;
    }
    [data-testid="stSidebar"] { background: rgba(4, 16, 27, .94); }
    .metric-card {
        padding: 16px;
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 8px;
        background: rgba(255,255,255,.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def dataset() -> pd.DataFrame:
    return prepare_dataset(load_weather_data())


def ensure_model():
    if not LATEST_MODEL_PATH.exists():
        train_all_models()
    return joblib.load(LATEST_MODEL_PATH)


df = dataset()
page = st.sidebar.radio("Navigate", ["Home", "Prediction", "Analytics", "Model Insights"])
metrics = read_json(METRICS_PATH, {})
model_info = read_json(MODEL_INFO_PATH, {})

if page == "Home":
    st.title("RainSense AI")
    st.caption("Next-generation rainfall prediction and weather intelligence platform")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical Days", f"{len(df):,}")
    c2.metric("Rainy Days", f"{int(df['rain'].sum()):,}")
    c3.metric("Avg Rainfall", f"{df['rainfall_mm'].mean():.1f} mm")
    c4.metric("Champion Model", model_info.get("model_name", "Train pending"))
    if metrics:
        leaderboard = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
        st.plotly_chart(px.bar(leaderboard, x="model", y="f1", color="roc_auc", title="Model Performance Overview"), width="stretch")
    st.plotly_chart(px.area(rainfall_trends(df), x="date", y="rainfall_mm", title="Monthly Rainfall Trend"), width="stretch")

elif page == "Prediction":
    st.title("Rainfall Prediction")
    left, right = st.columns([1, 1.2])
    with left:
        temperature = st.slider("Temperature", -10.0, 50.0, 28.0)
        humidity = st.slider("Humidity", 0.0, 100.0, 78.0)
        pressure = st.slider("Pressure", 900.0, 1060.0, 1006.0)
        wind_speed = st.slider("Wind Speed", 0.0, 60.0, 16.0)
        cloud_cover = st.slider("Cloud Cover", 0.0, 100.0, 72.0)
        sunshine_hours = st.slider("Sunshine Hours", 0.0, 14.0, 3.5)
        month = st.selectbox("Month", list(range(1, 13)), index=6)
    payload = {
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "cloud_cover": cloud_cover,
        "sunshine_hours": sunshine_hours,
        "month": month,
    }
    result = predict_rainfall(payload)
    with right:
        st.metric("Rain Probability", f"{result['rain_probability']}%")
        st.metric("Rainfall Prediction", result["rainfall_prediction"])
        st.metric("Estimated Rainfall", f"{result['estimated_rainfall_mm']} mm")
        st.metric("Confidence Score", f"{result['confidence_score']}%")

elif page == "Analytics":
    st.title("Advanced Weather Analytics")
    trend = rainfall_trends(df)
    forecast = monthly_forecast(df)
    st.plotly_chart(px.line(trend, x="date", y="rainfall_mm", title="Historical Monthly Rainfall"), width="stretch")
    st.plotly_chart(px.bar(forecast, x="date", y="forecast_rainfall_mm", title="Monthly Rainfall Forecast"), width="stretch")
    st.plotly_chart(px.imshow(df.pivot_table(index="season", columns="month", values="rainfall_mm", aggfunc="mean").fillna(0), title="Rainfall Heatmap"), width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(seasonal_patterns(df), x="season", y="avg_rainfall_mm", title="Seasonal Rainfall"), width="stretch")
    anomalies = detect_weather_anomalies(df)
    c2.plotly_chart(px.scatter(anomalies, x="date", y="rainfall_mm", color="season", title="Weather Anomalies"), width="stretch")

elif page == "Model Insights":
    st.title("Model Insights")
    model = ensure_model()
    x = df[FEATURE_COLUMNS]
    y = df["rain"]
    importance = feature_importance(model, x, y)
    st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h", title="Feature Importance"), width="stretch")
    if metrics:
        leaderboard = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"}).sort_values("f1", ascending=False)
        st.dataframe(leaderboard, width="stretch")
    shap_values, shap_sample = shap_summary(model, x)
    if shap_values is None:
        st.info("SHAP package or model-specific explainer is unavailable; showing robust feature importance instead.")
    else:
        st.write("SHAP analysis is available in the model object; export plots from notebooks or extend this page for static SHAP figures.")
