from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# loan model 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "model.pkl")

model = joblib.load(MODEL_PATH)

# PredictRequest is YOUR class that uses BaseModel
class PredictRequest(BaseModel):
    lag_1: float
    lag_7: float
    lag_14: float
    rolling_mean_7: float
    rolling_mean_14: float
    id_encoded: int

# create APP

app =FastAPI(title= "Demand Forecasting API")

@app.get("/health")
def health():
    return {"status": "ok"}

# Metrics endpoint
@app.get("/metrics")
def metrics():
    return {
        "MAE": 0.4819,
        "RMSE": 1.1259
    }

# Predict endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    features = np.array([[
        request.lag_1,
        request.lag_7,
        request.lag_14,
        request.rolling_mean_7,
        request.rolling_mean_14,
        request.id_encoded
    ]])
    prediction = model.predict(features)
    return {"predicted_sales": round(float(prediction[0]), 2)}