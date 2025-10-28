from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

# Load model
model = joblib.load("models/model.pkl")

app = FastAPI(title="Car Price Prediction API")

# Prometheus metrics setup
Instrumentator().instrument(app).expose(app)

# ✅ Define input schema
class CarFeatures(BaseModel):
    year: int
    present_price: float
    driven_kms: float
    owner: int
    fuel_type: str
    seller_type: str
    transmission: str

# ✅ Endpoint with schema
@app.post("/predict")
def predict(features: CarFeatures):
    data = pd.DataFrame([features.dict()])
    prediction = model.predict(data)
    return {"predicted_price": float(prediction[0])}
