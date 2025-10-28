from fastapi import FastAPI
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator


model = joblib.load("models/model.pkl")

app = FastAPI()

# Set up Prometheus instrumentation correctly.
# Use instrument(app) then expose(app). No need for a custom lifespan handler here.
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.post("/predict")
def predict(features:dict):
    data=pd.DataFrame([features])
    prediction=model.predict(data)
    return{'car price ': prediction[0]}