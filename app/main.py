from fastapi import FastAPI
import joblib
import pandas as pd

model=joblib.load("models/model.pkl")

app=FastAPI()

@app.post("/predict")
def predict(features:dict):
    data=pd.DataFrame([features])
    prediction=model.predict(data)
    return{'car price ': prediction[0]}