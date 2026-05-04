from fastapi import FastAPI
import pandas as pd
import joblib

# create app
app = FastAPI()

# load model
model = joblib.load("models/best_model.joblib")

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"prediction": float(pred[0])}