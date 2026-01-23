from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(
    alcohol: float,
    sulphates: float,
    volatile_acidity: float,
    density: float,
    chlorides: float,
    total_sulfur_dioxide: float,
    fixed_acidity: float,
    pH: float
):
    features = np.array([[
        alcohol,
        sulphates,
        volatile_acidity,
        density,
        chlorides,
        total_sulfur_dioxide,
        fixed_acidity,
        pH
    ]])

    prediction = model.predict(features)
    return {"predicted_quality": float(prediction[0])}
