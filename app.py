from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model from CI-generated artifacts
model = joblib.load("outputs/model.pkl")

# IMPORTANT: feature order MUST match training
FEATURE_ORDER = [
    "alcohol",
    "sulphates",
    "volatile_acidity",
    "density",
    "chlorides",
    "total_sulfur_dioxide",
    "fixed_acidity",
    "pH"
]

class WineInput(BaseModel):
    alcohol: float
    sulphates: float
    volatile_acidity: float
    density: float
    chlorides: float
    total_sulfur_dioxide: float
    fixed_acidity: float
    pH: float

@app.post("/predict")
def predict(data: WineInput):
    features = np.array([[getattr(data, f) for f in FEATURE_ORDER]])
    prediction = model.predict(features)

    return {
        "name": "Challa Sreeja",
        "roll_no": "2022BCS0121",
        "wine_quality": int(round(prediction[0]))
    }
