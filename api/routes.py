import numpy as np
from fastapi import APIRouter

from api.schemas import PatientData, PredictionResponse
from api.model_loader import load_model

router = APIRouter()
model = load_model()

THRESHOLD = 0.4
MODEL_NAME = "Random Forest"

def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"                    

@router.post("/predict", response_model = PredictionResponse)
def predict_diabetes(data: PatientData):
    features = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    probability = model.predict_proba(features)[0][1]
    prediction = "Diabetes" if probability >= THRESHOLD else "No Diabetes"
    risk_level = get_risk_level(probability)

    return {
        "prediction": prediction,
        "probability": round(probability, 3),
        "risk_level": risk_level,
        "model": MODEL_NAME,
         "threshold_used": THRESHOLD
    }
