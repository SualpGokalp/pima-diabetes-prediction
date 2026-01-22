import numpy as np
from fastapi import APIRouter

from api.schemas import PatientData, PredictionResponse
from api.model_loader import load_model

router = APIRouter()
model = load_model()


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
    prediction = "Diabetes" if probability >= 0.5 else "No Diabetes"

    return {
     "prediction": prediction,
      "probability": round(probability, 3)
    }
