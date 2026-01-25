import numpy as np
from fastapi import APIRouter

from api.config import MODEL_NAME, MODEL_VERSION, THRESHOLD
from api.schemas import PatientData, PredictionResponse
from api.model_loader import load_model

router = APIRouter()
model = load_model()

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

    message = (
        "High risk detected. Medical follow-up is recommended."
        if prediction == "Diabetes"
        else "Low risk detected. Continue regular monitoring"
    )

    return {
        "prediction": prediction,
        "probability": round(probability, 3),
        "risk_level": risk_level,
        "message": message,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "threshold_used": THRESHOLD
    }

@router.get("/feature-importance")
def feature_importance():
    rf_model = model.named_steps["model"]
    features = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    importances = rf_model.feature_importances_

    return dict(sorted(
        zip(features, importances),
        key = lambda x: x[1],
        reverse= True
    ))


