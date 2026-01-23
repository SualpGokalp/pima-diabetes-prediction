from pydantic import BaseModel

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str