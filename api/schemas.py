from pydantic import BaseModel, Field

class PatientData(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=50, le=300)
    BloodPressure: float = Field(..., ge=30, le=200)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=900)
    BMI: float = Field(..., ge=10, le=80)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=5)
    Age: int = Field(..., ge=1, le=120)

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    message: str
    model: str
    model_version: str
    threshold_used: float