from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict diabetes risk using a logistic regression model",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def health_check():
    return{"status": "API is running"}
