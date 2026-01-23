from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "modeling"/ "logistic_pipeline.joblib"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

