import joblib

MODEL_PATH = "modeling/logistic_model.joblib"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

