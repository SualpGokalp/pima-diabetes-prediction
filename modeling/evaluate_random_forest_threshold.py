import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data_processing import load_data, clean_data

pipeline = joblib.load("modeling/random_forest_pipeline.joblib")

cols_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df = load_data("data/diabetes.csv")
df = clean_data(df, cols_with_zero)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

y_proba = pipeline.predict_proba(X_test)[:, 1]

THRESHOLD = 0.40

y_pred_threshold = (y_proba >= THRESHOLD).astype(int)

print(f"\nRandom Forest with Threshold = {THRESHOLD}\n")
print(classification_report(y_test, y_pred_threshold))