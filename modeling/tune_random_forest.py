import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import load_data, clean_data

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

pipeline = Pipeline(
    steps=[
        ("İmputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("scaler", StandardScaler()),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
        ),
    ]
)


param_distributions = {
    "model__n_estimators": [200, 300, 400, 500],
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_leaf": [1, 3, 5, 10],
    "model__max_features": ["sqrt", "log2", 0.5],
    "model__class_weight": ["balanced"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X, y)

print("\nBest ROC-AUC:", search.best_score_)
print("Best Parameters:")
for k, v in search.best_params_.items():
    print(f"{k}: {v}")