import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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
y = df ["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = Pipeline(steps = [
    ("imputer", KNNImputer(
        n_neighbors = 5,
        weights="distance"
    )),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators = 300,
        max_depth = None,
        min_samples_leaf = 5,
        class_weight ="balanced",
        random_state = 42
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, "modeling/random_forest_pipeline.joblib")

print("\nModel saved as random_forest_pipeline.joblib")