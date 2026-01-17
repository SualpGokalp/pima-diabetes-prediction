import matplotlib.pyplot as plt
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

print("Dataset shape:", df.shape)

print("\nOutcome distribution (counts):")
print(df["Outcome"].value_counts())

print("\nOutcome distribution (ratios):")
print(df["Outcome"].value_counts(normalize=True))

plt.figure(figsize=(5,4))
plt.hist(df["Outcome"], bins=2)
plt.xticks([0,1], ["No Diabetes", "Diabetes"])
plt.title("Outcome Distribution")
plt.ylabel("Count")
plt.show()