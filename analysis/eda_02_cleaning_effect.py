import matplotlib.pyplot as plt
from src.data_processing import load_data, clean_data, compare_before_after

cols_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df_before = load_data("data/diabetes.csv")
df_after = clean_data(df_before, cols_with_zero)

changes = compare_before_after(df_before, df_after, cols_with_zero)

for col, diff in changes.items():
    print(f"\n🔹 {col} - changed rows: {len(diff)}")
    print(diff.head())

col = "Glucose"
bins = range(40, 210, 10)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(df_before[col], bins=bins)
plt.title("Before Cleaning")

plt.subplot(1,2,2)
plt.hist(df_after[col], bins=bins)
plt.title("After Cleaning")

plt.show()
