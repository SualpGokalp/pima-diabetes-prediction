import  os
import pandas as pd
from src.data_processing import load_data, clean_data

def test_cleaning_and_save():
    df = load_data('data/diabetes.csv')

    cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

    df_cleaned = clean_data(df.copy(), cols_with_zero)

    print("Before cleaning - first 5 rows: ")
    print(df.head(5))

    print("\nAfter cleaning - first 5 rows: ")
    print(df_cleaned.head(5))

    for col in cols_with_zero:
        zero_before = (df[col == 0]).sum()
        zero_after = (df_cleaned[col] == 0).sum()
        print(f"{col}: number of zero values -> before: {zero_before}, after: {zero_after}")

    output_folder = 'tests/output'
    os.makedirs(output_folder, exit_ok=True)
    output_path = os.path.join(output_folder, 'cleaned_head.csv')
    df_cleaned.head(5).to_csv(output_path, index=False)
    print(f"Cleaned first 5 rows saved -> {output_path}")
                
