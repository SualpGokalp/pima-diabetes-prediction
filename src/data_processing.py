import pandas as pd
from typing import List

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

def clean_data(df: pd.DataFrame, cols_with_zero: list) -> pd.DataFrame:
    for col in cols_with_zero:
        median_value = df[df[col] != 0][col].median()
        df[col] = df[col].replace(0,median_value)
    return df
