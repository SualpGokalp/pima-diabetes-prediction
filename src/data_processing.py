import pandas as pd

def load_data(filepath: str) -> pd.DataFrame
    df = pd.read_csv(filepath)
    return df

def clean_data(df: pd.DataFrame, cols_with_zero: list) -> pd.DataFrame:

