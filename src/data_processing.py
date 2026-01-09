import pandas as pd
from typing import List, Dict

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

def clean_data(df: pd.DataFrame, cols_with_zero: list) -> pd.DataFrame:
    if not cols_with_zero:
        raise ValueError("cols_with_zero cannot be empty.")
    
    df_clean = df.copy()

    for col in cols_with_zero:
        if col not in df_clean.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        
        non_zero = df_clean.loc[df_clean[col] !=0, col]

        if non_zero.empty:
              raise ValueError(
                 f"Column '{col}' has no non-zero values. Median cannot be computed"
                  )
        median_value = non_zero.median()
        
        df_clean.loc[df_clean[col] == 0 ,col] = median_value

    return df_clean

def compare_before_after(
        before: pd.DataFrame,
        after: pd.DataFrame,
        cols: List[str]
) -> Dict[str, pd.DataFrame]:
    changes = {}

    for col in cols:
        mask = before[col] != after[col]
        changes[col] = pd.DataFrame({
            "before": before.loc[mask,col],
            "after": after.loc[mask, col]
        })
    return changes
    
    