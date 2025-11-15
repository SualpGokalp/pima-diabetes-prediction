import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def show_basic_info(df: pd.DataFrame) -> None:
    print("First 5 rows of the data:")
    print(df.head(5))
    print("\nDataFrame information:")
    print(df.info())
    print("\nStatistical summary of numeric columns:")
    print(df.describe())

def plot_outcome_distribution(df: pd.DataFrame, target_col: str = 'Outcome') -> None:
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=df)
    plt.title(f"Distribution of {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10,8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()
