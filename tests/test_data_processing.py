import pandas as pd
import pytest

from src.data_processing import load_data, clean_data

def test_load_data_returns_dataframe():
    df = load_data("data/diabetes.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] > 0

def test_clean_data_replaces_zero_with_median():
    df = pd.DataFrame({
        "Glucose": [0, 100, 120, 0],
        "BMI": [0, 30.0, 35.0, 0],
        "Outcome": [0, 1, 1, 0]
    })

    cleaned = clean_data(df, ["Glucose", "BMI"])

    assert cleaned.loc[0, "Glucose"] == 110
    assert cleaned.loc[3, "Glucose"] == 110

    assert cleaned.loc[0, "BMI"] == 32.5
    assert cleaned.loc[3, "BMI"] == 32.5

def test_clean_data_does_not_modify_original_df():
    df = pd.DataFrame({"Glucose": [0, 100], "Outcome": [0,1]})
    df_original = df.copy()

    _ = clean_data(df,["Glucose"])

    pd.testing.assert_frame_equal(df, df_original)

def test_clean_data_raises_for_missing_column():
    df = pd.DataFrame({"Glucose": [0, 100], "Outcome": [0,1]})

    with pytest.raises(KeyError):
        clean_data(df, ["BMI"])

def test_clean_data_raises_for_empty_cols_list():
    df = pd.DataFrame({"Glucose":[0, 100], "Outcome": [0, 1]})

    with pytest.raises(ValueError):
        clean_data(df, [])

def test_clean_data_raises_if_all_values_zero_in_column():
    df = pd.DataFrame({"Glucose": [0, 0, 0], "Outcome": [0, 1, 0]})

    with pytest.raises(ValueError):
        clean_data(df,["Glucose"])

def test_clean_data_multiple_columns():
    df = pd.DataFrame({
        "Glucose": [0, 100, 0],
        "BloodPressure": [70, 0, 80],
        "BMI": [0, 30.0, 35.0],
        "Outcome": [1, 0, 1]
    })
    cleaned = clean_data(df, ["Glucose", "BloodPressure", "BMI"])

    assert cleaned.loc[0, "Glucose"] == 100
    assert cleaned.loc[1, "BloodPressure"] == 75
    assert cleaned.loc[0, "BMI"] == 32.5

def test_clean_data_eral_dataset():
    cols_with_zero = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    df = load_data("data/diabetes.csv")
    cleaned = clean_data(df, cols_with_zero)

    for col in cols_with_zero:
        assert (cleaned[col] == 0).sum() == 0