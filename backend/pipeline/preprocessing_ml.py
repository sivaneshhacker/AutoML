import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import re
from dateutil import parser


# User uploads csv

def read_file(path):
    
    if (os.path.exists(path) is True):
        file_name_ext = os.path.splitext(path)[1].lower()
        
        if (file_name_ext == ".csv"):
            df = pd.read_csv(path)
        
        elif (file_name_ext in [".xlsx", "xls"]):
            df = pd.read_excel(path)

        elif (file_name_ext == '.parquet'):
            return pd.read_parquet(path)

        elif (file_name_ext in [".txt", ".tsv"]):
            df = pd.read_csv(path, sep="\t")
        
        else:
            raise ValueError(f"File format of {file_name_ext} not supported"
            "Only csv, xlsx, xls, txt and tsv are supported")
    
    else:
        raise FileNotFoundError(f"Path incorrect, no such file {path}")
    
    return df


def remove_space(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
    )
    
    object_columns = df.select_dtypes(include=["object","string"]).columns

    for col in object_columns:
        df[col] = (
            df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).replace(["nan","NaN", "Nan", "null","Null"], np.nan)
        )

    return df


def assign_column_name(df : pd.DataFrame) -> pd.DataFrame:
    i=1
    new_columns = []
    df = df.copy()

    for col in df.columns:
        if col == "":
            new_columns.append(f"no_name_{i}")
            i += 1
        else:
            new_columns.append(col)

    df.columns = new_columns

    return df


def remove_duplicate_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    seen = {}
    cols_to_drop = []

    for i in range(df.shape[1]):
        series = df.iloc[:, i]
        normalized = tuple(None if pd.isna(x) else x for x in series.values)

        if normalized in seen:
            cols_to_drop.append(df.columns[i])
        else:
            seen[normalized] = i

    df = df.drop(columns=cols_to_drop)
    return df


def _looks_like_datetime(series: pd.Series) -> bool:
    sample = series.dropna().astype(str)
    
    if len(sample) > 300:
        sample = sample.sample(300, random_state=42)
    else:
        sample = sample.sample(max(1, len(sample)//2), random_state=42)

    if sample.empty:
        return False
    
    date_pattern = re.compile(r"(\d{1,4}[-/:\\]\d{1,2}[-/:\\]\d{1,4})|(\d{1,2}[-/:\\]\d{1,2})|(\d{4}[-/:\\]\d{1,2})|(\d{1,2}[-/:\\]\d{4})|(\d{2}:\d{2}:\d{2})|(\d{2}:\d{2})")
    matches = sample.apply(lambda x: bool(date_pattern.search(x)))
    return matches.mean() > 0.3


def _robust_parse(x: str):
    try:
        return parser.parse(x, fuzzy=True, dayfirst=False)
    except Exception:
        try:
            return parser.parse(x, fuzzy=True, dayfirst=True)
        except Exception:
            return pd.NaT


def object_to_datetime(df: pd.DataFrame, min_parse_ratio: float = 0.6) -> pd.DataFrame:
    
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in obj_cols:
        s = df[col]

        # Normalize nulls
        s = s.replace(["", " ", "null", "NaN", "nan", "None", "?"], np.nan)

        if not _looks_like_datetime(s):
            continue

        sample = s.dropna().astype(str)
        if sample.empty:
            continue

        parsed = sample.apply(_robust_parse)
        success_ratio = parsed.notna().mean()

        if success_ratio >= min_parse_ratio:
            full_parsed = s.astype(str).apply(lambda x: _robust_parse(x) if pd.notna(x) else pd.NaT)
            df[col] = pd.to_datetime(full_parsed, errors="coerce")

    return df


def object_to_numeric(df: pd.DataFrame, min_conversion_ratio: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    
    for col in obj_cols:
        # First, strip whitespace from the series
        s = df[col].astype(str).str.strip()
        
        # Replace common null representations
        s = s.replace(
            ["", " ", "nan", "NaN", "None", "none", "null", "NULL", "?"], 
            np.nan
        )
        
        # Attempt numeric conversion
        numeric = pd.to_numeric(s, errors="coerce")
        
        # Calculate how many non-null values successfully converted
        original_non_null = s.notna().sum()
        converted_non_null = numeric.notna().sum()
        
        # Only convert if most values succeeded (or if there were no values to begin with)
        if original_non_null == 0 or (converted_non_null / original_non_null) >= min_conversion_ratio:
            if numeric.notna().any():  # At least one valid number
                # Check if all non-null values are integers
                non_null = numeric.dropna()
                df[col] = numeric.astype("float64")
    
    return df



def remove_constant_columns(
    df: pd.DataFrame,
    input_cols: list[str],
    near_constant_threshold: float = 0.95,
    drop_constant: bool = True,
    drop_near_constant: bool = True,
    return_report: bool = True
):
    
    df = df.copy()
    
    constant_cols = []
    near_constant_cols = []

    for col in input_cols:
        s = df[col]

        non_null = s.dropna()

        if non_null.empty:
            continue

        if non_null.nunique() <= 1:
            constant_cols.append(col)
            continue

        value_counts = non_null.value_counts(normalize=True, dropna=True)
        dominance_ratio = value_counts.iloc[0]

        if dominance_ratio >= near_constant_threshold:
            near_constant_cols.append(col)

    if drop_constant and constant_cols:
        df = df.drop(columns=constant_cols)

    if drop_near_constant and near_constant_cols:
        df = df.drop(columns=near_constant_cols)

    if return_report:
        report = {
            "constant_columns": constant_cols,
            "near_constant_columns": near_constant_cols,
            "threshold": near_constant_threshold
        }
        return df, report

    return df


def remove_high_cardinality(
    df: pd.DataFrame,
    cardinality_ratio_threshold: float = 0.7,
    absolute_unique_threshold: int | None = None,
    drop: bool = True,
    return_report: bool = True
):
    df = df.copy()
    input_cols = df.select_dtypes(include=["object", "string"]).columns
    high_cardinality_cols = []

    n_rows = len(df)

    for col in input_cols:
        s = df[col].dropna()

        if s.empty:
            continue

        nunique = s.nunique()
        ratio = nunique / len(s) if len(s) > 0 else 0

        if absolute_unique_threshold is not None:
            if nunique >= absolute_unique_threshold:
                high_cardinality_cols.append(col)
                continue

        if ratio >= cardinality_ratio_threshold:
            high_cardinality_cols.append(col)

    if drop and high_cardinality_cols:
        df = df.drop(columns=high_cardinality_cols)

    if return_report:
        report = {
            "high_cardinality_columns": high_cardinality_cols,
            "ratio_threshold": cardinality_ratio_threshold,
            "absolute_unique_threshold": absolute_unique_threshold
        }
        return df, report

    return df


def remove_duplicate_rows(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    duplicate_row_num = df.duplicated().sum()

    if duplicate_row_num > 0:
        df.drop_duplicates(inplace=True)

    return df



def separate_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def remove_full_row_nulls(X: pd.DataFrame, y: pd.Series):
    mask = X.isna().all(axis=1)
    X = X.loc[~mask].reset_index(drop=True)
    y = y.loc[~mask].reset_index(drop=True)
    return X, y


def remove_high_null_features(
    X: pd.DataFrame,
    threshold: float = 0.6
):
    null_ratio = X.isna().mean()
    drop_cols = null_ratio[null_ratio >= threshold].index.tolist()
    X = X.drop(columns=drop_cols)
    return X, drop_cols


def impute_numeric(X: pd.DataFrame, strategy: str = "median"):
    num_cols = X.select_dtypes(include=[np.number]).columns
    if strategy == "mean":
        X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    elif strategy == "median":
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    elif strategy == "zero":
        X[num_cols] = X[num_cols].fillna(0)
    return X


def impute_categorical(X: pd.DataFrame, strategy: str = "mode"):
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    if strategy == "mode":
        for c in cat_cols:
            if not X[c].mode().empty:
                X[c] = X[c].fillna(X[c].mode()[0])
            else:
                X[c] = X[c].fillna("missing")
    elif strategy == "missing":
        X[cat_cols] = X[cat_cols].fillna("missing")
    return X


def remove_rows_with_target_null(X: pd.DataFrame, y: pd.Series):
    mask = y.isna()
    X = X.loc[~mask].reset_index(drop=True)
    y = y.loc[~mask].reset_index(drop=True)
    return X, y


def detect_corrupted_rows(X: pd.DataFrame, threshold: float = 0.8):
    null_ratio = X.isna().mean(axis=1)
    return null_ratio >= threshold



def supervised_null_pipeline(
    df: pd.DataFrame,
    target_col: str,
    feature_null_threshold: float = 0.6,
    corrupted_row_threshold: float = 0.8,
    num_strategy: str = "median",
    cat_strategy: str = "mode",
    return_report: bool = True
):

    df = df.copy()

    X, y = separate_target(df, target_col)

    X, y = remove_rows_with_target_null(X, y)

    X, y = remove_full_row_nulls(X, y)

    corrupted_mask = detect_corrupted_rows(X, corrupted_row_threshold)
    X = X.loc[~corrupted_mask].reset_index(drop=True)
    y = y.loc[~corrupted_mask].reset_index(drop=True)

    X, dropped_high_null_cols = remove_high_null_features(X, feature_null_threshold)

    X = impute_numeric(X, num_strategy)
    X = impute_categorical(X, cat_strategy)

    out = X.copy()
    out[target_col] = y.values

    if return_report:
        report = {
            "dropped_high_null_features": dropped_high_null_cols,
            "removed_corrupted_rows": int(corrupted_mask.sum()),
            "feature_null_threshold": feature_null_threshold,
            "row_corruption_threshold": corrupted_row_threshold,
            "num_impute_strategy": num_strategy,
            "cat_impute_strategy": cat_strategy
        }
        return out, report

    return out



def dataframe_details_ml(df):
    df_shape = df.shape
    df_duplicate_sum = df.duplicated().sum()
    df_null_sum = df.isnull().sum().sum()
    
    df_columns = df.columns

    object_columns = []
    num_columns = []
    date_columns = []
    bool_columns = []
    category_columns = []

    object_columns = df.select_dtypes(include=["object", "string"]).columns
    num_columns = df.select_dtypes(include="number").columns
    date_columns = df.select_dtypes(include=["datetime", "timedelta"]).columns
    bool_columns = df.select_dtypes(include="bool").columns
    category_columns = df.select_dtypes(include="category").columns

    df_head = df.head()
    df_info = df.info()
    df_describe = df.describe()
    df_nunique = df.nunique()
    
    print(f"Dataframe Shape: {df_shape}")
    print(f"Duplicated Rows: {df_duplicate_sum}")
    print(f"Null Value Rows Presence: {df_null_sum}")
    print(f"Dataframe Columns: {df_columns}")

    print(object_columns)
    print(num_columns)

    print(date_columns)
    print(bool_columns)
    print(category_columns)

    print("\n\n",df_head)
    print(df_info)
    print("\n\n",df_describe)
    print("\n\n", df_nunique)

    sns.heatmap(df[num_columns].corr(), cmap="inferno", annot=True)
    plt.plot()

    return (df_head, df_info, df_describe, df_nunique, df_duplicate_sum, df_null_sum, object_columns, 
            num_columns, date_columns, bool_columns, category_columns)



def preprocessing_pipeline_ml(df : pd.DataFrame, target: str) -> pd.DataFrame:
    
    df = remove_space(df)
    df = assign_column_name(df)

    df = object_to_datetime(df)
    df = object_to_numeric(df)

    if df.shape[1] > 1:
        input_cols = df.drop(columns=[target], axis=1).columns
        df_input = df[input_cols]

        if df_input.shape[1] > 1:
            df_input = remove_duplicate_column(df_input)
        
        if df_input.shape[1] > 1:    
            df_input, constant_report = remove_constant_columns(df_input, df_input.columns)
            df_input, cardinality_report = remove_high_cardinality(df_input)

        df_input[target] = df[target].values
    else:
        df_input = df
        constant_report = None
        cardinality_report = None

    df_input = remove_duplicate_rows(df_input)

    if df_input.isnull().sum().sum() > 0:
        df_input, null_report = supervised_null_pipeline(df_input, target_col=target)
    else:
        null_report = None

    return df_input, null_report, constant_report, cardinality_report


def preprocessing_ml(df, target):


    # target = "origin"
    input_columns = []

    null_report = None
    constant_report = None
    cardinality_report = None


    df_new, null_report, constant_report, cardinality_report = preprocessing_pipeline_ml(df, target = target)


    (df_head, df_info, df_describe, df_nunique, df_duplicate_sum, df_null_sum, object_columns, 
    num_columns, date_columns, bool_columns, category_columns) = dataframe_details_ml(df_new)

    print(null_report)
    print(constant_report)
    print(cardinality_report)

    df_new.to_csv(".\output\df_new.csv", index=False)   # csv path (preprocessed)
