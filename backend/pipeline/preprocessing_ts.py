import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import re
from datetime import timedelta
from dateutil import parser
import warnings
warnings.filterwarnings('ignore')


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


# Userselects target variable and input columns
# target = "keep_me_updated"
date_column = "date"


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


def object_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in obj_cols:
        s = df[col].astype(str)

        s = s.replace(
            ["", " ", "  ", "nan", "NaN", "None", "none", "null", "NULL", "?"], 
            np.nan
        )

        numeric = pd.to_numeric(s, errors="coerce")

        non_null_ratio = numeric.notna().mean()

        if non_null_ratio > 0.6:
            non_null = numeric.dropna()

            if not non_null.empty and (non_null % 1 == 0).all():
                df[col] = numeric.astype("Int64")
            else:
                df[col] = numeric.astype("float64")

    return df



def remove_constant_columns(
    df: pd.DataFrame,
    input_cols: list[str],
    near_constant_threshold: float = 0.95,
    drop_constant: bool = True,
    drop_near_constant: bool = True,
    return_report: bool = False
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
    return_report: bool = False
):
    df = df.copy()
    input_cols = df.select_dtypes(include=["object", "string"])
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

    print(f"Removed {duplicate_row_num} duplicate rows")

    return df



def check_datetime_monotonic(df, date_col=None):
    if date_col is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index is not DatetimeIndex and no date_col specified")
        dates = df.index
    else:
        dates = pd.to_datetime(df[date_col])
    
    is_monotonic = dates.is_monotonic_increasing
    is_strictly_monotonic = dates.is_monotonic_increasing and not dates.duplicated().any()
    
    if not is_monotonic:
        if isinstance(dates, pd.DatetimeIndex):
            diffs = dates.to_series().diff()
        else:
            diffs = dates.diff()
        negative_diffs = diffs[diffs < timedelta(0)]
        
        return {
            'is_monotonic': False,
            'is_strictly_monotonic': False,
            'violations': len(negative_diffs),
            'violation_indices': negative_diffs.index.tolist()
        }
    elif not is_strictly_monotonic:
        duplicates = dates[dates.duplicated(keep=False)]
        return {
            'is_monotonic': True,
            'is_strictly_monotonic': False,
            'duplicates': len(duplicates),
            'duplicate_dates': duplicates.unique().tolist()
        }
    else:
        return {
            'is_monotonic': True,
            'is_strictly_monotonic': True,
            'message': 'Datetime is strictly monotonically increasing'
        }


def fix_datetime_order(df, date_col=None):
    print("Fixing datetime order...")
    
    if date_col is None:
        df_sorted = df.sort_index()
        print(f"  Sorted by index")
    else:
        df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
        print(f"  Sorted by column '{date_col}'")
    
    return df_sorted


def identify_datetime_duplicates(df, date_col=None):
    if date_col is None:
        duplicated_mask = df.index.duplicated(keep=False)
        duplicate_dates = df.index[duplicated_mask].unique()
    else:
        duplicated_mask = df[date_col].duplicated(keep=False)
        duplicate_dates = df.loc[duplicated_mask, date_col].unique()
    
    duplicate_count = duplicated_mask.sum()
    
    if duplicate_count > 0:
        print(f"\nFound {duplicate_count} duplicate datetime entries across {len(duplicate_dates)} unique dates")
        return df[duplicated_mask]
    else:
        print("\nNo duplicate datetime entries found")
        return pd.DataFrame()


def handle_datetime_duplicates(df, date_col=None, method='mean'):
    duplicates = identify_datetime_duplicates(df, date_col)
    
    if len(duplicates) == 0:
        return df
    
    print(f"Handling duplicates using method: '{method}'")
    
    if date_col is None:
        if method == 'first':
            df_clean = df[~df.index.duplicated(keep='first')]
        elif method == 'last':
            df_clean = df[~df.index.duplicated(keep='last')]
        elif method in ['mean', 'sum']:
            numeric_cols = df.select_dtypes(include='number').columns
            categorical_cols = df.select_dtypes(exclude='number').columns
            
            agg_dict = {}
            for col in numeric_cols:
                agg_dict[col] = method
            for col in categorical_cols:
                agg_dict[col] = 'first'
            
            df_clean = df.groupby(df.index).agg(agg_dict)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        if method == 'first':
            df_clean = df.drop_duplicates(subset=[date_col], keep='first')
        elif method == 'last':
            df_clean = df.drop_duplicates(subset=[date_col], keep='last')
        elif method in ['mean', 'sum']:
            numeric_cols = df.select_dtypes(include='number').columns
            categorical_cols = df.select_dtypes(exclude='number').columns
            
            agg_dict = {}
            for col in numeric_cols:
                if col != date_col:
                    agg_dict[col] = method
            for col in categorical_cols:
                if col != date_col:
                    agg_dict[col] = 'first'
            
            df_clean = df.groupby(date_col).agg(agg_dict).reset_index()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    removed = len(df) - len(df_clean)
    print(f"  Removed {removed} duplicate rows")
    print(f"  Final shape: {df_clean.shape}")
    
    return df_clean


def identify_datetime_gaps(df, date_col=None, expected_freq=None):
    if date_col is None:
        dates = df.index
    else:
        dates = pd.to_datetime(df[date_col])
    
    if expected_freq is None:
        expected_freq = pd.infer_freq(dates)
        if expected_freq is None:
            if isinstance(dates, pd.DatetimeIndex):
                time_diffs = dates.to_series().diff().dropna()
            else:
                time_diffs = dates.diff().dropna()
            median_diff = time_diffs.median()
            
            if median_diff <= pd.Timedelta(hours=1):
                expected_freq = 'H'
            elif median_diff <= pd.Timedelta(days=1):
                expected_freq = 'D'
            elif median_diff <= pd.Timedelta(days=7):
                expected_freq = 'W'
            elif median_diff <= pd.Timedelta(days=31):
                expected_freq = 'MS'
            else:
                expected_freq = 'YS'
            
            print(f"Inferred frequency: {expected_freq}")
    
    expected_range = pd.date_range(start=dates.min(), end=dates.max(), freq=expected_freq)
    
    existing_dates = set(dates)
    expected_dates = set(expected_range)
    missing_dates = sorted(expected_dates - existing_dates)
    
    if len(missing_dates) > 0:
        print(f"\nFound {len(missing_dates)} missing datetime points")
        print(f"  Expected total: {len(expected_range)}")
        print(f"  Actual total: {len(dates)}")
        print(f"  Missing: {len(missing_dates)}")
        
        gaps = []
        if missing_dates:
            gap_start = missing_dates[0]
            gap_end = missing_dates[0]
            
            for i in range(1, len(missing_dates)):
                expected_next = gap_end + pd.tseries.frequencies.to_offset(expected_freq)
                if missing_dates[i] == expected_next:
                    gap_end = missing_dates[i]
                else:
                    gaps.append({
                        'gap_start': gap_start,
                        'gap_end': gap_end,
                        'gap_length': len(pd.date_range(gap_start, gap_end, freq=expected_freq))
                    })
                    gap_start = missing_dates[i]
                    gap_end = missing_dates[i]
            
            gaps.append({
                'gap_start': gap_start,
                'gap_end': gap_end,
                'gap_length': len(pd.date_range(gap_start, gap_end, freq=expected_freq))
            })
        
        gaps_df = pd.DataFrame(gaps)
        print(f"\nIdentified {len(gaps_df)} continuous gap(s):")
        print(gaps_df.to_string(index=False))
        
        return {
            'has_gaps': True,
            'missing_count': len(missing_dates),
            'missing_dates': missing_dates,
            'gaps': gaps_df,
            'expected_freq': expected_freq
        }
    else:
        print("\nNo datetime gaps found - sequence is complete")
        return {
            'has_gaps': False,
            'expected_freq': expected_freq
        }
    

def fill_datetime_gaps(df, date_col=None, method='interpolate', expected_freq=None):
    gap_info = identify_datetime_gaps(df, date_col, expected_freq)
    
    if not gap_info['has_gaps']:
        return df
    
    print(f"\nFilling gaps using method: '{method}'")
    
    freq = gap_info['expected_freq']
    
    if date_col is None:
        complete_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        df_filled = df.reindex(complete_range)
    else:
        complete_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=freq)
        df_complete = pd.DataFrame({date_col: complete_range})
        df_filled = df_complete.merge(df, on=date_col, how='left')
    
    numeric_cols = df_filled.select_dtypes(include='number').columns
    
    if method == 'interpolate':
        df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method='linear', limit_direction='both')
    elif method == 'ffill':
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='ffill')
    elif method == 'bfill':
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='bfill')
    elif method == 'zero':
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(0)
    elif method == 'mean':
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
    else:
        raise ValueError(f"Unknown method: {method}")
    
    categorical_cols = df_filled.select_dtypes(exclude='number').columns
    if len(categorical_cols) > 0:
        df_filled[categorical_cols] = df_filled[categorical_cols].fillna(method='ffill')
        df_filled[categorical_cols] = df_filled[categorical_cols].fillna(method='bfill')
    
    added = len(df_filled) - len(df)
    print(f"  Added {added} rows to fill gaps")
    print(f"  Final shape: {df_filled.shape}")
    
    return df_filled


def analyze_null_patterns(df, date_col=None):
    print("\n" + "="*80)
    print("NULL VALUE ANALYSIS")
    print("="*80)
    
    total_cells = df.shape[0] * df.shape[1]
    total_nulls = df.isnull().sum().sum()
    null_percentage = (total_nulls / total_cells) * 100
    
    print(f"\nOverall Statistics:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Total nulls: {total_nulls:,}")
    print(f"  Null percentage: {null_percentage:.2f}%")
    
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    
    if len(null_counts) > 0:
        print(f"\nColumns with null values:")
        for col, count in null_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")
    else:
        print("\nNo null values found!")
        return {
            'total_nulls': 0,
            'null_percentage': 0,
            'columns_with_nulls': [],
            'partial_null_rows': 0,
            'full_null_rows': 0,
            'null_blocks': []
        }
    
    row_null_counts = df.isnull().sum(axis=1)
    
    partial_null_rows = ((row_null_counts > 0) & (row_null_counts < df.shape[1])).sum()
    full_null_rows = (row_null_counts == df.shape[1]).sum()
    
    print(f"\nRow-wise Analysis:")
    print(f"  Partial null rows (some columns missing): {partial_null_rows}")
    print(f"  Full null rows (all columns missing): {full_null_rows}")
    
    null_blocks = identify_null_blocks(df, date_col)
    
    return {
        'total_nulls': total_nulls,
        'null_percentage': null_percentage,
        'column_nulls': null_counts.to_dict(),
        'partial_null_rows': partial_null_rows,
        'full_null_rows': full_null_rows,
        'null_blocks': null_blocks
    }


def identify_null_blocks(df, date_col=None, min_block_size=3):
    if date_col is None and isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
    elif date_col is not None:
        dates = df[date_col]
    else:
        dates = None
    
    blocks = []
    
    for col in df.columns:
        if col == date_col:
            continue
        
        is_null = df[col].isnull()
        
        null_groups = (is_null != is_null.shift()).cumsum()
        
        for group_id in null_groups[is_null].unique():
            group_indices = null_groups[null_groups == group_id].index
            block_size = len(group_indices)
            
            if block_size >= min_block_size:
                block_info = {
                    'column': col,
                    'start_idx': group_indices[0],
                    'end_idx': group_indices[-1],
                    'block_size': block_size
                }
                
                if dates is not None:
                    block_info['start_date'] = dates.iloc[group_indices[0]]
                    block_info['end_date'] = dates.iloc[group_indices[-1]]
                
                blocks.append(block_info)
    
    if blocks:
        print(f"\nFound {len(blocks)} continuous null block(s) (>= {min_block_size} consecutive nulls):")
        for i, block in enumerate(blocks, 1):
            if 'start_date' in block:
                print(f"  Block {i}: Column '{block['column']}', Size: {block['block_size']}, "
                      f"From {block['start_date']} to {block['end_date']}")
            else:
                print(f"  Block {i}: Column '{block['column']}', Size: {block['block_size']}, "
                      f"Rows {block['start_idx']} to {block['end_idx']}")
    
    return blocks


def handle_time_series_nulls(df, date_col=None, strategy='auto'):
    print("\n" + "="*80)
    print("HANDLING NULL VALUES")
    print("="*80)
    
    df_clean = df.copy()
    
    null_analysis = analyze_null_patterns(df_clean, date_col)
    
    if null_analysis['total_nulls'] == 0:
        print("\nNo null values to handle!")
        return df_clean
    
    print(f"\nUsing strategy: '{strategy}'")
    
    if null_analysis['full_null_rows'] > 0:
        print(f"\nDropping {null_analysis['full_null_rows']} full-null rows...")
        df_clean = df_clean.dropna(how='all')
    
    if strategy in ['auto', 'drop_cols']:
        high_null_cols = []
        for col, count in null_analysis['column_nulls'].items():
            if col != date_col and (count / len(df_clean)) > 0.5:
                high_null_cols.append(col)
        
        if high_null_cols:
            print(f"\nDropping {len(high_null_cols)} columns with >50% nulls:")
            print(f"  {high_null_cols}")
            df_clean = df_clean.drop(columns=high_null_cols)
    
    numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
    if date_col in numeric_cols:
        numeric_cols.remove(date_col)
    
    categorical_cols = df_clean.select_dtypes(exclude='number').columns.tolist()
    if date_col in categorical_cols:
        categorical_cols.remove(date_col)
    
    if strategy == 'auto':
        if numeric_cols:
            print(f"\nInterpolating {len(numeric_cols)} numeric columns...")
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(
                method='linear', 
                limit_direction='both',
                limit_area='inside'
            )
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill')
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='bfill')
        
        if categorical_cols:
            print(f"Forward filling {len(categorical_cols)} categorical columns...")
            df_clean[categorical_cols] = df_clean[categorical_cols].fillna(method='ffill')
            df_clean[categorical_cols] = df_clean[categorical_cols].fillna(method='bfill')
    
    elif strategy == 'interpolate':
        if numeric_cols:
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(
                method='linear', 
                limit_direction='both'
            )
        if categorical_cols:
            df_clean[categorical_cols] = df_clean[categorical_cols].fillna(method='ffill')
    
    elif strategy == 'ffill':
        df_clean = df_clean.fillna(method='ffill')
        df_clean = df_clean.fillna(method='bfill')
    
    elif strategy == 'bfill':
        df_clean = df_clean.fillna(method='bfill')
        df_clean = df_clean.fillna(method='ffill')
    
    elif strategy == 'drop_rows':
        print("\nDropping all rows with any null values...")
        df_clean = df_clean.dropna()
    
    remaining_nulls = df_clean.isnull().sum().sum()
    
    if remaining_nulls > 0:
        print(f"\nWarning: {remaining_nulls} null values still remain")
        print("Consider using a different strategy or manually handling these nulls")
    else:
        print(f"\nAll null values handled successfully!")
    
    print(f"\nFinal shape: {df_clean.shape} (removed {len(df) - len(df_clean)} rows)")
    
    return df_clean


def preprocess_time_series(df, date_col, 
                           handle_duplicates=True, duplicate_method='mean',
                           handle_gaps=True, gap_fill_method='interpolate',
                           handle_nulls=True, null_strategy='auto',
                           expected_freq=None):
    print("\n" + "="*80)
    print("TIME SERIES PREPROCESSING PIPELINE")
    print("="*80)
    
    report = {
        'original_shape': df.shape,
        'steps_applied': []
    }
    
    df_processed = df.copy()
    
    print("\n1. PARSING DATETIME")
    print("-"*80)
    df_processed[date_col] = pd.to_datetime(df_processed[date_col], errors='coerce')
    
    failed_dates = df_processed[date_col].isnull().sum()
    if failed_dates > 0:
        print(f"  Warning: {failed_dates} dates failed to parse - dropping these rows")
        df_processed = df_processed.dropna(subset=[date_col])
        report['steps_applied'].append(f"Dropped {failed_dates} rows with invalid dates")
    
    print("\n2. CHECKING DATETIME ORDER")
    print("-"*80)
    monotonic_check = check_datetime_monotonic(df_processed, date_col)
    print(monotonic_check)
    
    if not monotonic_check['is_monotonic']:
        df_processed = fix_datetime_order(df_processed, date_col)
        report['steps_applied'].append("Fixed datetime ordering")
    
    if handle_duplicates:
        print("\n3. HANDLING DATETIME DUPLICATES")
        print("-"*80)
        initial_len = len(df_processed)
        df_processed = handle_datetime_duplicates(df_processed, date_col, method=duplicate_method)
        removed = initial_len - len(df_processed)
        if removed > 0:
            report['steps_applied'].append(f"Handled {removed} duplicate timestamps using '{duplicate_method}'")
    
    print("\n4. SETTING DATETIME INDEX")
    print("-"*80)
    df_processed = df_processed.set_index(date_col)
    df_processed = df_processed.sort_index()
    print(f"  Index set: {df_processed.index[0]} to {df_processed.index[-1]}")
    
    if handle_gaps:
        print("\n5. HANDLING DATETIME GAPS")
        print("-"*80)
        initial_len = len(df_processed)
        df_processed = fill_datetime_gaps(df_processed, date_col=None, 
                                         method=gap_fill_method, 
                                         expected_freq=expected_freq)
        added = len(df_processed) - initial_len
        if added > 0:
            report['steps_applied'].append(f"Filled {added} datetime gaps using '{gap_fill_method}'")
    
    if handle_nulls:
        print("\n6. HANDLING NULL VALUES")
        print("-"*80)
        initial_len = len(df_processed)
        df_processed = handle_time_series_nulls(df_processed, date_col=None, 
                                                strategy=null_strategy)
        removed = initial_len - len(df_processed)
        if removed > 0:
            report['steps_applied'].append(f"Removed {removed} rows while handling nulls")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nOriginal shape: {report['original_shape']}")
    print(f"Final shape: {df_processed.shape}")
    print(f"Rows added: {df_processed.shape[0] - report['original_shape'][0]}")
    
    print("\nSteps applied:")
    for step in report['steps_applied']:
        print(f"  {step}")
    
    print("\nFinal Validation:")
    final_nulls = df_processed.isnull().sum().sum()
    print(f"  Remaining nulls: {final_nulls}")
    
    monotonic_final = df_processed.index.is_monotonic_increasing
    print(f"  Monotonic increasing: {monotonic_final}")
    
    duplicates_final = df_processed.index.duplicated().sum()
    print(f"  Duplicate timestamps: {duplicates_final}")
    
    report['final_shape'] = df_processed.shape
    report['final_nulls'] = final_nulls
    report['is_monotonic'] = monotonic_final
    report['has_duplicates'] = duplicates_final > 0
    
    return df_processed, report




def dataframe_details(df):
    df_shape = df.shape
    df_null_sum = df.isnull().sum().sum()
    df_duplicate_sum = df.duplicated().sum()
    
    df_columns = df.columns

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

    if len(num_columns) > 0:
        sns.heatmap(df[num_columns].corr(), cmap="inferno", annot=True)
        plt.show()

    return (df_head, df_info, df_describe, df_nunique, df_duplicate_sum, df_null_sum, object_columns, 
            num_columns, date_columns, bool_columns, category_columns)


def preprocessing_pipeline(df: pd.DataFrame, target) -> pd.DataFrame:
    
    df = remove_space(df)
    df = assign_column_name(df)

    if df.shape[1] > 1:
        input_cols = df.drop(columns=[target], axis=1).columns
        df_input = df[input_cols]

    if df_input.shape[1] > 1:
        df_input = remove_duplicate_column(df_input)
    
    df_input = object_to_datetime(df_input)
    df_input = object_to_numeric(df_input)

    if df_input.shape[1] > 1:    
        df_input = remove_constant_columns(df_input, df_input.columns)
        df_input = remove_high_cardinality(df_input)

    df_input = df_input.join(df[target])
    df = remove_duplicate_rows(df_input)

    df_clean, report = preprocess_time_series(df=df, date_col=date_column)

    return df_clean


def preprocessing__ts(df, target):
    # df = pd.read_csv(file_path)   # user uploaded csv


    df_clean = preprocessing_pipeline(df, target)


    (df_head, df_info, df_describe, df_nunique, df_duplicate_sum, df_null_sum, object_columns, 
        num_columns, date_columns, bool_columns, category_columns) = dataframe_details(df_clean)


    df_clean.to_csv("../output/df.csv", index=True)   # csv path (preprocessed)
