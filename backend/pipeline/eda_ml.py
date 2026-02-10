import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")


# =====================================================================
# SECTION 7: EDA FUNCTIONS
# =====================================================================

def detect_leakage(X: pd.DataFrame, y: pd.Series, threshold: float = 0.99):
    """Detect potential data leakage"""
    leakage_cols = []

    for col in X.columns:
        if X[col].dtype in [np.number]:
            corr = abs(X[col].corr(y))
            if corr >= threshold:
                leakage_cols.append((col, corr))

    if leakage_cols:
        print("\nPOTENTIAL DATA LEAKAGE DETECTED:")
        for col, corr in leakage_cols:
            print(f"  - {col}: correlation = {corr:.4f}")

    return leakage_cols


def eda_numerical(df: pd.DataFrame, target: str):
    """EDA for numerical features"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target in num_cols:
        num_cols.remove(target)

    if len(num_cols) == 0:
        print("No numerical features found.")
        return

    print(f"\n{'='*60}")
    print("NUMERICAL FEATURES EDA")
    print(f"{'='*60}")

    # Distribution plots
    n_cols = min(3, len(num_cols))
    n_rows = (len(num_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for idx, col in enumerate(num_cols):
        if idx < len(axes):
            axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')

    # Hide extra subplots
    for idx in range(len(num_cols), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    # Statistics
    print("\nNumerical Features Statistics:")
    print(df[num_cols].describe())


def eda_categorical(df: pd.DataFrame, target: str):
    """EDA for categorical features"""
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if target in cat_cols:
        cat_cols.remove(target)

    if len(cat_cols) == 0:
        print("No categorical features found.")
        return

    print(f"\n{'='*60}")
    print("CATEGORICAL FEATURES EDA")
    print(f"{'='*60}")

    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
        print(f"Unique values: {df[col].nunique()}")

    # Bar plots for categorical features (limit to top categories)
    n_cols = min(2, len(cat_cols))
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for idx, col in enumerate(cat_cols):
        if idx < len(axes):
            top_cats = df[col].value_counts().head(10)
            axes[idx].bar(range(len(top_cats)), top_cats.values)
            axes[idx].set_xticks(range(len(top_cats)))
            axes[idx].set_xticklabels(top_cats.index, rotation=45, ha='right')
            axes[idx].set_title(f'Top Categories in {col}')
            axes[idx].set_ylabel('Count')

    # Hide extra subplots
    for idx in range(len(cat_cols), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def eda_target(df: pd.DataFrame, target: str, task: str):
    """EDA for target variable"""
    print(f"\n{'='*60}")
    print(f"TARGET VARIABLE: {target}")
    print(f"{'='*60}")

    if task == "regression":
        print(f"\nTarget Statistics:")
        print(df[target].describe())

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(df[target].dropna(), bins=30, edgecolor='black')
        plt.title(f'Distribution of {target}')
        plt.xlabel(target)
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        plt.boxplot(df[target].dropna())
        plt.title(f'Boxplot of {target}')
        plt.ylabel(target)

        plt.tight_layout()
        plt.show()

    else:  # classification
        print(f"\nTarget Distribution:")
        print(df[target].value_counts())
        print(f"\nTarget Proportions:")
        print(df[target].value_counts(normalize=True))

        plt.figure(figsize=(10, 5))
        df[target].value_counts().plot(kind='bar', edgecolor='black')
        plt.title(f'Distribution of {target}')
        plt.xlabel(target)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# =====================================================================
# SECTION 8: OUTLIER DETECTION & HANDLING
# =====================================================================

def detect_outliers_iqr(df: pd.DataFrame, target: str):
    """Detect outliers using IQR method"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target in num_cols:
        num_cols.remove(target)

    if len(num_cols) == 0:
        return {}

    outlier_info = {}

    print(f"\n{'='*60}")
    print("OUTLIER DETECTION (IQR Method)")
    print(f"{'='*60}")

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100

        outlier_info[col] = {
            'count': outlier_count,
            'percentage': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

        if outlier_count > 0:
            print(f"\n{col}:")
            print(f"  Outliers: {outlier_count} ({outlier_pct:.2f}%)")
            print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

    return outlier_info


def visualize_outliers(df: pd.DataFrame, target: str):
    """Visualize outliers with boxplots"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target in num_cols:
        num_cols.remove(target)

    if len(num_cols) == 0:
        return

    n_cols = min(3, len(num_cols))
    n_rows = (len(num_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for idx, col in enumerate(num_cols):
        if idx < len(axes):
            axes[idx].boxplot(df[col].dropna())
            axes[idx].set_title(f'Boxplot of {col}')
            axes[idx].set_ylabel(col)

    for idx in range(len(num_cols), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def handle_outliers(df: pd.DataFrame, target: str, method: str = "clip"):
    """Handle outliers using specified method"""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target in num_cols:
        num_cols.remove(target)

    if method == "clip":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    elif method == "log":
        for col in num_cols:
            if (df[col] > 0).all():
                df[col] = np.log1p(df[col])

    return df


# =====================================================================
# SECTION 9: SKEWNESS DETECTION & TRANSFORMATION
# =====================================================================

def detect_skewness(df: pd.DataFrame, target: str, threshold: float = 0.5):
    """Detect skewness in numerical features"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target in num_cols:
        num_cols.remove(target)

    if len(num_cols) == 0:
        return {}

    skew_info = {}

    print(f"\n{'='*60}")
    print("SKEWNESS DETECTION")
    print(f"{'='*60}")

    for col in num_cols:
        skew_val = df[col].skew()
        skew_info[col] = skew_val

        if abs(skew_val) > threshold:
            print(f"\n{col}: {skew_val:.3f} (Skewed)")
        else:
            print(f"{col}: {skew_val:.3f}")

    return skew_info


def transform_skewness(df: pd.DataFrame, target: str, threshold: float = 0.5):
    """Transform skewed features"""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target in num_cols:
        num_cols.remove(target)

    transformed_cols = []

    for col in num_cols:
        skew_val = df[col].skew()

        if abs(skew_val) > threshold:
            if (df[col] > 0).all():
                df[col] = np.log1p(df[col])
                transformed_cols.append(col)

    if transformed_cols:
        print(f"\nTransformed columns: {transformed_cols}")

    return df


def visualize_skewness(df: pd.DataFrame, target: str):
    """Visualize skewness with distribution plots"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target in num_cols:
        num_cols.remove(target)

    if len(num_cols) == 0:
        return

    n_cols = min(3, len(num_cols))
    n_rows = (len(num_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for idx, col in enumerate(num_cols):
        if idx < len(axes):
            axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            skew_val = df[col].skew()
            axes[idx].set_title(f'{col} (Skew: {skew_val:.2f})')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')

    for idx in range(len(num_cols), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


# =====================================================================
# SECTION 10: CLASS IMBALANCE HANDLING
# =====================================================================

def check_class_imbalance(y: pd.Series, threshold: float = 0.3):
    """Check for class imbalance"""
    value_counts = y.value_counts()
    proportions = y.value_counts(normalize=True)

    print(f"\n{'='*60}")
    print("CLASS IMBALANCE CHECK")
    print(f"{'='*60}")
    print(f"\nClass Distribution:")
    print(value_counts)
    print(f"\nClass Proportions:")
    print(proportions)

    min_prop = proportions.min()

    if min_prop < threshold:
        print(f"\nCLASS IMBALANCE DETECTED!")
        print(f"Minimum class proportion: {min_prop:.2%}")
        return True
    else:
        print(f"\n✓ Classes are balanced (minimum proportion: {min_prop:.2%})")
        return False


def handle_class_imbalance(X: pd.DataFrame, y: pd.Series, method: str = "smote"):
    """Handle class imbalance"""
    if method == "smote":
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"\nApplied SMOTE. New shape: {X_resampled.shape}")
            return X_resampled, y_resampled
        except:
            print("SMOTE failed. Using RandomOverSampler instead.")
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            print(f"\nApplied RandomOverSampler. New shape: {X_resampled.shape}")
            return X_resampled, y_resampled

    elif method == "oversample":
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        print(f"\nApplied RandomOverSampler. New shape: {X_resampled.shape}")
        return X_resampled, y_resampled

    elif method == "undersample":
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        print(f"\nApplied RandomUnderSampler. New shape: {X_resampled.shape}")
        return X_resampled, y_resampled

    else:
        print("No resampling applied.")
        return X, y


# =====================================================================
# SECTION 11: MULTICOLLINEARITY DETECTION
# =====================================================================

def detect_multicollinearity_corr(df: pd.DataFrame, target: str, threshold: float = 0.8):
    """Detect multicollinearity using correlation matrix"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target in num_cols:
        num_cols.remove(target)

    if len(num_cols) < 2:
        print("Not enough numerical features for multicollinearity check.")
        return []

    corr_matrix = df[num_cols].corr().abs()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # Find highly correlated pairs
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    if high_corr_pairs:
        print(f"\n{'='*60}")
        print("HIGH CORRELATION DETECTED")
        print(f"{'='*60}")
        for col1, col2, corr_val in high_corr_pairs:
            print(f"{col1} <-> {col2}: {corr_val:.3f}")

    return high_corr_pairs


def calculate_vif(df: pd.DataFrame, target: str):
    """Calculate Variance Inflation Factor"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target in num_cols:
        num_cols.remove(target)

    if len(num_cols) < 2:
        print("Not enough numerical features for VIF calculation.")
        return pd.DataFrame()

    X = df[num_cols]

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(f"\n{'='*60}")
    print("VARIANCE INFLATION FACTOR (VIF)")
    print(f"{'='*60}")
    print("\nVIF > 10 indicates high multicollinearity")
    print(vif_data.sort_values('VIF', ascending=False))

    return vif_data


def remove_multicollinear_features(df: pd.DataFrame, target: str, vif_threshold: float = 10):
    """Remove features with high VIF"""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target in num_cols:
        num_cols.remove(target)

    if len(num_cols) < 2:
        return df, []

    removed_cols = []

    while True:
        X = df[num_cols]
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        max_vif = vif_data['VIF'].max()

        if max_vif > vif_threshold:
            col_to_remove = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
            print(f"Removing {col_to_remove} (VIF: {max_vif:.2f})")
            num_cols.remove(col_to_remove)
            removed_cols.append(col_to_remove)
        else:
            break

    df = df[num_cols + [target]]

    return df, removed_cols


# =====================================================================
# SECTION 12: FEATURE-TARGET RELATIONSHIP
# =====================================================================

def analyze_feature_target_relationship(df: pd.DataFrame, target: str, task: str):
    """Analyze relationship between features and target"""
    print(f"\n{'='*60}")
    print("FEATURE-TARGET RELATIONSHIP")
    print(f"{'='*60}")

    if task == "regression":
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target in num_cols:
            num_cols.remove(target)

        if len(num_cols) == 0:
            print("No numerical features for correlation analysis.")
            return

        correlations = df[num_cols + [target]].corr()[target].drop(target).sort_values(ascending=False)

        print("\nCorrelation with Target:")
        print(correlations)

        # Visualize
        plt.figure(figsize=(10, max(6, len(correlations) * 0.3)))
        correlations.plot(kind='barh')
        plt.title(f'Feature Correlation with {target}')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.show()

        return correlations
    

def comprehensive_eda_ml(df, target, task,
        test_size=0.2,
        handle_outliers_flag=True,
        outlier_method="clip",
        handle_skewness_flag=True,
        handle_imbalance_flag=True,
        imbalance_method="smote",
        remove_multicollinearity_flag=True,
        vif_threshold=10,
        n_features=10):
    
    print("\n[3/15] Analyzing target variable...")
    eda_target(df, target, task)

    # Step 4: EDA - Numerical
    print("\n[4/15] Analyzing numerical features...")
    eda_numerical(df, target)

    # Step 5: EDA - Categorical
    print("\n[5/15] Analyzing categorical features...")
    eda_categorical(df, target)

    # Step 6: Train-test split
    print("\n[6/15] Splitting data...")
    X = df.drop(columns=[target])
    y = df[target]

    if task == "classification":
        y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if task=="classification" else None
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Step 7: Leakage detection
    print("\n[7/15] Detecting potential data leakage...")
    detect_leakage(X_train, y_train)

    # Step 8: Outlier detection and handling
    if handle_outliers_flag:
        print("\n[8/15] Detecting and handling outliers...")
        outlier_info = detect_outliers_iqr(pd.concat([X_train, y_train], axis=1), target)
        visualize_outliers(pd.concat([X_train, y_train], axis=1), target)

        temp_df = pd.concat([X_train, y_train], axis=1)
        temp_df = handle_outliers(temp_df, target, method=outlier_method)
        X_train = temp_df.drop(columns=[target])
        y_train = temp_df[target]
        print("Outliers handled.")
    else:
        print("\n[8/15] Skipping outlier handling...")

    # Step 9: Skewness detection and transformation
    if handle_skewness_flag:
        print("\n[9/15] Detecting and transforming skewness...")
        skew_info = detect_skewness(pd.concat([X_train, y_train], axis=1), target)
        visualize_skewness(pd.concat([X_train, y_train], axis=1), target)

        temp_df = pd.concat([X_train, y_train], axis=1)
        temp_df = transform_skewness(temp_df, target)
        X_train = temp_df.drop(columns=[target])
        y_train = temp_df[target]
        print("Skewness transformed.")
    else:
        print("\n[9/15] Skipping skewness transformation...")

    # Step 10: Class imbalance handling (classification only)
    if task == "classification":
        print("\n[10/15] Checking class imbalance...")
        is_imbalanced = check_class_imbalance(y_train)

        if is_imbalanced and handle_imbalance_flag:
            print("Handling class imbalance...")
            X_train, y_train = handle_class_imbalance(X_train, y_train, method=imbalance_method)
        elif is_imbalanced:
            print("Imbalance detected but not handled (set handle_imbalance_flag=True to handle)")
    else:
        print("\n[10/15] Skipping class imbalance check (regression task)...")

    # Step 11: Multicollinearity detection
    print("\n[11/15] Detecting multicollinearity...")
    temp_df = pd.concat([X_train, y_train], axis=1)
    high_corr_pairs = detect_multicollinearity_corr(temp_df, target)
    vif_data = calculate_vif(temp_df, target)

    if remove_multicollinearity_flag:
        print("Removing multicollinear features...")
        temp_df, removed_cols = remove_multicollinear_features(temp_df, target, vif_threshold)
        X_train = temp_df.drop(columns=[target])
        X_test = X_test.drop(columns=removed_cols, errors='ignore')
        print(f"Removed {len(removed_cols)} features due to multicollinearity")

    # Step 12: Feature-target relationship
    print("\n[12/15] Analyzing feature-target relationship...")
    analyze_feature_target_relationship(pd.concat([X_train, y_train], axis=1), target, task)