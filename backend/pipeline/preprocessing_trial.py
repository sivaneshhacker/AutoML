import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class AutoPreprocessor:
    """
    Comprehensive preprocessing pipeline for automobile datasets.
    Handles duplicates, missing values, outliers, unit standardization,
    scaling, normalization, and encoding.
    """
    
    def __init__(self,
                 scaling_method='standard',  # 'standard', 'minmax', or None
                 missing_strategy='auto',     # 'auto', 'mean', 'median', 'knn', 'drop'
                 outlier_method='iqr',        # 'iqr', 'zscore', or None
                 outlier_threshold=1.5):      # IQR multiplier or z-score threshold
        
        self.scaling_method = scaling_method
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        # Store fitted transformers for consistency
        self.scalers = {}
        self.label_encoders = {}
        self.imputers = {}
        self.feature_names = None
        self.categorical_cols = []
        self.numerical_cols = []
        
        # Preprocessing report
        self.report = {
            'duplicates_removed': 0,
            'missing_values_handled': {},
            'outliers_handled': {},
            'units_standardized': {},
            'columns_encoded': [],
            'columns_scaled': []
        }
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        print("=" * 70)
        print("AUTOMOBILE DATASET PREPROCESSING PIPELINE")
        print("=" * 70)
        
        df = df.copy()
        
        # 0. Handle various missing value representations
        df = self._recognize_missing_values(df)
        
        # 1. Basic cleaning
        df = self._handle_duplicates(df)
        df = self._standardize_column_names(df)
        
        # 2. Identify column types
        self._identify_column_types(df)
        
        # 3. Unit standardization (automobile-specific)
        df = self._standardize_units(df)
        
        # 4. Handle missing values
        df = self._handle_missing_values(df)
        
        # 5. Handle outliers
        df = self._handle_outliers(df)
        
        # 6. Encode categorical variables
        df = self._encode_categorical(df)
        
        # 7. Scale/normalize numerical features
        df = self._scale_features(df)
        
        # Store final feature names
        self.feature_names = df.columns.tolist()
        
        # Print report
        self._print_report()
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        df = self._standardize_column_names(df)
        df = self._standardize_units(df)
        
        # Apply fitted transformations
        for col in self.numerical_cols:
            if col in df.columns and col in self.imputers:
                df[[col]] = self.imputers[col].transform(df[[col]])
        
        for col in self.categorical_cols:
            if col in df.columns and col in self.label_encoders:
                # Handle unseen categories
                df[col] = df[col].map(lambda x: x if x in self.label_encoders[col].classes_ 
                                      else self.label_encoders[col].classes_[0])
                df[col] = self.label_encoders[col].transform(df[col])
        
        for col in self.numerical_cols:
            if col in df.columns and col in self.scalers:
                df[[col]] = self.scalers[col].transform(df[[col]])
        
        return df[self.feature_names]
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        self.report['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            print(f"\n✓ Removed {duplicates_removed} duplicate rows")
        else:
            print("\n✓ No duplicates found")
        
        return df
    
    def _recognize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recognize and convert various missing value representations to NaN.
        Common representations: ?, NA, N/A, na, n/a, -, --, empty strings, etc.
        """
        # Common missing value representations
        missing_values = ['?', 'NA', 'N/A', 'na', 'n/a', 'nan', 'NaN', 
                         '-', '--', '', ' ', 'null', 'NULL', 'None', 'NONE',
                         'missing', 'MISSING', 'unknown', 'UNKNOWN']
        
        # Replace in all columns
        df = df.replace(missing_values, np.nan)
        
        # Also handle whitespace-only strings
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
                df[col] = df[col].replace('', np.nan)
        
        missing_found = df.isnull().sum().sum()
        if missing_found > 0:
            print(f"\n✓ Recognized {missing_found} missing values (?, NA, -, etc.)")
        
        return df
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        df.columns = [col.lower().strip().replace(' ', '_').replace('-', '_') 
                      for col in df.columns]
        return df
    
    def _identify_column_types(self, df: pd.DataFrame):
        """Identify numerical and categorical columns."""
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"\n✓ Identified {len(self.numerical_cols)} numerical columns")
        print(f"✓ Identified {len(self.categorical_cols)} categorical columns")
    
    def _standardize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize units in automobile datasets.
        Common conversions:
        - MPG variations
        - Weight (lbs to kg)
        - Engine displacement
        - Power (hp variations)
        """
        print("\n" + "-" * 70)
        print("UNIT STANDARDIZATION")
        print("-" * 70)
        
        # MPG standardization
        mpg_cols = [col for col in df.columns if 'mpg' in col.lower() or 'mileage' in col.lower()]
        for col in mpg_cols:
            if col in self.numerical_cols:
                # Convert km/l to mpg if needed (1 km/l = 2.352 mpg)
                if df[col].max() < 50:  # Likely km/l
                    df[col] = df[col] * 2.352
                    self.report['units_standardized'][col] = 'km/l → mpg'
                    print(f"  • {col}: Converted km/l to mpg")
        
        # Weight standardization (convert to kg if in lbs)
        weight_cols = [col for col in df.columns if 'weight' in col.lower() or 'curb' in col.lower()]
        for col in weight_cols:
            if col in self.numerical_cols and df[col].max() > 10000:  # Likely in lbs
                df[col] = df[col] * 0.453592  # lbs to kg
                self.report['units_standardized'][col] = 'lbs → kg'
                print(f"  • {col}: Converted lbs to kg")
        
        # Engine displacement standardization
        engine_cols = [col for col in df.columns if 'engine' in col.lower() or 'displacement' in col.lower()]
        for col in engine_cols:
            if col in self.numerical_cols:
                # Ensure consistent unit (liters)
                if df[col].max() > 100:  # Likely in cc
                    df[col] = df[col] / 1000  # cc to liters
                    self.report['units_standardized'][col] = 'cc → liters'
                    print(f"  • {col}: Converted cc to liters")
        
        # Horsepower standardization
        hp_cols = [col for col in df.columns if 'horse' in col.lower() or 'power' in col.lower() or col.lower() == 'hp']
        for col in hp_cols:
            if col in self.numerical_cols:
                # Clean string values if present
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.extract('(\d+\.?\d*)')[0].astype(float)
        
        if not self.report['units_standardized']:
            print("  • No unit conversions needed")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using various strategies."""
        print("\n" + "-" * 70)
        print("MISSING VALUE IMPUTATION")
        print("-" * 70)
        
        total_missing = df.isnull().sum().sum()
        
        if total_missing == 0:
            print("  • No missing values found")
            return df
        
        # Numerical columns
        for col in self.numerical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                
                if self.missing_strategy == 'auto':
                    # Auto strategy: KNN if <30% missing, median otherwise
                    if missing_pct < 30:
                        strategy = 'knn'
                    else:
                        strategy = 'median'
                else:
                    strategy = self.missing_strategy
                
                if strategy == 'knn':
                    self.imputers[col] = KNNImputer(n_neighbors=5)
                    df[[col]] = self.imputers[col].fit_transform(df[[col]])
                    method = 'KNN'
                elif strategy == 'mean':
                    self.imputers[col] = SimpleImputer(strategy='mean')
                    df[[col]] = self.imputers[col].fit_transform(df[[col]])
                    method = 'Mean'
                elif strategy == 'median':
                    self.imputers[col] = SimpleImputer(strategy='median')
                    df[[col]] = self.imputers[col].fit_transform(df[[col]])
                    method = 'Median'
                elif strategy == 'drop':
                    df = df.dropna(subset=[col])
                    method = 'Dropped'
                
                self.report['missing_values_handled'][col] = {
                    'count': missing_count,
                    'percentage': f"{missing_pct:.2f}%",
                    'method': method
                }
                print(f"  • {col}: {missing_count} values ({missing_pct:.1f}%) - {method}")
        
        # Categorical columns
        for col in self.categorical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                
                # Use mode for categorical
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                
                self.report['missing_values_handled'][col] = {
                    'count': missing_count,
                    'percentage': f"{missing_pct:.2f}%",
                    'method': 'Mode'
                }
                print(f"  • {col}: {missing_count} values ({missing_pct:.1f}%) - Mode")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numerical columns."""
        if self.outlier_method is None:
            return df
        
        print("\n" + "-" * 70)
        print("OUTLIER HANDLING")
        print("-" * 70)
        
        for col in self.numerical_cols:
            initial_count = len(df)
            
            if self.outlier_method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    # Cap outliers instead of removing
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    self.report['outliers_handled'][col] = {
                        'count': outliers,
                        'method': 'IQR capping'
                    }
                    print(f"  • {col}: {outliers} outliers capped using IQR")
            
            elif self.outlier_method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > self.outlier_threshold).sum()
                
                if outliers > 0:
                    # Cap at threshold * std
                    mean = df[col].mean()
                    std = df[col].std()
                    lower_bound = mean - self.outlier_threshold * std
                    upper_bound = mean + self.outlier_threshold * std
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    self.report['outliers_handled'][col] = {
                        'count': outliers,
                        'method': 'Z-score capping'
                    }
                    print(f"  • {col}: {outliers} outliers capped using Z-score")
        
        if not self.report['outliers_handled']:
            print("  • No significant outliers detected")
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        if not self.categorical_cols:
            return df
        
        print("\n" + "-" * 70)
        print("CATEGORICAL ENCODING")
        print("-" * 70)
        
        for col in self.categorical_cols:
            # Use label encoding for now (can be extended to one-hot for low cardinality)
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            self.report['columns_encoded'].append(col)
            print(f"  • {col}: Label encoded ({len(self.label_encoders[col].classes_)} categories)")
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale/normalize numerical features."""
        if self.scaling_method is None:
            return df
        
        print("\n" + "-" * 70)
        print("FEATURE SCALING")
        print("-" * 70)
        
        for col in self.numerical_cols:
            if self.scaling_method == 'standard':
                self.scalers[col] = StandardScaler()
                df[[col]] = self.scalers[col].fit_transform(df[[col]])
                method = 'StandardScaler'
            elif self.scaling_method == 'minmax':
                self.scalers[col] = MinMaxScaler()
                df[[col]] = self.scalers[col].fit_transform(df[[col]])
                method = 'MinMaxScaler'
            
            self.report['columns_scaled'].append(col)
        
        print(f"  • Applied {method} to {len(self.numerical_cols)} numerical columns")
        
        return df
    
    def _print_report(self):
        """Print preprocessing summary report."""
        print("\n" + "=" * 70)
        print("PREPROCESSING SUMMARY")
        print("=" * 70)
        print(f"✓ Duplicates removed: {self.report['duplicates_removed']}")
        print(f"✓ Missing values handled: {len(self.report['missing_values_handled'])} columns")
        print(f"✓ Outliers handled: {len(self.report['outliers_handled'])} columns")
        print(f"✓ Units standardized: {len(self.report['units_standardized'])} columns")
        print(f"✓ Categorical encoded: {len(self.report['columns_encoded'])} columns")
        print(f"✓ Numerical scaled: {len(self.report['columns_scaled'])} columns")
        print("=" * 70)
    
    def get_report(self) -> Dict:
        """Return detailed preprocessing report."""
        return self.report
