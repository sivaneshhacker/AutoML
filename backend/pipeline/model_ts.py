import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')


def create_lag_features(df, target_col, lags=[1, 2, 3, 7, 14, 30]):
    df_features = df.copy()
    for lag in lags:
        df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
    return df_features


def create_rolling_features(df, target_col, windows=[7, 14, 30]):
    df_features = df.copy()
    for window in windows:
        df_features[f'{target_col}_rolling_mean_{window}'] = df_features[target_col].rolling(window=window).mean()
        df_features[f'{target_col}_rolling_std_{window}'] = df_features[target_col].rolling(window=window).std()
        df_features[f'{target_col}_rolling_min_{window}'] = df_features[target_col].rolling(window=window).min()
        df_features[f'{target_col}_rolling_max_{window}'] = df_features[target_col].rolling(window=window).max()
    return df_features


def create_time_features(df):
    df_features = df.copy()
    df_features['year'] = df_features.index.year
    df_features['month'] = df_features.index.month
    df_features['day'] = df_features.index.day
    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['dayofyear'] = df_features.index.dayofyear
    df_features['quarter'] = df_features.index.quarter
    df_features['weekofyear'] = df_features.index.isocalendar().week
    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
    df_features['is_month_start'] = df_features.index.is_month_start.astype(int)
    df_features['is_month_end'] = df_features.index.is_month_end.astype(int)
    df_features['is_quarter_start'] = df_features.index.is_quarter_start.astype(int)
    df_features['is_quarter_end'] = df_features.index.is_quarter_end.astype(int)
    return df_features


def create_difference_features(df, target_col, periods=[1, 7, 30]):
    df_features = df.copy()
    for period in periods:
        df_features[f'{target_col}_diff_{period}'] = df_features[target_col].diff(period)
        df_features[f'{target_col}_pct_change_{period}'] = df_features[target_col].pct_change(period)
    return df_features


def create_exponential_features(df, target_col, spans=[7, 14, 30]):
    df_features = df.copy()
    for span in spans:
        df_features[f'{target_col}_ewm_mean_{span}'] = df_features[target_col].ewm(span=span).mean()
        df_features[f'{target_col}_ewm_std_{span}'] = df_features[target_col].ewm(span=span).std()
    return df_features


def engineer_all_features(df, target_col, 
                          create_lags=True, lag_periods=[1, 2, 3, 7, 14],
                          create_rolling=True, rolling_windows=[7, 14, 30],
                          create_time=True,
                          create_diff=True, diff_periods=[1, 7],
                          create_ewm=True, ewm_spans=[7, 14]):
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df_features = df.copy()
    initial_features = df_features.shape[1]
    
    if create_lags:
        print(f"\nCreating lag features: {lag_periods}")
        df_features = create_lag_features(df_features, target_col, lags=lag_periods)
    
    if create_rolling:
        print(f"Creating rolling features: {rolling_windows}")
        df_features = create_rolling_features(df_features, target_col, windows=rolling_windows)
    
    if create_time:
        print("Creating time-based features")
        df_features = create_time_features(df_features)
    
    if create_diff:
        print(f"Creating difference features: {diff_periods}")
        df_features = create_difference_features(df_features, target_col, periods=diff_periods)
    
    if create_ewm:
        print(f"Creating exponential weighted features: {ewm_spans}")
        df_features = create_exponential_features(df_features, target_col, spans=ewm_spans)
    
    df_features = df_features.dropna()
    
    final_features = df_features.shape[1]
    print(f"\nFeature engineering complete:")
    print(f"  Original features: {initial_features}")
    print(f"  Final features: {final_features}")
    print(f"  New features created: {final_features - initial_features}")
    print(f"  Rows after dropna: {df_features.shape[0]}")
    
    return df_features


def prepare_train_test_split(df, target_col, test_size=0.2):
    print("\n" + "="*80)
    print("TRAIN-TEST SPLIT")
    print("="*80)
    
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    print(f"\nTrain set: {X_train.shape[0]} samples ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Test set: {X_test.shape[0]} samples ({test_df.index[0]} to {test_df.index[-1]})")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, method='standard'):
    print("\n" + "="*80)
    print("FEATURE SCALING")
    print("="*80)
    
    if method == 'standard':
        scaler = StandardScaler()
        scaler_name = 'StandardScaler'
    elif method == 'minmax':
        scaler = MinMaxScaler()
        scaler_name = 'MinMaxScaler'
    elif method == 'robust':
        scaler = RobustScaler()
        scaler_name = 'RobustScaler'
    else:
        scaler = StandardScaler()
        scaler_name = 'StandardScaler'
    
    print(f"Using: {scaler_name}")
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"Scaled {X_train.shape[1]} features")
    
    return X_train_scaled, X_test_scaled, scaler


def get_regression_models():
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        'LightGBM': LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=False),
        'KNN': KNeighborsRegressor(n_jobs=-1),
        'SVR': SVR()
    }
    return models


def get_classification_models():
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
        'KNN': KNeighborsClassifier(n_jobs=-1),
        'SVC': SVC(probability=True, random_state=42)
    }
    return models


def get_regression_param_grid(model_name):
    param_grids = {
        'Ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'Lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0]
        },
        'ElasticNet': {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.2, 0.5, 0.8]
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 50]
        },
        'CatBoost': {
            'iterations': [100, 200],
            'learning_rate': [0.01, 0.1],
            'depth': [4, 6]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'SVR': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear']
        }
    }
    return param_grids.get(model_name, {})


def get_classification_param_grid(model_name):
    param_grids = {
        'LogisticRegression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 50]
        },
        'CatBoost': {
            'iterations': [100, 200],
            'learning_rate': [0.01, 0.1],
            'depth': [4, 6]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'SVC': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear']
        }
    }
    return param_grids.get(model_name, {})


def time_series_cv_regression(model, X_train, y_train, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = {
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'mape': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        
        cv_scores['mse'].append(mean_squared_error(y_val, y_pred))
        cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
        cv_scores['mae'].append(mean_absolute_error(y_val, y_pred))
        cv_scores['r2'].append(r2_score(y_val, y_pred))
        
        mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-10))) * 100
        cv_scores['mape'].append(mape)
    
    cv_scores_mean = {k: np.mean(v) for k, v in cv_scores.items()}
    
    return cv_scores_mean


def time_series_cv_classification(model, X_train, y_train, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    n_classes = len(np.unique(y_train))
    average_method = 'binary' if n_classes == 2 else 'weighted'
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        
        cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
        cv_scores['precision'].append(precision_score(y_val, y_pred, average=average_method, zero_division=0))
        cv_scores['recall'].append(recall_score(y_val, y_pred, average=average_method, zero_division=0))
        cv_scores['f1'].append(f1_score(y_val, y_pred, average=average_method, zero_division=0))
    
    cv_scores_mean = {k: np.mean(v) for k, v in cv_scores.items()}
    
    return cv_scores_mean


def hyperparameter_tuning_regression(model_name, model, param_grid, X_train, y_train, n_splits=3):
    from sklearn.model_selection import GridSearchCV
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_


def hyperparameter_tuning_classification(model_name, model, param_grid, X_train, y_train, n_splits=3):
    from sklearn.model_selection import GridSearchCV
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_


def calculate_regression_metrics(y_true, y_pred):
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    }
    return metrics


def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    n_classes = len(np.unique(y_true))
    average_method = 'binary' if n_classes == 2 else 'weighted'
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average_method, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average_method, zero_division=0),
        'F1': f1_score(y_true, y_pred, average=average_method, zero_division=0)
    }
    
    if y_pred_proba is not None and n_classes == 2:
        try:
            metrics['ROC_AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            metrics['ROC_AUC'] = None
    
    return metrics


def train_regression_models(X_train, X_test, y_train, y_test, 
                           tune_hyperparameters=True, 
                           cv_splits=5,
                           models_to_use=None):
    
    print("\n" + "="*80)
    print("TIME SERIES REGRESSION MODELING")
    print("="*80)
    
    all_models = get_regression_models()
    
    if models_to_use:
        models = {k: v for k, v in all_models.items() if k in models_to_use}
    else:
        models = all_models
    
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*80}")
        print(f"Training: {model_name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        if tune_hyperparameters:
            param_grid = get_regression_param_grid(model_name)
            if param_grid:
                print(f"Hyperparameter tuning with {len(param_grid)} parameters...")
                model, best_params = hyperparameter_tuning_regression(
                    model_name, model, param_grid, X_train, y_train, n_splits=min(3, cv_splits)
                )
                print(f"Best parameters: {best_params}")
            else:
                print("No parameter grid defined, using default parameters")
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        print(f"\nCross-validation ({cv_splits} splits)...")
        cv_scores = time_series_cv_regression(model, X_train, y_train, n_splits=cv_splits)
        
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_metrics = calculate_regression_metrics(y_train, y_train_pred)
        test_metrics = calculate_regression_metrics(y_test, y_test_pred)
        
        training_time = time.time() - start_time
        
        result = {
            'Model': model_name,
            'Train_RMSE': train_metrics['RMSE'],
            'Test_RMSE': test_metrics['RMSE'],
            'Train_MAE': train_metrics['MAE'],
            'Test_MAE': test_metrics['MAE'],
            'Train_R2': train_metrics['R2'],
            'Test_R2': test_metrics['R2'],
            'CV_RMSE': cv_scores['rmse'],
            'CV_MAE': cv_scores['mae'],
            'CV_R2': cv_scores['r2'],
            'Training_Time': training_time
        }
        
        results.append(result)
        trained_models[model_name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        print(f"\nTest Metrics:")
        print(f"  RMSE: {test_metrics['RMSE']:.4f}")
        print(f"  MAE: {test_metrics['MAE']:.4f}")
        print(f"  R2: {test_metrics['R2']:.4f}")
        print(f"  Training Time: {training_time:.2f}s")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_RMSE', ascending=True).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    
    return results_df, trained_models


def train_classification_models(X_train, X_test, y_train, y_test,
                               tune_hyperparameters=True,
                               cv_splits=5,
                               models_to_use=None):
    
    print("\n" + "="*80)
    print("TIME SERIES CLASSIFICATION MODELING")
    print("="*80)
    
    all_models = get_classification_models()
    
    if models_to_use:
        models = {k: v for k, v in all_models.items() if k in models_to_use}
    else:
        models = all_models
    
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*80}")
        print(f"Training: {model_name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        if tune_hyperparameters:
            param_grid = get_classification_param_grid(model_name)
            if param_grid:
                print(f"Hyperparameter tuning with {len(param_grid)} parameters...")
                model, best_params = hyperparameter_tuning_classification(
                    model_name, model, param_grid, X_train, y_train, n_splits=min(3, cv_splits)
                )
                print(f"Best parameters: {best_params}")
            else:
                print("No parameter grid defined, using default parameters")
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        print(f"\nCross-validation ({cv_splits} splits)...")
        cv_scores = time_series_cv_classification(model, X_train, y_train, n_splits=cv_splits)
        
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        y_train_proba = None
        y_test_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_train_proba = model.predict_proba(X_train)
                y_test_proba = model.predict_proba(X_test)
            except:
                pass
        
        train_metrics = calculate_classification_metrics(y_train, y_train_pred, y_train_proba)
        test_metrics = calculate_classification_metrics(y_test, y_test_pred, y_test_proba)
        
        training_time = time.time() - start_time
        
        result = {
            'Model': model_name,
            'Train_Accuracy': train_metrics['Accuracy'],
            'Test_Accuracy': test_metrics['Accuracy'],
            'Train_F1': train_metrics['F1'],
            'Test_F1': test_metrics['F1'],
            'Train_Precision': train_metrics['Precision'],
            'Test_Precision': test_metrics['Precision'],
            'Train_Recall': train_metrics['Recall'],
            'Test_Recall': test_metrics['Recall'],
            'CV_Accuracy': cv_scores['accuracy'],
            'CV_F1': cv_scores['f1'],
            'Training_Time': training_time
        }
        
        if test_metrics.get('ROC_AUC') is not None:
            result['Test_ROC_AUC'] = test_metrics['ROC_AUC']
        
        results.append(result)
        trained_models[model_name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'y_train_proba': y_train_proba,
            'y_test_proba': y_test_proba
        }
        
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {test_metrics['Accuracy']:.4f}")
        print(f"  F1: {test_metrics['F1']:.4f}")
        print(f"  Precision: {test_metrics['Precision']:.4f}")
        print(f"  Recall: {test_metrics['Recall']:.4f}")
        print(f"  Training Time: {training_time:.2f}s")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_Accuracy', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    
    return results_df, trained_models


def plot_regression_results(results_df, trained_models, y_test, output_dir='./model_output'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models_list = results_df['Model'].tolist()
    test_rmse = results_df['Test_RMSE'].tolist()
    test_mae = results_df['Test_MAE'].tolist()
    test_r2 = results_df['Test_R2'].tolist()
    training_time = results_df['Training_Time'].tolist()
    
    axes[0, 0].barh(models_list, test_rmse, color='steelblue')
    axes[0, 0].set_xlabel('RMSE')
    axes[0, 0].set_title('Test RMSE by Model')
    axes[0, 0].invert_yaxis()
    
    axes[0, 1].barh(models_list, test_mae, color='coral')
    axes[0, 1].set_xlabel('MAE')
    axes[0, 1].set_title('Test MAE by Model')
    axes[0, 1].invert_yaxis()
    
    axes[1, 0].barh(models_list, test_r2, color='seagreen')
    axes[1, 0].set_xlabel('R2 Score')
    axes[1, 0].set_title('Test R2 Score by Model')
    axes[1, 0].invert_yaxis()
    
    axes[1, 1].barh(models_list, training_time, color='mediumpurple')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_title('Training Time by Model')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/model_comparison.png")
    
    best_model_name = results_df.iloc[0]['Model']
    best_model_data = trained_models[best_model_name]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    y_test_pred = best_model_data['y_test_pred']
    
    axes[0].scatter(y_test, y_test_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'Actual vs Predicted - {best_model_name}')
    
    residuals = y_test - y_test_pred
    axes[1].scatter(y_test_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'Residual Plot - {best_model_name}')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/best_model_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/best_model_predictions.png")
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    time_index = y_test.index
    ax.plot(time_index, y_test.values, label='Actual', linewidth=2, marker='o', markersize=4)
    ax.plot(time_index, y_test_pred, label='Predicted', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Time Series Prediction - {best_model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_series_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/time_series_prediction.png")


def plot_classification_results(results_df, trained_models, y_test, output_dir='./model_output'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models_list = results_df['Model'].tolist()
    test_accuracy = results_df['Test_Accuracy'].tolist()
    test_f1 = results_df['Test_F1'].tolist()
    test_precision = results_df['Test_Precision'].tolist()
    test_recall = results_df['Test_Recall'].tolist()
    
    axes[0, 0].barh(models_list, test_accuracy, color='steelblue')
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_title('Test Accuracy by Model')
    axes[0, 0].invert_yaxis()
    
    axes[0, 1].barh(models_list, test_f1, color='coral')
    axes[0, 1].set_xlabel('F1 Score')
    axes[0, 1].set_title('Test F1 Score by Model')
    axes[0, 1].invert_yaxis()
    
    axes[1, 0].barh(models_list, test_precision, color='seagreen')
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_title('Test Precision by Model')
    axes[1, 0].invert_yaxis()
    
    axes[1, 1].barh(models_list, test_recall, color='mediumpurple')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_title('Test Recall by Model')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/model_comparison.png")
    
    best_model_name = results_df.iloc[0]['Model']
    best_model_data = trained_models[best_model_name]
    
    y_test_pred = best_model_data['y_test_pred']
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/confusion_matrix.png")
    
    if best_model_data['y_test_proba'] is not None:
        from sklearn.metrics import roc_curve, auc
        
        n_classes = len(np.unique(y_test))
        
        if n_classes == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            y_test_proba = best_model_data['y_test_proba'][:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {best_model_name}')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_dir}/roc_curve.png")


def time_series_regression_pipeline(df, target_col, date_col=None,
                                   test_size=0.2,
                                   scale_method='standard',
                                   tune_hyperparameters=True,
                                   cv_splits=5,
                                   models_to_use=None,
                                   output_dir='./ts_regression_output'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("TIME SERIES REGRESSION PIPELINE")
    print("="*80)
    
    df_features = engineer_all_features(
        df, target_col,
        create_lags=True, lag_periods=[1, 2, 3, 7, 14],
        create_rolling=True, rolling_windows=[7, 14, 30],
        create_time=True,
        create_diff=True, diff_periods=[1, 7],
        create_ewm=True, ewm_spans=[7, 14]
    )
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df_features, target_col, test_size=test_size
    )
    
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, method=scale_method
    )
    
    results_df, trained_models = train_regression_models(
        X_train_scaled, X_test_scaled, y_train, y_test,
        tune_hyperparameters=tune_hyperparameters,
        cv_splits=cv_splits,
        models_to_use=models_to_use
    )
    
    results_df.to_csv(f'{output_dir}/model_results.csv', index=False)
    print(f"\nSaved results to: {output_dir}/model_results.csv")
    
    plot_regression_results(results_df, trained_models, y_test, output_dir=output_dir)
    
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]['model']
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}")
    print(f"Test R2: {results_df.iloc[0]['Test_R2']:.4f}")
    
    return {
        'results_df': results_df,
        'trained_models': trained_models,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'scaler': scaler,
        'feature_columns': X_train.columns.tolist()
    }


def time_series_classification_pipeline(df, target_col, date_col=None,
                                       test_size=0.2,
                                       scale_method='standard',
                                       tune_hyperparameters=True,
                                       cv_splits=5,
                                       models_to_use=None,
                                       output_dir='./ts_classification_output'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("TIME SERIES CLASSIFICATION PIPELINE")
    print("="*80)
    
    df_features = engineer_all_features(
        df, target_col,
        create_lags=True, lag_periods=[1, 2, 3, 7, 14],
        create_rolling=True, rolling_windows=[7, 14, 30],
        create_time=True,
        create_diff=True, diff_periods=[1, 7],
        create_ewm=True, ewm_spans=[7, 14]
    )
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df_features, target_col, test_size=test_size
    )
    
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, method=scale_method
    )
    
    results_df, trained_models = train_classification_models(
        X_train_scaled, X_test_scaled, y_train, y_test,
        tune_hyperparameters=tune_hyperparameters,
        cv_splits=cv_splits,
        models_to_use=models_to_use
    )
    
    results_df.to_csv(f'{output_dir}/model_results.csv', index=False)
    print(f"\nSaved results to: {output_dir}/model_results.csv")
    
    plot_classification_results(results_df, trained_models, y_test, output_dir=output_dir)
    
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]['model']
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {results_df.iloc[0]['Test_Accuracy']:.4f}")
    print(f"Test F1: {results_df.iloc[0]['Test_F1']:.4f}")
    
    return {
        'results_df': results_df,
        'trained_models': trained_models,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'scaler': scaler,
        'feature_columns': X_train.columns.tolist()
    }

def model_ts(df, task, target):
    # df = pd.read_csv("")   # csv path (preprocessed)

    # task = ""
    # target = ""

    if task == "time_series_regression":
        results = time_series_regression_pipeline(df, target_col=target)

    elif task == "time_series_classification":
        results = time_series_classification_pipeline(df, target_col=target)
    
    return results