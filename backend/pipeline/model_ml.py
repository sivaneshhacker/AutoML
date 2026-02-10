import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, HuberRegressor,
    PassiveAggressiveRegressor, SGDClassifier,
    PassiveAggressiveClassifier
)

from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor,
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)

from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_ridge import KernelRidge

import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
import mlflow
import shap

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    StandardScaler
)

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    f1_score
)

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)

from skopt import BayesSearchCV

def remove_duplicate_rows(df):
    return df.drop_duplicates()

def handle_null_values(df):
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

'''
def show_target_correlation(df, target):
    corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    print("\nCorrelation with Target:\n")
    print(corr)
    return corr
'''

def encode_features(df):
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    joblib.dump(encoders, "../models/encoder.pkl")
    return df

def scale_features(df):
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    df[num_cols] = scaler.fit_transform(df[num_cols])
    joblib.dump(scaler, "../models/scaler.pkl")
    return df

def select_features(X, y, task, k=10):
    if task == "regression":
        selector = SelectKBest(f_regression, k=k)
    else:
        selector = SelectKBest(mutual_info_classif, k=k)

    X_new = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()]
    joblib.dump(selected_cols, "../models/selected_features.pkl")

    return pd.DataFrame(X_new, columns=selected_cols)

def get_models(task):

    # ================= REGRESSION =================
    if task == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),

            "SVR": SVR(),

            "KNN": KNeighborsRegressor(),

            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "DecisionTreeDepth5": DecisionTreeRegressor(max_depth=5, random_state=42),

            "RandomForest": RandomForestRegressor(random_state=42),
            "ExtraTrees": ExtraTreesRegressor(random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42),

            "HuberRegressor": HuberRegressor(),
            "PassiveAggressive": PassiveAggressiveRegressor(),
            "KernelRidge": KernelRidge()
        }

    # ================= CLASSIFICATION =================
    else:
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),

            "SVC_RBF": SVC(kernel="rbf"),
            "SVC_Linear": SVC(kernel="linear"),

            "KNN": KNeighborsClassifier(),

            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "DecisionTreeDepth5": DecisionTreeClassifier(max_depth=5, random_state=42),

            "RandomForest": RandomForestClassifier(random_state=42),
            "ExtraTrees": ExtraTreesClassifier(random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42),

            "GaussianNB": GaussianNB(),

            "SGDClassifier": SGDClassifier(),
            "PassiveAggressive": PassiveAggressiveClassifier(),

            "RidgeClassifier": LogisticRegression(penalty="l2"),
            "LinearSVC": SVC(kernel="linear", probability=True)
        }

def bayesian_tuning(model, X, y):
    search = BayesSearchCV(
        model,
        {
            "n_estimators": (50, 300),
            "max_depth": (3, 15)
        },
        n_iter=10,
        cv=5,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_

def evaluate_models(models, X_train, X_test, y_train, y_test, task):
    rows = []

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            joblib.dump(model, f"../models/{name}.pkl")

            ytr = model.predict(X_train)
            yte = model.predict(X_test)

            # ================= REGRESSION =================
            if task == "regression":
                train_r2 = r2_score(y_train, ytr)
                test_r2 = r2_score(y_test, yte)

                train_mse = mean_squared_error(y_train, ytr)
                test_mse = mean_squared_error(y_test, yte)

                train_rmse = np.sqrt(train_mse)
                test_rmse = np.sqrt(test_mse)

                train_mae = mean_absolute_error(y_train, ytr)
                test_mae = mean_absolute_error(y_test, yte)

                # MLflow logging
                mlflow.log_metric("train_r2", train_r2)
                mlflow.log_metric("test_r2", test_r2)
                mlflow.log_metric("train_rmse", train_rmse)
                mlflow.log_metric("test_rmse", test_rmse)
                mlflow.log_metric("train_mae", train_mae)
                mlflow.log_metric("test_mae", test_mae)
                mlflow.log_metric("train_time", train_time)

                rows.append({
                    "Model": name,
                    "Train_R2": train_r2,
                    "Test_R2": test_r2,
                    "Train_RMSE": train_rmse,
                    "Test_RMSE": test_rmse,
                    "Train_MAE": train_mae,
                    "Test_MAE": test_mae,
                    "Train_Time(s)": train_time
                })

            # ================= CLASSIFICATION =================
            else:
                train_acc = accuracy_score(y_train, ytr)
                test_acc = accuracy_score(y_test, yte)

                train_prec = precision_score(y_train, ytr, average="weighted", zero_division=0)
                test_prec = precision_score(y_test, yte, average="weighted", zero_division=0)

                train_rec = recall_score(y_train, ytr, average="weighted", zero_division=0)
                test_rec = recall_score(y_test, yte, average="weighted", zero_division=0)

                train_f1 = f1_score(y_train, ytr, average="weighted", zero_division=0)
                test_f1 = f1_score(y_test, yte, average="weighted", zero_division=0)

                # MLflow logging
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("train_precision", train_prec)
                mlflow.log_metric("test_precision", test_prec)
                mlflow.log_metric("train_recall", train_rec)
                mlflow.log_metric("test_recall", test_rec)
                mlflow.log_metric("train_f1", train_f1)
                mlflow.log_metric("test_f1", test_f1)
                mlflow.log_metric("train_time", train_time)

                rows.append({
                    "Model": name,
                    "Train_Accuracy": train_acc,
                    "Test_Accuracy": test_acc,
                    "Train_Precision": train_prec,
                    "Test_Precision": test_prec,
                    "Train_Recall": train_rec,
                    "Test_Recall": test_rec,
                    "Train_F1": train_f1,
                    "Test_F1": test_f1,
                    "Train_Time(s)": train_time
                })

    return pd.DataFrame(rows)

def plot_model_comparison(df, task):
    if task == "regression":
        train_metric_col = "Train_R2"
        test_metric_col = "Test_R2"
    else:
        train_metric_col = "Train_Accuracy"
        test_metric_col = "Test_Accuracy"

    df.set_index("Model")[[train_metric_col, test_metric_col]].plot.bar()
    plt.title(f"Model Comparison - {task.capitalize()} Metrics")
    plt.show()

def explain_model(model, X):
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="interventional"
    )

    shap_values = explainer(
        X,
        check_additivity=False  
    )

    shap.summary_plot(shap_values, X)

def predict_new_row(row_df):
    encoders = joblib.load("../models/encoder.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    selected_cols = joblib.load("../models/selected_features.pkl")
    model = joblib.load("../models/RandomForest.pkl")

    for col, enc in encoders.items():
        if col in row_df.columns:
            row_df[col] = enc.transform(row_df[col])

    row_df = row_df[selected_cols]
    row_df[:] = scaler.transform(row_df)

    return model.predict(row_df)


def run_pipeline(csv_path, target, task):
    df = pd.read_csv(csv_path)

    df = remove_duplicate_rows(df)
    df = handle_null_values(df)

    # IMPORTANT: For classification, ensure target is integer type
    if task == "classification":
        df[target] = df[target].astype(int)

    df = encode_features(df)
    df = scale_features(df)

    X = df.drop(columns=[target])
    y = df[target]
    
    # CRITICAL: For classification, y must be integer, not float
    if task == "classification":
        y = y.astype(int)

    X = select_features(X, y, task)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models(task)
    results = evaluate_models(models, X_train, X_test, y_train, y_test, task)

    print(results)
    plot_model_comparison(results, task)

    if "RandomForest" in models:
        model_to_explain = models["RandomForest"]
    elif "GradientBoosting" in models:
        model_to_explain = models["GradientBoosting"]
    else:
        model_to_explain = models[list(models.keys())[0]]
        print(f"Warning: Using {list(models.keys())[0]} for explanation. SHAP TreeExplainer might not support it.")

    explain_model(model_to_explain, X_train)


def run_regression(csv_path, target):
    run_pipeline(csv_path, target, task="regression")

def run_classification(csv_path, target):
    run_pipeline(csv_path, target, task="classification")


def model_ml(df, target, task):
    # df = pd.read_csv("C:/Users/sivan/OneDrive/Desktop/AutoML/backend/output/df.csv")
    print(df.columns)
    print(df.dtypes)


    # csv_path = "C:/Users/sivan/OneDrive/Desktop/AutoML/backend/output/df.csv"   # csv path (preprocessed)
    # target = "origin"
    # task = "classification"


    if task == "regression":
        run_regression(csv_path, target)
    elif task == "classification":
        run_classification(csv_path, target)
    else:
        print("Invalid task specified. Please choose 'regression' or 'classification'.")


