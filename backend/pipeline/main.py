# =====================================================================
# EXAMPLE USAGE
# =====================================================================
from full_pipeline import run_complete_pipeline

if __name__ == "__main__":
    # Example configuration
    csv_path = "/content/Automobile_data_kaggle.csv"  # Change this to your file path
    target = "price"     # Change this to your target column
    task = "regression"      # or "regression"
    
    # Run pipeline
    results = run_complete_pipeline(
        csv_path=csv_path,
        target=target,
        task=task,
        test_size=0.2,
        handle_outliers_flag=True,
        outlier_method="clip",
        handle_skewness_flag=True,
        handle_imbalance_flag=True,  # Set to True for classification with imbalanced classes
        imbalance_method="smote",
        remove_multicollinearity_flag=True,
        vif_threshold=10,
        n_features=10
    )