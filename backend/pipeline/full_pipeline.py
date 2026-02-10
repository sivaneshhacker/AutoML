import pandas as pd


from pipeline.preprocessing_ts import preprocessing_pipeline, dataframe_details, preprocessing__ts
from pipeline.preprocessing_ml import preprocessing_pipeline_ml, dataframe_details_ml, preprocessing_ml

from pipeline.eda_ts import comprehensive_eda, eda_ts
from pipeline.eda_ml import comprehensive_eda_ml

from pipeline.model_ts import time_series_regression_pipeline, time_series_classification_pipeline, model_ts
from pipeline.model_ml import run_regression, run_classification, model_ml

from pipeline.unsupervised import UnsupervisedMLPipeline, AutoEDA
from pipeline.preprocessing_trial import AutoPreprocessor

def run_complete_pipeline(df, target, task):
    # csv_path = "C:/Users/sivan/OneDrive/Desktop/AutoML/backend/" + file_path   # Preprocessed csv path
    # print(f"csvPath = {csv_path}")
    # df = pd.read_csv(csv_path)   # User uploaded
    # task = ""
    # target = ""
    if "date" in df.columns:
        preprocessing__ts(df, target)
    df = pd.read_csv("C:/Users/sivan/OneDrive/Desktop/AutoML/backend/output/df.csv",index_col=0)  # csv path (preprocessed)
    preprocessing_ml(df, target)
   
    eda_result = eda_ts(df, target)

    model_ts_results = model_ts(df, task, target)
    model_ml(df, target, task)




    if task in ("time_series_classification", "time_series_regression"):
        try:
            df = preprocessing_pipeline(df_input, target=target)

        except (ValueError, KeyError, TypeError) as e:
            print("\n[WARNING] Main preprocessing failed. Falling back to AutoPreprocessor.")
            print(f"Reason: {e}")

            preprocessor = AutoPreprocessor(
                scaling_method='standard',
                missing_strategy='auto',
                outlier_method='iqr',
                outlier_threshold=1.5
            )
                
            df = preprocessor.fit_transform(df_input)

            report = preprocessor.get_report()
            print("\nFallback Preprocessing Report:")
            print(report)

        
        (df_head, df_info, df_describe, df_nunique, df_duplicate_sum, df_null_sum, object_columns, 
        num_columns, date_columns, bool_columns, category_columns) = dataframe_details(df)

        result = comprehensive_eda(df, target_col = target)

        if task == "time_series_regression":
            results = time_series_regression_pipeline(df, target_col=target)

        elif task == "time_series_classification":
            results = time_series_classification_pipeline(df, target_col=target)


    elif task in ("classification", "regression"):
        try:
            df = preprocessing_pipeline_ml(df_input, target=target)

        except (ValueError, KeyError, TypeError) as e:
            print("\n[WARNING] Main preprocessing failed. Falling back to AutoPreprocessor.")
            print(f"Reason: {e}")

            preprocessor = AutoPreprocessor(
                scaling_method='standard',
                missing_strategy='auto',
                outlier_method='iqr',
                outlier_threshold=1.5
            )
                
            df = preprocessor.fit_transform(df_input)

            report = preprocessor.get_report()
            print("\nFallback Preprocessing Report:")
            print(report)
        

        (df_head, df_info, df_describe, df_nunique, df_duplicate_sum, df_null_sum, object_columns, 
        num_columns, date_columns, bool_columns, category_columns) = dataframe_details_ml(df)

        comprehensive_eda_ml(df, target, task)

        if task == "regression":
            run_regression(csv_path, target)
        
        elif task == "classification":
            run_classification(csv_path, target)


    elif task == "unsupervised":
        try:
            df = preprocessing_pipeline_ml(df_input, target=target)

        except (ValueError, KeyError, TypeError) as e:
            print("\n[WARNING] Main preprocessing failed. Falling back to AutoPreprocessor.")
            print(f"Reason: {e}")

            preprocessor = AutoPreprocessor(
                scaling_method='standard',
                missing_strategy='auto',
                outlier_method='iqr',
                outlier_threshold=1.5
            )
                
            df = preprocessor.fit_transform(df_input)

            report = preprocessor.get_report()
            print("\nFallback Preprocessing Report:")
            print(report)


        unsupervised_eda = AutoEDA(df)

        unsupervised_pipeline = UnsupervisedMLPipeline(df, n_clusters_range=(2, 8))

        results = unsupervised_pipeline.run_pipeline()

    else:
        raise ValueError("Task need to be correct")









