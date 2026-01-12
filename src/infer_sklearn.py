# src/infer_sklearn.py
import monitoring

import os
import mlflow
import mlflow.sklearn
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, monotonically_increasing_id

spark = SparkSession.builder.getOrCreate()

CATEGORICAL_COLS = ["Gender", "Age", "City_Category", "Stay_In_Current_City_Years"]
NUMERIC_COLS = [
    "Occupation",
    "Marital_Status",
    "Product_Category_1",
    "Product_Category_2",
    "Product_Category_3",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def load_model_from_latest_run(experiment_name: str):
    """
    Loads the model from the latest MLflow run in the given experiment.
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No MLflow runs found in experiment: {experiment_name}")

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model, run_id


def load_and_clean_test(table_name: str, sample_n: int | None = None) -> pd.DataFrame:
    """
    Loads UC test table, safely casts numeric columns, returns pandas DF of features.
    """
    df = spark.table(table_name)

    for c in NUMERIC_COLS:
        if c in df.columns:
            df = df.withColumn(c, expr(f"try_cast({c} as double)"))

    # Keep IDs if present so we can join predictions back
    id_cols = [c for c in ["User_ID", "Product_ID"] if c in df.columns]

    needed = id_cols + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {missing}")

    df = df.select(*needed)

    # Optional sampling
    if sample_n is not None and sample_n > 0:
        df = df.limit(sample_n)

    pdf = df.toPandas()
    return pdf


def write_predictions_to_uc(pred_pdf: pd.DataFrame, output_table: str):
    """
    Writes predictions to a UC table.
    """
    pred_sdf = spark.createDataFrame(pred_pdf)
    (pred_sdf.write
        .mode("overwrite")
        .format("delta")
        .saveAsTable(output_table)
    )


def main():
    test_table = os.getenv("TEST_TABLE", "main.default.test")
    output_table = os.getenv("PRED_TABLE", "main.default.black_friday_predictions")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "/Shared/black-friday-mlops")

    sample_n_env = os.getenv("SAMPLE_N", "0").strip()
    sample_n = int(sample_n_env) if sample_n_env.isdigit() else 0

    model, run_id = load_model_from_latest_run(experiment_name)
    print(f"Loaded model from run_id: {run_id}")

    test_pdf = load_and_clean_test(test_table, sample_n=sample_n if sample_n > 0 else None)

    # Build X for inference
    X = test_pdf[FEATURE_COLS]
    preds = model.predict(X)

    # Attach predictions
    out = test_pdf.copy()
    out["prediction"] = preds

    # Write predictions to UC
    write_predictions_to_uc(out, output_table)
    print(f"Wrote predictions to UC table: {output_table}")
    
    monitoring_table = os.getenv("MONITOR_TABLE", "main.default.black_friday_monitoring")

    monitoring.log_monitoring_row(
        monitoring_table,
        mlflow_run_id=run_id,
        test_table=test_table,
        pred_table=output_table,
    )
    print(f"Appended monitoring row to: {monitoring_table}")

    print("Preview rows:")
    print(out.head(5))


if __name__ == "__main__":
    main()
