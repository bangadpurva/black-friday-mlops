# src/train_sparkml.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor

import mlflow
import mlflow.spark

spark = SparkSession.builder.getOrCreate()

def load_and_clean(table_name: str):
    df = spark.table(table_name)

    # Safely cast numeric columns (handles "null" strings by returning NULL)
    numeric_like = [
        "Occupation",
        "Marital_Status",
        "Product_Category_1",
        "Product_Category_2",
        "Product_Category_3",
        "Purchase",
    ]
    for c in numeric_like:
        if c in df.columns:
            df = df.withColumn(c, expr(f"try_cast({c} as double)"))

    # Drop IDs (optional, but usually improves generalization)
    for c in ["User_ID", "Product_ID"]:
        if c in df.columns:
            df = df.drop(c)

    return df


def build_pipeline(label_col: str = "Purchase"):
    categorical_cols = ["Gender", "Age", "City_Category", "Stay_In_Current_City_Years"]
    numeric_cols = [
        "Occupation",
        "Marital_Status",
        "Product_Category_1",
        "Product_Category_2",
        "Product_Category_3",
    ]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in categorical_cols
    ]

    imputer = Imputer(
        inputCols=numeric_cols,
        outputCols=[f"{c}_imp" for c in numeric_cols],
    )

    assembler = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in categorical_cols]
        + [f"{c}_imp" for c in numeric_cols],
        outputCol="features",
    )

    # rf = RandomForestRegressor(
    #     labelCol=label_col,
    #     featuresCol="features",
    #     numTrees=int(os.getenv("NUM_TREES", "200")),
    #     maxDepth=int(os.getenv("MAX_DEPTH", "12")),
    #     seed=42,
    # )
    gbt = GBTRegressor(
        labelCol=label_col,
        featuresCol="features",
        maxIter=int(os.getenv("MAX_ITER", "50")),   # number of boosting iterations
        maxDepth=int(os.getenv("MAX_DEPTH", "5")),  # keep small for serverless
        stepSize=float(os.getenv("STEP_SIZE", "0.1")),
        seed=42,
    )

return Pipeline(stages=indexers + encoders + [imputer, assembler, gbt])

    return Pipeline(stages=indexers + encoders + [imputer, assembler, rf])


def main():
    # ---- Config (override via env vars later for jobs/CI) ----
    train_table = os.getenv("TRAIN_TABLE", "main.default.train")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "/Shared/black-friday-mlops")
    model_name = os.getenv("REGISTERED_MODEL", "main.default.black_friday_purchase_model")

    # Required for serverless/shared clusters: point MLflow temp dir to UC Volume
    # Create volume first: main.default.mlflow_tmp
    os.environ.setdefault("MLFLOW_DFS_TMP", "/Volumes/main/default/mlflow_tmp")

    mlflow.set_experiment(experiment_name)

    df = load_and_clean(train_table)
    train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)

    pipeline = build_pipeline(label_col="Purchase")

    with mlflow.start_run():
        model = pipeline.fit(train_df)
        preds = model.transform(val_df)

        rmse = RegressionEvaluator(
            labelCol="Purchase", predictionCol="prediction", metricName="rmse"
        ).evaluate(preds)
        r2 = RegressionEvaluator(
            labelCol="Purchase", predictionCol="prediction", metricName="r2"
        ).evaluate(preds)
        mae = RegressionEvaluator(
            labelCol="Purchase", predictionCol="prediction", metricName="mae"
        ).evaluate(preds)

        # mlflow.log_param("model_type", "RandomForestRegressor")
        # mlflow.log_param("num_trees", int(os.getenv("NUM_TREES", "200")))
        # mlflow.log_param("max_depth", int(os.getenv("MAX_DEPTH", "12")))
        # mlflow.log_metric("rmse", rmse)
        # mlflow.log_metric("mae", mae)
        # mlflow.log_metric("r2", r2)
        mlflow.log_param("model_type", "GBTRegressor")
        mlflow.log_param("max_iter", int(os.getenv("MAX_ITER", "50")))
        mlflow.log_param("max_depth", int(os.getenv("MAX_DEPTH", "5")))
        mlflow.log_param("step_size", float(os.getenv("STEP_SIZE", "0.1")))

        mlflow.spark.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        # Register to UC Model Registry (if enabled in your workspace)
        try:
            mlflow.register_model(model_uri, model_name)
            print(f"Registered model to: {model_name}")
        except Exception as e:
            print("Model registration skipped (may not be enabled). Error:")
            print(e)

        print(f"Done. Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")


if __name__ == "__main__":
    main()
