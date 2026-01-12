# src/train_sparkml.py

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pyspark.ml import Pipeline
from pyspark.ml.feature import FeatureHasher, VectorAssembler, Imputer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark

# ✅ REQUIRED in scripts (notebooks create this automatically)
spark = SparkSession.builder.getOrCreate()

LABEL_COL = "Purchase"

CATEGORICAL_COLS = ["Gender", "Age", "City_Category", "Stay_In_Current_City_Years"]
NUMERIC_COLS = [
    "Occupation",
    "Marital_Status",
    "Product_Category_1",
    "Product_Category_2",
    "Product_Category_3",
]

REQUIRED_COLS = CATEGORICAL_COLS + NUMERIC_COLS + [LABEL_COL]


def load_and_clean(table_name: str):
    """
    Loads a UC table and:
    - safely casts numeric columns using try_cast (handles 'null' strings)
    - selects ONLY the required columns to prevent high-cardinality leakage (e.g., Product_ID)
    """
    df = spark.table(table_name)

    # Safely cast numeric columns (handles "null" strings by returning NULL)
    for c in NUMERIC_COLS + [LABEL_COL]:
        if c in df.columns:
            df = df.withColumn(c, expr(f"try_cast({c} as double)"))

    # HARD SELECT: prevents User_ID/Product_ID or any other extra columns from entering pipeline
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {missing}")

    df = df.select(*REQUIRED_COLS)
    return df


def build_pipeline(label_col: str = LABEL_COL):
    """
    Spark-Connect-friendly pipeline:
    - Imputer for numeric columns
    - FeatureHasher for categoricals (NO label dictionary stored -> small model)
    - VectorAssembler to combine hashed + numeric
    - LinearRegression
    """
    imputer = Imputer(
        inputCols=NUMERIC_COLS,
        outputCols=[f"{c}_imp" for c in NUMERIC_COLS],
    )

    # Hash categorical columns into a fixed-size feature vector
    # Increase HASH_DIM if you want fewer collisions (2^18 is a good default)
    hash_dim = int(os.getenv("HASH_DIM", str(2**18)))  # 262,144
    hasher = FeatureHasher(
        inputCols=CATEGORICAL_COLS,
        outputCol="cat_hashed",
        numFeatures=hash_dim,
    )

    assembler = VectorAssembler(
        inputCols=["cat_hashed"] + [f"{c}_imp" for c in NUMERIC_COLS],
        outputCol="features",
    )

    lr = LinearRegression(
        labelCol=label_col,
        featuresCol="features",
        regParam=float(os.getenv("REG_PARAM", "0.1")),
        elasticNetParam=float(os.getenv("ELASTIC_NET", "0.0")),  # 0=ridge, 1=lasso
        maxIter=int(os.getenv("MAX_ITER", "50")),
    )

    return Pipeline(stages=[imputer, hasher, assembler, lr])


def main():
    # ---- Config (override via env vars later for jobs/CI) ----
    train_table = os.getenv("TRAIN_TABLE", "main.default.train")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "/Shared/black-friday-mlops")
    model_name = os.getenv("REGISTERED_MODEL", "main.default.black_friday_purchase_model")

    # Required for serverless/shared clusters: point MLflow temp dir to UC Volume
    # Create volume once: CREATE VOLUME IF NOT EXISTS main.default.mlflow_tmp;
    os.environ.setdefault("MLFLOW_DFS_TMP", "/Volumes/main/default/mlflow_tmp")

    mlflow.set_experiment(experiment_name)

    df = load_and_clean(train_table)
    print("Training columns:", df.columns)

    train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)

    pipeline = build_pipeline(label_col=LABEL_COL)

    with mlflow.start_run():
        model = pipeline.fit(train_df)
        preds = model.transform(val_df)

        rmse = RegressionEvaluator(
            labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse"
        ).evaluate(preds)
        r2 = RegressionEvaluator(
            labelCol=LABEL_COL, predictionCol="prediction", metricName="r2"
        ).evaluate(preds)
        mae = RegressionEvaluator(
            labelCol=LABEL_COL, predictionCol="prediction", metricName="mae"
        ).evaluate(preds)

        mlflow.log_param("model_type", "LinearRegression+FeatureHasher")
        mlflow.log_param("hash_dim", int(os.getenv("HASH_DIM", str(2**18))))
        mlflow.log_param("reg_param", float(os.getenv("REG_PARAM", "0.1")))
        mlflow.log_param("elastic_net", float(os.getenv("ELASTIC_NET", "0.0")))
        mlflow.log_param("max_iter", int(os.getenv("MAX_ITER", "50")))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model (should be Spark-Connect-safe now)
        mlflow.spark.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        # Register to UC Model Registry (if enabled)
        try:
            mlflow.register_model(model_uri, model_name)
            print(f"Registered model to: {model_name}")
        except Exception as e:
            print("Model registration skipped (may not be enabled). Error:")
            print(e)

        print(f"Done. Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")


if __name__ == "__main__":
    main()
