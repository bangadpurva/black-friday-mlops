# src/monitoring.py

import datetime as dt
from typing import Optional, Dict, Any

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, min as spark_min, max as spark_max, lit

spark = SparkSession.builder.getOrCreate()


def compute_prediction_stats(pred_table: str) -> Dict[str, Any]:
    """
    Compute basic prediction stats from a UC table that contains a 'prediction' column.
    """
    df = spark.table(pred_table)

    if "prediction" not in df.columns:
        raise ValueError(f"'{pred_table}' does not have a 'prediction' column. Columns: {df.columns}")

    agg = (
        df.select(col("prediction").cast("double").alias("prediction"))
          .agg(
              mean("prediction").alias("pred_mean"),
              stddev("prediction").alias("pred_stddev"),
              spark_min("prediction").alias("pred_min"),
              spark_max("prediction").alias("pred_max"),
          )
          .collect()[0]
    )

    row_count = df.count()

    return {
        "row_count": int(row_count),
        "pred_mean": float(agg["pred_mean"]) if agg["pred_mean"] is not None else None,
        "pred_stddev": float(agg["pred_stddev"]) if agg["pred_stddev"] is not None else None,
        "pred_min": float(agg["pred_min"]) if agg["pred_min"] is not None else None,
        "pred_max": float(agg["pred_max"]) if agg["pred_max"] is not None else None,
    }


def log_monitoring_row(
    monitoring_table: str,
    *,
    mlflow_run_id: str,
    test_table: str,
    pred_table: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a single monitoring row into a UC Delta table (creates it if missing).
    """
    stats = compute_prediction_stats(pred_table)

    now_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    row = {
        "timestamp_utc": now_utc,
        "mlflow_run_id": mlflow_run_id,
        "test_table": test_table,
        "pred_table": pred_table,
        **stats,
    }

    if extra:
        row.update(extra)

    sdf = spark.createDataFrame([row])

    (sdf.write
        .mode("append")
        .format("delta")
        .saveAsTable(monitoring_table)
    )
