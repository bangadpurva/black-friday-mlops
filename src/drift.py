# src/drift.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()


def get_latest_and_baseline(monitor_table: str, baseline_mode: str = "first"):
    """
    Returns (latest_row, baseline_row) as dicts.
    baseline_mode:
      - "first": earliest row in table
      - "previous": row immediately before latest
    """
    df = spark.table(monitor_table)

    w_desc = Window.orderBy(col("timestamp_utc").desc())
    latest = df.withColumn("rn", row_number().over(w_desc)).filter(col("rn") == 1).drop("rn")

    if baseline_mode == "previous":
        baseline = df.withColumn("rn", row_number().over(w_desc)).filter(col("rn") == 2).drop("rn")
    else:
        w_asc = Window.orderBy(col("timestamp_utc").asc())
        baseline = df.withColumn("rn", row_number().over(w_asc)).filter(col("rn") == 1).drop("rn")

    latest_rows = latest.collect()
    base_rows = baseline.collect()

    if not latest_rows or not base_rows:
        return None, None

    return latest_rows[0].asDict(), base_rows[0].asDict()


def drift_check_mean_std(
    monitor_table: str,
    *,
    baseline_mode: str = "first",
    max_mean_pct_change: float = 0.15,   # 15%
    max_std_pct_change: float = 0.25     # 25%
) -> dict:
    """
    Returns drift report dict; raises if drift exceeds thresholds.
    """
    latest, baseline = get_latest_and_baseline(monitor_table, baseline_mode=baseline_mode)

    # Not enough history to compare
    if latest is None or baseline is None:
        return {"status": "skipped", "reason": "insufficient_history"}

    def pct_change(new, old):
        if new is None or old is None or old == 0:
            return None
        return (new - old) / old

    mean_change = pct_change(latest.get("pred_mean"), baseline.get("pred_mean"))
    std_change = pct_change(latest.get("pred_stddev"), baseline.get("pred_stddev"))

    report = {
        "status": "ok",
        "baseline_mode": baseline_mode,
        "latest_timestamp": latest.get("timestamp_utc"),
        "baseline_timestamp": baseline.get("timestamp_utc"),
        "latest_pred_mean": latest.get("pred_mean"),
        "baseline_pred_mean": baseline.get("pred_mean"),
        "mean_pct_change": mean_change,
        "latest_pred_stddev": latest.get("pred_stddev"),
        "baseline_pred_stddev": baseline.get("pred_stddev"),
        "std_pct_change": std_change,
    }

    # Decide drift
    drifted = False
    reasons = []

    if mean_change is not None and abs(mean_change) > max_mean_pct_change:
        drifted = True
        reasons.append(f"mean_pct_change={mean_change:.3f} exceeds {max_mean_pct_change:.3f}")

    if std_change is not None and abs(std_change) > max_std_pct_change:
        drifted = True
        reasons.append(f"std_pct_change={std_change:.3f} exceeds {max_std_pct_change:.3f}")

    if drifted:
        report["status"] = "drift_detected"
        report["reasons"] = reasons
        raise RuntimeError(f"DRIFT DETECTED: {', '.join(reasons)}")

    return report
