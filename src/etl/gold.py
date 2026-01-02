import os
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, avg, stddev, lag, when, abs as spark_abs,
    lead, coalesce, lit
)

# --- CONFIGURATION ---
BUCKET_NAME = os.getenv("BUCKET_NAME", "finance_datalake")
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_key.json")
JAR_PATH = "/tmp/gcs-connector.jar"

spark = SparkSession.builder \
    .appName("Finance_Gold_Advanced_Features") \
    .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
    .config("spark.jars", JAR_PATH) \
    .config("spark.driver.extraClassPath", JAR_PATH) \
    .config("spark.executor.extraClassPath", JAR_PATH) \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
    .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", KEY_PATH) \
    .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") \
    .getOrCreate()


def calculate_technical_indicators(df):
    w_spec = Window.partitionBy("symbol").orderBy("trade_date")

    w_20 = w_spec.rowsBetween(-19, 0)
    df = df.withColumn("sma_20", avg("close").over(w_20)) \
        .withColumn("stddev_20", stddev("close").over(w_20)) \
        .withColumn("bollinger_upper", col("sma_20") + (col("stddev_20") * 2)) \
        .withColumn("bollinger_lower", col("sma_20") - (col("stddev_20") * 2))

    w_12 = w_spec.rowsBetween(-11, 0)
    w_26 = w_spec.rowsBetween(-25, 0)
    df = df.withColumn("ema_12", avg("close").over(w_12)) \
        .withColumn("ema_26", avg("close").over(w_26)) \
        .withColumn("macd_line", col("ema_12") - col("ema_26"))

    df = df.withColumn("diff", col("close") - lag("close", 1).over(w_spec))
    df = df.withColumn("gain", when(col("diff") > 0, col("diff")).otherwise(0)) \
        .withColumn("loss", when(col("diff") < 0, spark_abs(col("diff"))).otherwise(0))

    w_14 = w_spec.rowsBetween(-13, 0)
    df = df.withColumn("avg_gain", avg("gain").over(w_14)) \
        .withColumn("avg_loss", avg("loss").over(w_14))

    df = df.withColumn("rs", col("avg_gain") / when(col("avg_loss") == 0, 1).otherwise(col("avg_loss"))) \
        .withColumn("rsi_14", 100 - (100 / (1 + col("rs"))))

    return df


def create_advanced_gold():
    print("--- Début Gold Layer ---")

    try:
        df_prices = spark.read.parquet(f"gs://{BUCKET_NAME}/silver/prices/")
        df_fund = spark.read.parquet(f"gs://{BUCKET_NAME}/silver/fundamentals/")

        print(f"DEBUG: Lignes Prix trouvées : {df_prices.count()}")
        print(f"DEBUG: Lignes Fondamentaux trouvées : {df_fund.count()}")

    except Exception as e:
        print(f"[ERREUR] Impossible de lire Silver : {e}")
        return

    df_fund_clean = df_fund.withColumn("end_date_filled", coalesce(col("end_date"), lit("2099-12-31").cast("date")))

    cond = [
        df_prices.symbol == df_fund_clean.symbol,
        df_prices.trade_date >= df_fund_clean.start_date,
        df_prices.trade_date < df_fund_clean.end_date_filled
    ]

    df_joined = df_prices.join(df_fund_clean, cond, "left").drop(df_fund_clean.symbol)
    print(f"DEBUG: Lignes après Jointure : {df_joined.count()}")

    df_indicators = calculate_technical_indicators(df_joined)

    w_lead = Window.partitionBy("symbol").orderBy("trade_date")

    # Sélection
    df_final_temp = df_indicators.select(
        "symbol", "trade_date", "close", "volume",
        col("rsi_14"),
        col("macd_line"),
        ((col("close") - col("bollinger_lower")) / (col("bollinger_upper") - col("bollinger_lower"))).alias("pct_b"),
        col("pe_ratio"), col("debt_to_equity"),
        ((lead("close", 1).over(w_lead) - col("close")) / col("close")).alias("target_return_next_day")
    )

    df_filled = df_final_temp.na.fill(0, subset=["pe_ratio", "debt_to_equity", "rsi_14", "macd_line", "pct_b"])

    df_final = df_filled.dropna(subset=["trade_date", "close"])

    count_final = df_final.count()
    print(f"DEBUG: Lignes FINALES à écrire : {count_final}")

    if count_final == 0:
        print("[ALERTE] Le DataFrame final est vide ! Vérifie les dates de jointure ou les données Silver.")
    else:
        output_path = f"gs://{BUCKET_NAME}/gold/advanced_features/"
        print(f"Écriture Gold dans : {output_path}")
        df_final.write.mode("overwrite").partitionBy("symbol").parquet(output_path)
        print("Gold Advanced terminé avec succès.")


if __name__ == "__main__":
    create_advanced_gold()