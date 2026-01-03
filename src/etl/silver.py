import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, to_date, current_timestamp, lit
)
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from datetime import datetime

# --- CONFIGURATION ---
BUCKET_NAME = os.getenv("BUCKET_NAME", "finance_datalake")
TODAY_STR = datetime.now().strftime("%Y-%m-%d")

KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_key.json")
JAR_PATH = "/tmp/gcs-connector.jar"

PATH_BRONZE_PRICES = f"gs://{BUCKET_NAME}/bronze/twelvedata/prices/dt={TODAY_STR}/"
PATH_SILVER_PRICES = f"gs://{BUCKET_NAME}/silver/prices/"

PATH_BRONZE_FUND = f"gs://{BUCKET_NAME}/bronze/fmp/ratios/dt={TODAY_STR}/"
PATH_SILVER_FUND = f"gs://{BUCKET_NAME}/silver/fundamentals/"

spark = SparkSession.builder \
    .appName("Finance_Bronze_To_Silver_ETL_Strict") \
    .config("spark.jars", JAR_PATH) \
    .config("spark.driver.extraClassPath", JAR_PATH) \
    .config("spark.executor.extraClassPath", JAR_PATH) \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
    .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", KEY_PATH) \
    .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
    .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") \
    .getOrCreate()


def process_prices_timeseries():
    print(f"--- Traitement prix (Schéma Strict & Overwrite) ---")

    price_schema = StructType([
        StructField("meta", StructType([
            StructField("symbol", StringType(), True),
            StructField("interval", StringType(), True)
        ]), True),
        StructField("values", ArrayType(StructType([
            StructField("datetime", StringType(), True),
            StructField("open", StringType(), True),
            StructField("high", StringType(), True),
            StructField("low", StringType(), True),
            StructField("close", StringType(), True),
            StructField("volume", StringType(), True)
        ])), True)
    ])

    try:
        df_raw = spark.read.schema(price_schema).json(PATH_BRONZE_PRICES)
    except Exception as e:
        print(f"[WARN] Erreur lecture ou pas de données PRIX : {e}")
        return

    if df_raw.count() == 0:
        print("[WARN] Fichiers JSON trouvés mais vides.")
        return

    df_silver = df_raw.select(
        col("meta.symbol").alias("symbol"),
        explode(col("values")).alias("data")
    ).select(
        col("symbol"),
        to_date(col("data.datetime")).alias("trade_date"),
        col("data.open").cast("double").alias("open"),
        col("data.high").cast("double").alias("high"),
        col("data.low").cast("double").alias("low"),
        col("data.close").cast("double").alias("close"),
        col("data.volume").cast("long").alias("volume"),

        current_timestamp().alias("ingestion_ts")
    )

    print(f"[INFO] Écriture (OVERWRITE) de Silver Prices dans : {PATH_SILVER_PRICES}")

    df_final_to_write = df_silver.dropna(subset=["trade_date", "close", "volume"])
    df_final_to_write.write.mode("overwrite").partitionBy("trade_date").parquet(PATH_SILVER_PRICES)

def process_fundamentals_scd2():
    print(f"--- Traitement des fondamentaux ---")
    try:
        df_raw = spark.read.json(PATH_BRONZE_FUND)
        if df_raw.count() == 0: return
    except Exception:
        print("[INFO] Pas de nouveaux fondamentaux.")
        return

    expected_cols = ["symbol", "date", "priceToEarningsRatio", "debtToEquityRatio"]
    for c in expected_cols:
        if c not in df_raw.columns:
            df_raw = df_raw.withColumn(c, lit(None))

    df_clean = df_raw.select(
        col("symbol"),
        col("date").cast("date").alias("report_date"),
        col("priceToEarningsRatio").cast("double").alias("pe_ratio"),
        col("debtToEquityRatio").cast("double").alias("debt_to_equity"),
        lit(TODAY_STR).cast("date").alias("effective_date")
    )

    print(f"[INFO] Ajout des fondamentaux dans : {PATH_SILVER_FUND}")
    df_clean.write.mode("overwrite").parquet(PATH_SILVER_FUND)
    print(f"[SUCCESS] Silver Fundamentals mis à jour.")


if __name__ == "__main__":
    process_prices_timeseries()
    process_fundamentals_scd2()