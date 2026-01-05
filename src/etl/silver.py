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

    df_new = spark.read.json(PATH_BRONZE_FUND)
    df_new = df_new.withColumn("effective_date", lit(TODAY_STR).cast("date"))

    try:
        df_existing = spark.read.parquet(PATH_SILVER_FUND)
    except:
        df_existing = None

    if df_existing:
        df_to_expire = df_existing.alias("old") \
            .join(df_new.alias("new"),
                  (col("old.symbol") == col("new.symbol")) &
                  (col("old.is_current") == True) &
                  ((col("old.pe_ratio") != col("new.pe_ratio")) |
                   (col("old.debt_to_equity") != col("new.debt_to_equity"))),
                  "inner") \
            .select("old.*") \
            .withColumn("is_current", lit(False)) \
            .withColumn("expiration_date", lit(TODAY_STR).cast("date"))

        df_unchanged = df_existing.join(
            df_to_expire,
            ["symbol", "effective_date"],
            "left_anti"
        )

        df_new_flagged = df_new \
            .withColumn("is_current", lit(True)) \
            .withColumn("expiration_date", lit(None).cast("date"))

        df_final = df_unchanged.union(df_to_expire).union(df_new_flagged)
    else:
        df_final = df_new \
            .withColumn("is_current", lit(True)) \
            .withColumn("expiration_date", lit(None).cast("date"))

    df_final.write.mode("overwrite").parquet(PATH_SILVER_FUND)

    print(f"[SUCCESS] Silver Fundamentals mis à jour.")


if __name__ == "__main__":
    process_prices_timeseries()
    process_fundamentals_scd2()