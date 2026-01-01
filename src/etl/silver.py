from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, to_date, current_timestamp,
    input_file_name, lit
)
from datetime import datetime
import os

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
    .appName("Finance_Bronze_To_Silver_ETL") \
    .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
    .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
    .config("spark.jars", JAR_PATH) \
    .config("spark.driver.extraClassPath", JAR_PATH) \
    .config("spark.executor.extraClassPath", JAR_PATH) \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
    .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", KEY_PATH) \
    .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") \
    .getOrCreate()


def union_dataframes_safe(df1, df2):
    """
    Fusionne deux DataFrames manuellement.
    Remplace allowMissingColumns=True qui n'existe pas sur les vieilles versions Spark.
    """
    if df1 is None: return df2
    if df2 is None: return df1

    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    all_cols = sorted(list(cols1.union(cols2)))

    def add_missing(df, ref_cols):
        for c in all_cols:
            if c not in ref_cols:
                df = df.withColumn(c, lit(None))
        return df.select(all_cols)

    df1_aligned = add_missing(df1, cols1)
    df2_aligned = add_missing(df2, cols2)

    return df1_aligned.unionByName(df2_aligned)


def process_prices_timeseries():
    print(f"--- {TODAY_STR} - Processing PRICES ---")
    try:
        df_raw = spark.read.json(PATH_BRONZE_PRICES)
    except Exception as e:
        print(f"[WARN] Pas de données de PRIX pour {TODAY_STR}.")
        return

    df_silver = df_raw.select(
        col("meta.symbol").alias("symbol"),
        col("meta.interval").alias("interval"),
        explode(col("values")).alias("data"),
        input_file_name().alias("source_file")
    ).select(
        col("symbol"),
        to_date(col("data.datetime")).alias("trade_date"),
        col("data.open").cast("double"),
        col("data.high").cast("double"),
        col("data.low").cast("double"),
        col("data.close").cast("double"),
        col("data.volume").cast("long"),
        current_timestamp().alias("ingestion_ts")
    )

    # Optimisation écriture
    df_silver.dropDuplicates(["symbol", "trade_date"]) \
        .write.mode("append").partitionBy("trade_date").parquet(PATH_SILVER_PRICES)
    print(f"[SUCCESS] Silver Prices mis à jour.")


def process_fundamentals_scd2():
    print(f"--- {TODAY_STR} - Processing FUNDAMENTALS (SCD2) ---")
    try:
        df_new_raw = spark.read.json(PATH_BRONZE_FUND)
    except Exception as e:
        print(f"[WARN] Pas de données FONDAMENTALES pour {TODAY_STR}.")
        return

    required_cols = ["peRatioTTM", "debtToEquityTTM", "currentRatioTTM"]
    for c in required_cols:
        if c not in df_new_raw.columns:
            print(f"[WARN] Colonne '{c}' manquante. Ajout de NULL.")
            df_new_raw = df_new_raw.withColumn(c, lit(None))

    df_new = df_new_raw.select(
        col("symbol"),
        lit(TODAY_STR).cast("date").alias("report_date"),
        col("peRatioTTM").cast("double").alias("pe_ratio"),
        col("debtToEquityTTM").cast("double").alias("debt_to_equity"),
        col("currentRatioTTM").cast("double").alias("current_ratio"),
        lit(TODAY_STR).cast("date").alias("effective_date")
    )

    try:
        df_history = spark.read.parquet(PATH_SILVER_FUND)
        df_active = df_history.filter(col("is_current") == True)
        df_closed = df_history.filter(col("is_current") == False)
    except:
        print("[INFO] Premier run: Création Silver.")
        df_history, df_active, df_closed = None, None, None

    if df_active is None:
        df_final = df_new.withColumn("start_date", col("effective_date")) \
            .withColumn("end_date", lit(None).cast("date")) \
            .withColumn("is_current", lit(True))
    else:
        df_active_renamed = df_active.select(
            col("symbol").alias("h_symbol"), col("pe_ratio").alias("h_pe_ratio"),
            col("debt_to_equity").alias("h_debt_to_equity"), col("current_ratio").alias("h_current_ratio"),
            col("start_date"), col("end_date"), col("is_current"), col("report_date").alias("h_report_date")
        )

        joined = df_new.join(df_active_renamed, df_new.symbol == df_active_renamed.h_symbol, "left_outer")

        changed = joined.filter((col("h_symbol").isNotNull()) & (col("pe_ratio") != col("h_pe_ratio"))).select(
            col("h_symbol").alias("symbol"), col("start_date"), col("effective_date").alias("end_date"),
            lit(False).alias("is_current"), col("h_pe_ratio").alias("pe_ratio"),
            col("h_debt_to_equity").alias("debt_to_equity"), col("h_current_ratio").alias("current_ratio"),
            col("h_report_date").alias("report_date")
        )

        new_recs = joined.select(
            col("symbol"), col("effective_date").alias("start_date"), lit(None).cast("date").alias("end_date"),
            lit(True).alias("is_current"), col("pe_ratio"), col("debt_to_equity"), col("current_ratio"),
            col("report_date")
        )

        unchanged = df_active.join(df_new, df_active.symbol == df_new.symbol, "left_anti")

        step1 = union_dataframes_safe(df_closed, changed)
        step2 = union_dataframes_safe(step1, new_recs)
        df_final = union_dataframes_safe(step2, unchanged)

    print(f"[SUCCESS] Écriture SCD2 dans {PATH_SILVER_FUND}")
    df_final.write.mode("overwrite").parquet(PATH_SILVER_FUND)


if __name__ == "__main__":
    process_prices_timeseries()
    process_fundamentals_scd2()