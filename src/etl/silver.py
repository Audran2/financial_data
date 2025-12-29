from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, to_date, current_timestamp,
    input_file_name, lit
)
from datetime import datetime

# --- CONFIGURATION ---
BUCKET_NAME = "finance_datalake"
TODAY_STR = datetime.now().strftime("%Y-%m-%d")

PATH_BRONZE_PRICES = f"gs://{BUCKET_NAME}/bronze/twelvedata/prices/dt={TODAY_STR}/"
PATH_SILVER_PRICES = f"gs://{BUCKET_NAME}/silver/prices/"

PATH_BRONZE_FUND = f"gs://{BUCKET_NAME}/bronze/fmp/ratios/dt={TODAY_STR}/"
PATH_SILVER_FUND = f"gs://{BUCKET_NAME}/silver/fundamentals/"

spark = SparkSession.builder \
    .appName("Finance_Bronze_To_Silver_ETL") \
    .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
    .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
    .getOrCreate()


def process_prices_timeseries():
    print(f"--- {TODAY_STR} - Processing PRICES ---")

    try:
        df_raw = spark.read.json(PATH_BRONZE_PRICES)
    except Exception as e:
        print(f"Pas de données de prix pour aujourd'hui (path: {PATH_BRONZE_PRICES}): {e}")
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

    df_dedup = df_silver.dropDuplicates(["symbol", "trade_date"])

    df_dedup.write \
        .mode("append") \
        .partitionBy("trade_date") \
        .parquet(PATH_SILVER_PRICES)

    print(f"Ecriture terminée pour Silver Prices dans {PATH_SILVER_PRICES}")


def process_fundamentals_scd2():
    print(f"--- {TODAY_STR} - Processing FUNDAMENTALS (SCD2) ---")

    try:
        df_new_raw = spark.read.json(PATH_BRONZE_FUND)
    except Exception as e:
        print(f"Pas de données fondamentales pour aujourd'hui: {e}")
        return

    df_new = df_new_raw.select(
        col("symbol"),
        col("date").alias("report_date"),
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
        print("Premier run: Création de la table Silver Fundamentals.")
        df_history = None
        df_active = None
        df_closed = None

    if df_active is None:
        df_final = df_new.withColumn("start_date", col("effective_date")) \
            .withColumn("end_date", lit(None).cast("date")) \
            .withColumn("is_current", lit(True))
    else:
        df_active_renamed = df_active.select(
            col("symbol").alias("h_symbol"),
            col("pe_ratio").alias("h_pe_ratio"),
            col("debt_to_equity").alias("h_debt_to_equity"),
            col("current_ratio").alias("h_current_ratio"),
            col("start_date"),
            col("end_date"),
            col("is_current"),
            col("report_date").alias("h_report_date")
        )

        cond = [df_new.symbol == df_active_renamed.h_symbol]
        joined = df_new.join(df_active_renamed, cond, "left_outer")

        changed_records = joined.filter(
            (col("h_symbol").isNotNull()) &
            (col("pe_ratio") != col("h_pe_ratio"))
        ).select(
            col("h_symbol").alias("symbol"),
            col("start_date"),
            col("effective_date").alias("end_date"),
            lit(False).alias("is_current"),
            col("h_pe_ratio").alias("pe_ratio"),
            col("h_debt_to_equity").alias("debt_to_equity"),
            col("h_current_ratio").alias("current_ratio"),
            col("h_report_date").alias("report_date")
        )

        new_records = joined.select(
            col("symbol"),
            col("effective_date").alias("start_date"),
            lit(None).cast("date").alias("end_date"),
            lit(True).alias("is_current"),
            col("pe_ratio"),
            col("debt_to_equity"),
            col("current_ratio"),
            col("report_date")
        )

        unchanged_ids = df_active.join(df_new,
                                       df_active.symbol == df_new.symbol,
                                       "left_anti")

        df_final = df_closed.unionByName(changed_records, allowMissingColumns=True) \
            .unionByName(new_records, allowMissingColumns=True) \
            .unionByName(unchanged_ids, allowMissingColumns=True)

    print(f"Écriture/Overwrite SCD2 dans {PATH_SILVER_FUND}")

    df_final.write.mode("overwrite").parquet(PATH_SILVER_FUND)


if __name__ == "__main__":
    process_prices_timeseries()
    process_fundamentals_scd2()
    print("ETL Bronze -> Silver terminé.")