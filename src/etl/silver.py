import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, to_date, current_timestamp, lit, when
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
    print("[INFO] Traitement prix (Schéma Strict & Overwrite)")

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
        current_timestamp().alias("ingestion_ts"),
        lit("twelvedata").alias("source_api"),
        lit(TODAY_STR).cast("date").alias("ingestion_date")
    )

    df_validated = df_silver.withColumn(
        "is_valid",
        when(col("close").isNull() | (col("close") < 0), False)
        .when(col("volume").isNull() | (col("volume") < 0), False)
        .when(col("trade_date").isNull(), False)
        .when(col("open") > col("high"), False)
        .when(col("low") > col("close"), False)
        .otherwise(True)
    )

    invalid_count = df_validated.filter(col("is_valid") == False).count()
    if invalid_count > 0:
        print(f"[WARN] {invalid_count} lignes invalides détectées et rejetées :")
        df_validated.filter(col("is_valid") == False).select(
            "symbol", "trade_date", "open", "high", "low", "close", "volume"
        ).show(10, truncate=False)

    df_final = df_validated.filter(col("is_valid") == True).drop("is_valid")

    valid_count = df_final.count()
    print(f"[INFO] {valid_count} lignes valides à écrire")

    if valid_count == 0:
        print("[WARN] Aucune ligne valide après validation. Rien écrit.")
        return

    print(f"[INFO] Écriture (OVERWRITE) Silver Prices dans : {PATH_SILVER_PRICES}")
    df_final.write.mode("overwrite").partitionBy("trade_date").parquet(PATH_SILVER_PRICES)
    print("[SUCCESS] Silver Prices mis à jour.")


def process_fundamentals_scd2():
    print("[INFO] Traitement des fondamentaux (SCD Type 2)")

    try:
        df_raw = spark.read.json(PATH_BRONZE_FUND)
        if df_raw.count() == 0:
            print("[INFO] Pas de nouveaux fondamentaux.")
            return
    except Exception as e:
        print(f"[INFO] Pas de nouveaux fondamentaux : {e}")
        return

    expected_cols = ["symbol", "date", "priceToEarningsRatio", "debtToEquityRatio"]
    for c in expected_cols:
        if c not in df_raw.columns:
            df_raw = df_raw.withColumn(c, lit(None))

    df_new_clean = df_raw.select(
        col("symbol"),
        col("date").cast("date").alias("report_date"),
        col("priceToEarningsRatio").cast("double").alias("pe_ratio"),
        col("debtToEquityRatio").cast("double").alias("debt_to_equity"),
        lit(TODAY_STR).cast("date").alias("effective_date"),
        lit("fmp").alias("source_api"),
        current_timestamp().alias("ingestion_ts")
    ).dropna(subset=["symbol", "report_date"])

    new_count = df_new_clean.count()
    print(f"[INFO] {new_count} lignes de nouveaux fondamentaux après nettoyage")
    if new_count == 0:
        print("[INFO] Rien à traiter après nettoyage.")
        return

    try:
        df_existing = spark.read.parquet(PATH_SILVER_FUND)
        df_existing = df_existing.cache()
        count_existing = df_existing.count()
        print(f"[INFO] {count_existing} lignes existantes chargées depuis Silver")
        has_history = True

        if "is_current" not in df_existing.columns:
            print("[INFO] Migration vers SCD2 : ajout des colonnes is_current et expiration_date")
            df_existing = df_existing \
                .withColumn("is_current", lit(True)) \
                .withColumn("expiration_date", lit(None).cast("date"))
        if "source_api" not in df_existing.columns:
            df_existing = df_existing.withColumn("source_api", lit("fmp"))
        if "ingestion_ts" not in df_existing.columns:
            df_existing = df_existing.withColumn("ingestion_ts", current_timestamp())

    except Exception:
        df_existing = None
        has_history = False
        print("[INFO] Première ingestion — pas d'historique existant")

    if has_history and df_existing is not None:

        df_current = df_existing.filter(col("is_current") == True)

        df_changes = df_current.alias("old") \
            .join(df_new_clean.alias("new"),
                  (col("old.symbol") == col("new.symbol")) &
                  (col("old.report_date") == col("new.report_date")),
                  "inner") \
            .where(
                (col("old.pe_ratio") != col("new.pe_ratio")) |
                (col("old.debt_to_equity") != col("new.debt_to_equity"))
            ) \
            .select(
                col("old.symbol"),
                col("old.report_date"),
                col("old.pe_ratio"),
                col("old.debt_to_equity"),
                col("old.effective_date"),
                col("old.source_api"),
                col("old.ingestion_ts")
            )

        changes_count = df_changes.count()

        df_new_symbols = df_new_clean.alias("new") \
            .join(df_current.alias("old"),
                  (col("new.symbol") == col("old.symbol")) &
                  (col("new.report_date") == col("old.report_date")),
                  "left_anti")

        new_symbols_count = df_new_symbols.count()
        print(f"[INFO] Changements détectés : {changes_count} | Nouveaux symboles/dates : {new_symbols_count}")

        if changes_count > 0:
            df_to_expire = df_changes \
                .withColumn("is_current", lit(False)) \
                .withColumn("expiration_date", lit(TODAY_STR).cast("date"))

            df_unchanged = df_existing.join(
                df_changes,
                ["symbol", "report_date", "effective_date"],
                "left_anti"
            )
        else:
            print("[INFO] Aucune modification détectée sur les symboles existants")
            df_to_expire = spark.createDataFrame([], df_existing.schema)
            df_unchanged = df_existing

        df_existing.unpersist()

        df_new_flagged = df_new_clean \
            .withColumn("is_current", lit(True)) \
            .withColumn("expiration_date", lit(None).cast("date"))

        df_final = df_unchanged.unionByName(df_to_expire).unionByName(df_new_flagged)

    else:
        df_final = df_new_clean \
            .withColumn("is_current", lit(True)) \
            .withColumn("expiration_date", lit(None).cast("date"))

    df_final = df_final.cache()
    total = df_final.count()
    current = df_final.filter(col("is_current") == True).count()
    expired = df_final.filter(col("is_current") == False).count()
    print(f"[INFO] Métriques avant écriture — Total: {total} | Actifs: {current} | Expirés: {expired}")

    print(f"[INFO] Écriture dans : {PATH_SILVER_FUND}")
    df_final.write.mode("overwrite").parquet(PATH_SILVER_FUND)
    df_final.unpersist()

    print(f"[SUCCESS] Silver Fundamentals mis à jour — Total: {total} | Actifs: {current} | Expirés: {expired}")


if __name__ == "__main__":
    process_prices_timeseries()
    process_fundamentals_scd2()
