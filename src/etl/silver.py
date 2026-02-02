import logging
import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, to_date, current_timestamp, lit, when
)
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

# Set up custom SUCCESS log level for better visibility of successful operations
SUCCESS = 25
logging.addLevelName(SUCCESS, 'SUCCESS')

def success(self, message, *args, **kwargs):
    """Custom log method to track successful operations"""
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, message, args, **kwargs)

logging.Logger.success = success

# Configure logging format with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# GCS bucket and date configuration
BUCKET_NAME = os.getenv("BUCKET_NAME", "finance_datalake")
TODAY_STR = datetime.now().strftime("%Y-%m-%d")

# GCP authentication and Spark connector paths
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_key.json")
JAR_PATH = "/tmp/gcs-connector.jar"

# Data paths for bronze and silver layers
PATH_BRONZE_PRICES = f"gs://{BUCKET_NAME}/bronze/twelvedata/prices/dt={TODAY_STR}/"
PATH_SILVER_PRICES = f"gs://{BUCKET_NAME}/silver/prices/"

PATH_BRONZE_FUND = f"gs://{BUCKET_NAME}/bronze/fmp/ratios/dt={TODAY_STR}/"
PATH_SILVER_FUND = f"gs://{BUCKET_NAME}/silver/fundamentals/"

# Initialize Spark session with GCS connector
logger.info("Initializing Spark session with GCS connector...")
spark = SparkSession.builder \
    .appName("Finance_Bronze_To_Silver_ETL") \
    .config("spark.driver.memory", "4g") \
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
logger.info("Spark session initialized successfully")


def process_prices_timeseries():
    """
    Transform raw price data from bronze to silver layer

    This process:
    - Reads JSON price data from TwelveData API
    - Applies strict schema validation
    - Validates data quality (no negative prices, logical OHLC relationships)
    - Writes validated data to parquet format partitioned by trade date
    """
    logger.info("Starting price data transformation (bronze → silver)")

    # Define strict schema for price data to ensure data quality
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

    # Read bronze price data with schema validation
    try:
        logger.info(f"Reading price data from {PATH_BRONZE_PRICES}")
        df_raw = spark.read.schema(price_schema).json(PATH_BRONZE_PRICES)
        logger.debug(f"Raw price data loaded, checking record count...")
    except Exception as error:
        logger.error(f"Failed to read price data: {error}")
        return

    # Check if we actually have data to process
    row_count = df_raw.count()
    if row_count == 0:
        logger.warning("No price data found in bronze layer, skipping transformation")
        return

    logger.info(f"Loaded {row_count} raw price records")

    # Transform nested JSON structure into flat table
    logger.debug("Flattening nested JSON structure...")
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

    # Apply data quality rules
    logger.info("Applying data quality validations...")
    df_validated = df_silver.withColumn(
        "is_valid",
        when(col("close").isNull() | (col("close") < 0), False)     # Close price must exist and be positive
        .when(col("volume").isNull() | (col("volume") < 0), False)  # Volume must exist and be positive
        .when(col("trade_date").isNull(), False)                    # Date must exist
        .when(col("open") > col("high"), False)                     # Open can't be higher than high
        .when(col("low") > col("close"), False)                     # Low can't be higher than close
        .otherwise(True)
    )

    # Log validation results
    invalid_count = df_validated.filter(col("is_valid") == False).count()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid rows that will be filtered out")
        logger.debug("Sample of invalid rows:")
        df_validated.filter(col("is_valid") == False).select(
            "symbol", "trade_date", "open", "high", "low", "close", "volume"
        ).show(10, truncate=False)

    # Keep only valid records
    df_final = df_validated.filter(col("is_valid") == True).drop("is_valid")
    valid_count = df_final.count()

    logger.info(f"Validation complete: {valid_count} valid rows, {invalid_count} invalid rows filtered")

    if valid_count == 0:
        logger.warning("No valid data to write after validation")
        return

    # Write to silver layer partitioned by trade date for efficient querying
    logger.info(f"Writing {valid_count} records to {PATH_SILVER_PRICES}")
    df_final.write.mode("overwrite").partitionBy("trade_date").parquet(PATH_SILVER_PRICES)
    logger.success(f"✓ Silver prices updated successfully: {valid_count} records written")


def process_fundamentals_scd2():
    """
    Transform fundamental data from bronze to silver layer using SCD Type 2

    Slowly Changing Dimension Type 2 (SCD2) tracks historical changes by:
    - Keeping all versions of each record
    - Marking current records with is_current=True
    - Expiring old records when values change
    - Adding effective_date and expiration_date for time travel queries
    """
    logger.info("Starting fundamentals transformation with SCD Type 2")

    # Read raw fundamentals data from bronze
    try:
        logger.info(f"Reading fundamentals data from {PATH_BRONZE_FUND}")
        df_raw = spark.read.json(PATH_BRONZE_FUND)
        if df_raw.count() == 0:
            logger.info("No fundamentals data to process")
            return
    except Exception as error:
        logger.error(f"Failed to read fundamentals data: {error}")
        return

    # Ensure all expected columns exist (some API responses may be incomplete)
    expected_cols = ["symbol", "date", "priceToEarningsRatio", "debtToEquityRatio"]
    for c in expected_cols:
        if c not in df_raw.columns:
            logger.debug(f"Column '{c}' missing, adding as null")
            df_raw = df_raw.withColumn(c, lit(None))

    # Clean and standardize the new data
    logger.debug("Cleaning and standardizing new fundamentals data...")
    df_new_clean = df_raw.select(
        col("symbol"),
        col("date").cast("date").alias("report_date"),
        col("priceToEarningsRatio").cast("double").alias("pe_ratio"),
        col("debtToEquityRatio").cast("double").alias("debt_to_equity"),
        lit(TODAY_STR).cast("date").alias("effective_date"),
        lit("fmp").alias("source_api"),
        current_timestamp().alias("ingestion_ts")
    ).dropna(subset=["symbol", "report_date"])  # Symbol and date are required

    new_count = df_new_clean.count()
    logger.info(f"Cleaned {new_count} new fundamental records")

    if new_count == 0:
        logger.warning("No valid fundamental data to process after cleaning")
        return

    # Try to load existing silver data for SCD2 comparison
    try:
        logger.info("Loading existing fundamentals from silver layer...")
        df_existing = spark.read.parquet(PATH_SILVER_FUND)
        df_existing = df_existing.cache()
        count_existing = df_existing.count()
        logger.info(f"Loaded {count_existing} existing fundamental records")
        has_history = True

        # Handle schema evolution - add missing SCD2 columns if needed
        if "is_current" not in df_existing.columns:
            logger.info("Adding missing SCD2 columns to existing data")
            df_existing = df_existing \
                .withColumn("is_current", lit(True)) \
                .withColumn("expiration_date", lit(None).cast("date"))

        # Ensure source tracking column exists
        if "source_api" not in df_existing.columns:
            df_existing = df_existing.withColumn("source_api", lit("fmp"))

        # Ensure ingestion timestamp exists
        if "ingestion_ts" not in df_existing.columns:
            df_existing = df_existing.withColumn("ingestion_ts", current_timestamp())

    except Exception:
        df_existing = None
        has_history = False
        logger.info("No existing fundamentals found, initializing new dataset")

    # Apply SCD2 logic if we have historical data
    if has_history and df_existing is not None:
        logger.info("Applying SCD Type 2 logic to track changes...")

        # Get only current records to compare with new data
        df_current = df_existing.filter(col("is_current") == True)

        # Find records where values have changed
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

        # Find completely new records (new symbol/date combinations)
        df_new_symbols = df_new_clean.alias("new") \
            .join(df_current.alias("old"),
                  (col("new.symbol") == col("old.symbol")) &
                  (col("new.report_date") == col("old.report_date")),
                  "left_anti")

        new_symbols_count = df_new_symbols.count()
        logger.info(f"Change detection: {changes_count} records changed, {new_symbols_count} new records")

        # Expire old versions of changed records
        if changes_count > 0:
            logger.debug(f"Expiring {changes_count} old record versions")
            df_to_expire = df_changes \
                .withColumn("is_current", lit(False)) \
                .withColumn("expiration_date", lit(TODAY_STR).cast("date"))

            # Keep all records that didn't change
            df_unchanged = df_existing.join(
                df_changes,
                ["symbol", "report_date", "effective_date"],
                "left_anti"
            )
        else:
            logger.info("No changes detected, all records remain current")
            df_to_expire = spark.createDataFrame([], df_existing.schema)
            df_unchanged = df_existing

        df_existing.unpersist()

        # Mark all new records as current
        df_new_flagged = df_new_clean \
            .withColumn("is_current", lit(True)) \
            .withColumn("expiration_date", lit(None).cast("date"))

        # Combine unchanged + expired + new records
        df_final = df_unchanged.unionByName(df_to_expire).unionByName(df_new_flagged)

    else:
        # No history exists, just flag all new records as current
        logger.debug("Initializing dataset with all records marked as current")
        df_final = df_new_clean \
            .withColumn("is_current", lit(True)) \
            .withColumn("expiration_date", lit(None).cast("date"))

    # Calculate final metrics
    df_final = df_final.cache()
    total = df_final.count()
    current = df_final.filter(col("is_current") == True).count()
    expired = df_final.filter(col("is_current") == False).count()

    logger.info(f"Final dataset metrics: {total} total records ({current} current, {expired} expired)")

    # Write to silver layer
    logger.info(f"Writing to {PATH_SILVER_FUND}")
    df_final.write.mode("overwrite").parquet(PATH_SILVER_FUND)
    df_final.unpersist()

    logger.success(f"Silver fundamentals updated: {total} records ({current} current, {expired} historical)")


if __name__ == "__main__":
    logger.info("Starting silver layer ETL pipeline")

    # Process price data first (usually smaller and faster)
    process_prices_timeseries()

    # Then process fundamentals with SCD2 logic
    process_fundamentals_scd2()

    logger.success("Silver layer ETL pipeline completed")
