import logging
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, stddev, lag, when, abs as spark_abs,
    lead, row_number, desc, count, isnan, log, lit
)
from pyspark.sql.window import Window

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

# GCS bucket and GCP authentication configuration
BUCKET_NAME = os.getenv("BUCKET_NAME", "finance_datalake")
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_key.json")
JAR_PATH = "/tmp/gcs-connector.jar"

# Initialize Spark session with GCS connector
logger.info("Initializing Spark session for gold layer feature engineering...")
spark = SparkSession.builder \
    .appName("Finance_Gold_Advanced_Features") \
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

def calculate_technical_indicators(df):
    """
    Calculate a comprehensive set of technical indicators for trading signals

    Technical indicators computed:
    - SMA (Simple Moving Average): Average price over 20 days
    - Bollinger Bands: Volatility bands (mean ± 2 std deviations)
    - EMA (Exponential Moving Average): Weighted averages for 12 and 26 days
    - MACD (Moving Average Convergence Divergence): Momentum indicator
    - RSI (Relative Strength Index): Overbought/oversold indicator (0-100 scale)

    Args:
        df: Spark DataFrame with price data

    Returns:
        DataFrame enriched with technical indicators
    """
    logger.info("Calculating technical indicators...")

    # Window specification for per-symbol calculations ordered by date
    w_spec = Window.partitionBy("symbol").orderBy("trade_date")

    # Calculate Bollinger Bands (20-day SMA ± 2 standard deviations)
    logger.debug("Calculating Bollinger Bands (20-day window)")
    w_20 = w_spec.rowsBetween(-19, 0)  # Last 20 days including current
    df = df.withColumn("sma_20", avg("close").over(w_20)) \
           .withColumn("stddev_20", stddev("close").over(w_20)) \
           .withColumn("bollinger_upper", col("sma_20") + (col("stddev_20") * 2)) \
           .withColumn("bollinger_lower", col("sma_20") - (col("stddev_20") * 2))

    # EMA approximation using weighted SMA (close enough for feature engineering)
    # True EMA would require pandas_udf which causes Arrow issues on GitHub Actions
    logger.debug("Calculating EMA approximations (12 and 26-day)")
    w_12 = w_spec.rowsBetween(-11, 0)
    w_26 = w_spec.rowsBetween(-25, 0)
    df = df.withColumn("ema_12", avg("close").over(w_12)) \
        .withColumn("ema_26", avg("close").over(w_26))

    # Calculate MACD Line (difference between 12-day and 26-day EMA)
    logger.debug("Calculating MACD line")
    df = df.withColumn("macd_line", col("ema_12") - col("ema_26"))

    # Calculate SMA difference (momentum indicator)
    logger.debug("Calculating SMA momentum indicator")
    w_spec = Window.partitionBy("symbol").orderBy("trade_date")
    w_12 = w_spec.rowsBetween(-11, 0)
    w_26 = w_spec.rowsBetween(-25, 0)
    df = df.withColumn("sma_diff", avg("close").over(w_12) - avg("close").over(w_26))

    # Calculate RSI (Relative Strength Index) - measures momentum
    logger.debug("Calculating RSI (14-day)")
    df = df.withColumn("diff", col("close") - lag("close", 1).over(w_spec))
    df = df.withColumn("gain", when(col("diff") > 0, col("diff")).otherwise(0)) \
           .withColumn("loss", when(col("diff") < 0, spark_abs(col("diff"))).otherwise(0))

    w_14 = w_spec.rowsBetween(-13, 0)  # 14-day window for RSI
    df = df.withColumn("avg_gain", avg("gain").over(w_14)) \
           .withColumn("avg_loss", avg("loss").over(w_14))

    # RSI formula: 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss
    df = df.withColumn("rs", col("avg_gain") / when(col("avg_loss") == 0, 1).otherwise(col("avg_loss"))) \
           .withColumn("rsi_14", 100 - (100 / (1 + col("rs"))))

    # Clean up intermediate calculation columns
    df = df.drop("diff", "gain", "loss", "avg_gain", "avg_loss", "rs", "stddev_20")

    logger.info("Technical indicators calculated successfully")
    return df


def create_advanced_gold():
    """
    Create gold layer with ML-ready features by combining price and fundamental data

    This process:
    1. Loads price and fundamental data from silver layer
    2. Joins them using point-in-time logic (avoiding lookahead bias)
    3. Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
    4. Creates lag features and volatility measures
    5. Generates target variable (next day return)
    6. Outputs clean, ML-ready dataset partitioned by symbol
    """
    logger.info("Starting gold layer feature engineering")

    # Load silver layer data
    try:
        logger.info("Loading data from silver layer...")
        df_prices = spark.read.parquet(f"gs://{BUCKET_NAME}/silver/prices/")
        df_fund = spark.read.parquet(f"gs://{BUCKET_NAME}/silver/fundamentals/")

        price_count = df_prices.count()
        fund_count = df_fund.count()
        logger.info(f"Loaded {price_count} price records and {fund_count} fundamental records")
    except Exception as error:
        logger.error(f"Failed to read silver layer data: {error}")
        return

    # Filter to current fundamentals only (SCD2 latest version)
    logger.info("Filtering to current fundamental records only...")
    df_fund_current = df_fund.filter(col("is_current") == True)
    current_fund_count = df_fund_current.count()
    logger.debug(f"{current_fund_count} current fundamental records")

    # Prepare fundamentals for point-in-time join
    fund_renamed = df_fund_current.select(
        col("symbol").alias("f_symbol"),
        col("report_date").alias("f_date"),
        col("pe_ratio"),
        col("debt_to_equity")
    )

    # Broadcast hint for small fundamental table (optimizes join performance)
    fund_broadcast = spark.createDataFrame(fund_renamed.collect()).hint("broadcast")

    # Point-in-time join: match each price date with the most recent fundamental data
    # This avoids lookahead bias (using future fundamentals for past predictions)
    logger.info("Performing point-in-time join (prices + fundamentals)...")
    cond = [
        df_prices.symbol == fund_broadcast.f_symbol,
        fund_broadcast.f_date <= df_prices.trade_date  # Only use fundamentals available at trade time
    ]

    df_merged = df_prices.join(fund_broadcast, cond, "left")

    # For each price point, keep only the most recent fundamental data
    logger.debug("Selecting most recent fundamental for each trade date")
    w_filter = Window.partitionBy("symbol", "trade_date").orderBy(desc("f_date"))
    df_joined = df_merged.withColumn("rn", row_number().over(w_filter)) \
                         .filter(col("rn") == 1) \
                         .drop("rn", "f_symbol", "f_date")

    # Calculate all technical indicators
    df_calc = calculate_technical_indicators(df_joined)

    # Calculate daily returns using log returns (more stable for ML)
    logger.info("Creating ML features...")
    logger.debug("Calculating daily log returns")
    w_spec = Window.partitionBy("symbol").orderBy("trade_date")
    df_calc = df_calc.withColumn("daily_return", log(col("close") / lag("close", 1).over(w_spec)))

    # Create lag features (previous days' returns as predictors)
    logger.debug("Creating lag features (t-1, t-2, t-3)")
    df_calc = df_calc.withColumn("return_lag_1", lag("daily_return", 1).over(w_spec)) \
                     .withColumn("return_lag_2", lag("daily_return", 2).over(w_spec)) \
                     .withColumn("return_lag_3", lag("daily_return", 3).over(w_spec))

    # Calculate rolling volatility (10-day standard deviation of returns)
    logger.debug("Calculating 10-day volatility")
    w_vol = w_spec.rowsBetween(-9, 0)
    df_calc = df_calc.withColumn("volatility_10d", stddev("daily_return").over(w_vol))

    # Create target variable: next day's return (for supervised learning)
    logger.debug("Creating target variable (next day return)")
    w_lead = Window.partitionBy("symbol").orderBy("trade_date")
    bb_width = col("bollinger_upper") - col("bollinger_lower")

    # Select and engineer final features
    df_final_temp = df_calc.select(
        "symbol", "trade_date", "close", "volume",
        col("rsi_14"),                                                  # Momentum indicator
        col("macd_line"),                                               # Trend indicator
        col("sma_diff"),                                                # Another trend indicator
        col("return_lag_1"), col("return_lag_2"), col("return_lag_3"),  # Historical returns
        col("volatility_10d"),                                          # Risk measure
        # %B indicator: position within Bollinger Bands (0 = at lower band, 1 = at upper band)
        when(bb_width == 0, 0).otherwise(
            (col("close") - col("bollinger_lower")) / bb_width
        ).alias("pct_b"),
        col("pe_ratio"),                                                # Valuation metric
        col("debt_to_equity"),                                          # Financial health metric
        # Target variable: next day's return (what we want to predict)
        when(col("close") == 0, 0)
        .otherwise((lead("close", 1).over(w_lead) - col("close")) / col("close"))
        .alias("target_return_next_day"),
        lit("silver_prices + silver_fundamentals").alias("data_lineage")
    )

    # Fill missing values with 0 for ML compatibility
    logger.debug("Filling missing values")
    fill_cols = ["pe_ratio", "debt_to_equity", "rsi_14", "macd_line", "sma_diff", "pct_b",
                 "return_lag_1", "return_lag_2", "return_lag_3", "volatility_10d"]
    df_filled = df_final_temp.na.fill(0, subset=fill_cols)

    # Check data quality
    logger.debug("Checking data quality...")
    df_filled.select(
        count(when(col("trade_date").isNull(), col("trade_date"))).alias("nb_null_dates"),
        count(when(col("close").isNull() | isnan(col("close")), col("close"))).alias("nb_null_close")
    ).show()

    # Drop rows with null critical fields
    df_final = df_filled.dropna(subset=["trade_date", "close"])

    rows = df_final.count()
    logger.info(f"Feature engineering complete: {rows} ML-ready records")

    # Write to gold layer
    if rows > 0:
        output_path = f"gs://{BUCKET_NAME}/gold/advanced_features/"
        logger.info(f"Writing {rows} records to {output_path}")
        df_final.write.mode("overwrite").partitionBy("symbol").parquet(output_path)
        logger.success(f"Gold layer created successfully: {rows} records written to {output_path}")
    else:
        logger.warning("No data to write after feature engineering")


if __name__ == "__main__":
    logger.info("Starting gold layer ETL pipeline")
    create_advanced_gold()
    logger.info("Gold layer ETL pipeline completed")
