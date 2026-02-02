import logging
import os

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, exp, sum as spark_sum, min as spark_min,
    max as spark_max, avg as spark_avg, stddev as spark_stddev,
    count as spark_count, sqrt, row_number
)
from pyspark.sql.utils import AnalysisException
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
logger.info("Initializing Spark session for ML training and backtesting...")
spark = SparkSession.builder \
    .appName("Finance_Pro_ML_Backtest") \
    .config("spark.driver.memory", "4g") \
    .config("spark.jars", JAR_PATH) \
    .config("spark.driver.extraClassPath", JAR_PATH) \
    .config("spark.executor.extraClassPath", JAR_PATH) \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
    .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", KEY_PATH) \
    .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
    .config("spark.hadoop.fs.gs.committer.scheme", "hadoop") \
    .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm", "2") \
    .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") \
    .getOrCreate()
logger.info("Spark session initialized successfully")


def run_ml_analysis():
    """
    Complete ML pipeline for financial forecasting and backtesting

    This function:
    1. Loads engineered features from gold layer
    2. Trains a Gradient Boosted Trees model with cross-validation
    3. Generates predictions on test set
    4. Performs backtesting with two strategies:
       - Signal strategy: invest when predicted return > threshold
       - Ranking strategy: invest in top 40% predicted performers each day
    5. Calculates performance metrics (Sharpe ratio, alpha, drawdown, etc.)
    6. Generates future predictions for the latest available date
    """
    logger.info("Starting ML training and backtesting pipeline")

    # Load gold layer features
    logger.info("Loading gold layer features...")
    try:
        df_raw = spark.read.parquet(f"gs://{BUCKET_NAME}/gold/advanced_features/")
        row_count_raw = df_raw.count()
        logger.info(f"Loaded {row_count_raw} records from gold layer")
    except AnalysisException:
        logger.error("Gold layer directory not found or empty - run gold ETL first")
        return

    # Define features for ML model
    features = [
        "rsi_14", "macd_line", "sma_diff", "pct_b",                       # Technical indicators
        "pe_ratio", "debt_to_equity", "volume",                           # Fundamental & volume metrics
        "return_lag_1", "return_lag_2", "return_lag_3", "volatility_10d"  # Historical returns & volatility
    ]
    label_col = "target_return_next_day"

    # Clean data - remove rows with missing values in features or target
    logger.info("Preparing training data...")
    df_train_ready = df_raw.na.drop(subset=[label_col] + features)

    row_count = df_train_ready.count()
    logger.info(f"Clean dataset: {row_count} records (dropped {row_count_raw - row_count} rows with nulls)")

    if row_count == 0:
        logger.error("No data available after cleaning - cannot train model")
        return

    # Time-based train/test split to avoid lookahead bias
    split_date = "2024-06-01"
    logger.info(f"Splitting data at {split_date} (train < split_date, test >= split_date)")
    train = df_train_ready.filter(col("trade_date") < split_date)
    test = df_train_ready.filter(col("trade_date") >= split_date)

    train_count = train.count()
    test_count = test.count()
    logger.info(f"Train set: {train_count} records | Test set: {test_count} records")

    if train_count == 0 or test_count == 0:
        logger.error("Train or test set is empty - adjust split date")
        return

    # Build ML pipeline: assemble features → scale → GBT regressor
    logger.info("Building ML pipeline (VectorAssembler → StandardScaler → GBTRegressor)")
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    gbt = GBTRegressor(featuresCol="features", labelCol=label_col, seed=42)
    pipeline = Pipeline(stages=[assembler, scaler, gbt])

    # Hyperparameter tuning with cross-validation
    logger.info("Starting cross-validation hyperparameter tuning...")
    parameter_grid = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [5, 8]) \
        .addGrid(gbt.maxIter, [20, 40]) \
        .build()

    evaluator = RegressionEvaluator(labelCol=label_col, metricName="rmse")

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=parameter_grid,
        evaluator=evaluator,
        numFolds=3
    )

    logger.info("Training model (this may take several minutes)...")
    model = cv.fit(train)
    best_model = model.bestModel
    best_rmse = min(model.avgMetrics)
    logger.success(f"Model training complete - Best CV RMSE: {best_rmse:.6f}")

    # Generate predictions on test set
    logger.info("Generating predictions on test set...")
    predictions = best_model.transform(test)

    # Window specifications for backtesting calculations
    w_symbol = Window.partitionBy("symbol").orderBy("trade_date")
    w_date = Window.partitionBy("trade_date").orderBy(col("prediction").desc())

    # Calculate backtesting metrics with two strategies
    logger.info("Calculating backtest metrics...")
    logger.debug("Strategy 1: Signal-based (invest when prediction > -0.5%)")
    logger.debug("Strategy 2: Ranking-based (invest in top 6 predictions each day)")

    df_backtest = predictions.withColumn(
        # Strategy 1: Simple signal - invest if predicted return > -0.5%
        "strategy_signal", when(col("prediction") > -0.005, lit(1)).otherwise(lit(0))
    ).withColumn(
        # Rank predictions within each trading day
        "prediction_rank", row_number().over(w_date)
    ).withColumn(
        # Strategy 2: Ranking - invest only in top 6 predictions per day
        "strategy_signal_rank", when(col("prediction_rank") <= 6, lit(1)).otherwise(lit(0))
    ).withColumn(
        # Calculate actual returns for each strategy
        "strategy_return", col("strategy_signal") * col(label_col)
    ).withColumn(
        "strategy_return_rank", col("strategy_signal_rank") * col(label_col)
    ).withColumn(
        "market_return", col(label_col)
    ).withColumn(
        # Calculate cumulative returns over time
        "cum_strategy_return", spark_sum("strategy_return").over(w_symbol)
    ).withColumn(
        "cum_strategy_return_rank", spark_sum("strategy_return_rank").over(w_symbol)
    ).withColumn(
        "cum_market_return", spark_sum("market_return").over(w_symbol)
    ).withColumn(
        # Convert log returns back to wealth multiplier
        "strategy_wealth", exp(col("cum_strategy_return"))
    ).withColumn(
        "strategy_wealth_rank", exp(col("cum_strategy_return_rank"))
    ).withColumn(
        "market_wealth", exp(col("cum_market_return"))
    ).withColumn(
        # Track maximum wealth for drawdown calculation
        "strategy_wealth_max", spark_max("strategy_wealth").over(
            w_symbol.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
    ).withColumn(
        "strategy_wealth_rank_max", spark_max("strategy_wealth_rank").over(
            w_symbol.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
    ).withColumn(
        # Calculate drawdown (peak-to-trough decline)
        "drawdown", (col("strategy_wealth") - col("strategy_wealth_max")) / col("strategy_wealth_max")
    ).withColumn(
        "drawdown_rank", (col("strategy_wealth_rank") - col("strategy_wealth_rank_max")) / col("strategy_wealth_rank_max")
    )

    # Select relevant columns for output
    output_cols = [
        "symbol", "trade_date", "close",
        label_col, "prediction", "prediction_rank",
        "strategy_signal", "strategy_signal_rank",
        "strategy_return", "strategy_return_rank", "market_return",
        "cum_strategy_return", "cum_strategy_return_rank", "cum_market_return",
        "strategy_wealth", "strategy_wealth_rank", "market_wealth",
        "drawdown", "drawdown_rank"
    ] + features

    # Write backtest results
    path_backtest = f"gs://{BUCKET_NAME}/gold/backtest_results/"
    logger.info(f"Writing backtest results to {path_backtest}")
    df_backtest.select(output_cols).write.mode("overwrite").partitionBy("symbol").parquet(path_backtest)
    logger.success(f"Backtest results saved")

    # Calculate performance metrics per symbol
    logger.info("Calculating performance metrics per symbol...")
    trading_days = 252  # Annualization factor

    df_metrics = df_backtest.groupBy("symbol").agg(
        # Total returns
        spark_max("cum_strategy_return").alias("total_strategy_return"),
        spark_max("cum_market_return").alias("total_market_return"),
        spark_max("cum_strategy_return_rank").alias("total_strategy_return_rank"),

        # Trading statistics
        spark_count("trade_date").alias("num_trading_days"),
        spark_sum("strategy_signal").alias("num_days_invested"),
        spark_sum("strategy_signal_rank").alias("num_days_invested_rank"),

        # Return statistics (for Sharpe ratio calculation)
        spark_avg("strategy_return").alias("avg_daily_strategy_return"),
        spark_avg("market_return").alias("avg_daily_market_return"),
        spark_avg("strategy_return_rank").alias("avg_daily_strategy_return_rank"),
        spark_stddev("strategy_return").alias("stddev_strategy_return"),
        spark_stddev("market_return").alias("stddev_market_return"),
        spark_stddev("strategy_return_rank").alias("stddev_strategy_return_rank"),

        # Risk metrics
        spark_min("drawdown").alias("max_drawdown"),
        spark_min("drawdown_rank").alias("max_drawdown_rank"),

        # Model accuracy metrics
        spark_sum(
            when((col("prediction") > 0) & (col(label_col) > 0), lit(1)).otherwise(lit(0))
        ).alias("true_positives"),
        spark_sum(
            when(col("prediction") > 0, lit(1)).otherwise(lit(0))
        ).alias("total_positive_predictions")
    ).withColumn(
        # Sharpe Ratio: risk-adjusted return (annualized)
        "sharpe_ratio", when(
            col("stddev_strategy_return") == 0, lit(0)
        ).otherwise(
            (col("avg_daily_strategy_return") / col("stddev_strategy_return")) * sqrt(lit(trading_days))
        )
    ).withColumn(
        "sharpe_ratio_rank", when(
            col("stddev_strategy_return_rank") == 0, lit(0)
        ).otherwise(
            (col("avg_daily_strategy_return_rank") / col("stddev_strategy_return_rank")) * sqrt(lit(trading_days))
        )
    ).withColumn(
        "sharpe_ratio_market", when(
            col("stddev_market_return") == 0, lit(0)
        ).otherwise(
            (col("avg_daily_market_return") / col("stddev_market_return")) * sqrt(lit(trading_days))
        )
    ).withColumn(
        # Model precision: % of positive predictions that were correct
        "model_precision_pct", when(
            col("total_positive_predictions") == 0, lit(0)
        ).otherwise(
            (col("true_positives") / col("total_positive_predictions")) * 100
        )
    ).withColumn(
        # Alpha: excess return vs market (buy and hold)
        "alpha", col("total_strategy_return") - col("total_market_return")
    ).withColumn(
        "alpha_rank", col("total_strategy_return_rank") - col("total_market_return")
    )

    metrics_cols = [
        "symbol",
        "total_strategy_return", "total_strategy_return_rank", "total_market_return",
        "alpha", "alpha_rank",
        "num_trading_days", "num_days_invested", "num_days_invested_rank",
        "avg_daily_strategy_return", "avg_daily_strategy_return_rank", "avg_daily_market_return",
        "stddev_strategy_return", "stddev_strategy_return_rank", "stddev_market_return",
        "sharpe_ratio", "sharpe_ratio_rank", "sharpe_ratio_market",
        "max_drawdown", "max_drawdown_rank",
        "model_precision_pct"
    ]

    # Write metrics to GCS
    path_metrics = f"gs://{BUCKET_NAME}/gold/backtest_metrics/"
    logger.info(f"Writing performance metrics to {path_metrics}")
    df_metrics.select(metrics_cols).write.mode("overwrite").parquet(path_metrics)
    logger.success(f"Performance metrics saved")

    # Display key metrics
    logger.info("Performance Summary - Signal Strategy")
    df_metrics.select(
        "symbol", "total_strategy_return", "total_market_return", "alpha",
        "sharpe_ratio", "sharpe_ratio_market", "max_drawdown", "model_precision_pct"
    ).show(truncate=False)

    logger.info("Performance Summary - Ranking Strategy (Top 40%)")
    df_metrics.select(
        "symbol", "total_strategy_return_rank", "total_market_return", "alpha_rank",
        "sharpe_ratio_rank", "sharpe_ratio_market", "max_drawdown_rank"
    ).show(truncate=False)

    # Generate future predictions for latest available date
    logger.info("Generating predictions for latest date")

    max_date_row = df_raw.agg({"trade_date": "max"}).collect()[0]
    last_date = max_date_row[0]
    logger.info(f"Latest available date: {last_date}")

    # Get data for the latest date
    df_future = df_raw.filter(col("trade_date") == last_date)
    df_future_clean = df_future.na.drop(subset=features)

    future_count = df_future_clean.count()
    if future_count > 0:
        logger.info(f"Generating predictions for {future_count} symbols...")
        future_preds = best_model.transform(df_future_clean)

        # Add trading recommendation based on prediction
        final_future = future_preds.select(
            "symbol", "trade_date", "close", "prediction"
        ).withColumn(
            "recommendation", when(col("prediction") > 0, "BUY").otherwise("HOLD")
        ).withColumn(
            "prediction_pct", col("prediction") * 100
        )

        # Write future predictions
        path_future = f"gs://{BUCKET_NAME}/gold/future_predictions/"
        logger.info(f"Writing future predictions to {path_future}")
        final_future.write.mode("overwrite").parquet(path_future)
        logger.success(f"Future predictions saved for {future_count} symbols")

        # Show top recommendations
        logger.info("Top 5 Buy Recommendations:")
        final_future.filter(col("recommendation") == "BUY") \
            .orderBy(col("prediction").desc()) \
            .select("symbol", "close", "prediction_pct", "recommendation") \
            .show(5, truncate=False)
    else:
        logger.warning("Cannot generate future predictions - missing features for latest date")

    logger.info("ML pipeline completed successfully")


if __name__ == "__main__":
    logger.info("Starting financial ML analysis and backtesting")
    run_ml_analysis()
    logger.info("Analysis complete")
