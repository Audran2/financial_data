import os
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import (
    col, when, lit, exp, sum as spark_sum, min as spark_min,
    max as spark_max, avg as spark_avg, stddev as spark_stddev,
    count as spark_count, sqrt, row_number
)

# --- CONFIGURATION ---
BUCKET_NAME = os.getenv("BUCKET_NAME", "finance_datalake")
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_key.json")
JAR_PATH = "/tmp/gcs-connector.jar"

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


def run_ml_analysis():
    print("--- Chargement des données Gold ---")
    try:
        df_raw = spark.read.parquet(f"gs://{BUCKET_NAME}/gold/advanced_features/")
    except AnalysisException:
        print("[ERREUR] Dossier Gold vide ou introuvable.")
        return

    features = [
        "rsi_14", "macd_line", "sma_diff", "pct_b",
        "pe_ratio", "debt_to_equity", "volume",
        "return_lag_1", "return_lag_2", "return_lag_3", "volatility_10d"
    ]
    label_col = "target_return_next_day"

    print("Préparation des données pour l'entraînement...")
    df_train_ready = df_raw.na.drop(subset=[label_col] + features)

    row_count = df_train_ready.count()
    print(f"Données disponibles après nettoyage : {row_count} lignes")

    if row_count == 0:
        print("[ERREUR] Pas assez de données pour entraîner.")
        return

    split_date = "2024-06-01"
    train = df_train_ready.filter(col("trade_date") < split_date)
    test = df_train_ready.filter(col("trade_date") >= split_date)

    train_count = train.count()
    test_count = test.count()
    print(f"Train : {train_count} lignes | Test : {test_count} lignes")

    if train_count == 0 or test_count == 0:
        print("[ERREUR] Train ou test vide après split temporel.")
        return

    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    gbt = GBTRegressor(featuresCol="features", labelCol=label_col, seed=42)
    pipeline = Pipeline(stages=[assembler, scaler, gbt])

    print("Début de l'entraînement (Cross Validation)...")
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

    model = cv.fit(train)
    best_model = model.bestModel
    print(f"[INFO] RMSE moyen (CV) : {min(model.avgMetrics):.6f}")

    print("Génération des prédictions sur le test set...")
    predictions = best_model.transform(test)

    w_symbol = Window.partitionBy("symbol").orderBy("trade_date")
    w_date = Window.partitionBy("trade_date").orderBy(col("prediction").desc())

    df_backtest = predictions.withColumn(
        "strategy_signal", when(col("prediction") > -0.005, lit(1)).otherwise(lit(0))
    ).withColumn(
        "prediction_rank", row_number().over(w_date)
    ).withColumn(
        "strategy_signal_rank", when(col("prediction_rank") <= 6, lit(1)).otherwise(lit(0))
    ).withColumn(
        "strategy_return", col("strategy_signal") * col(label_col)
    ).withColumn(
        "strategy_return_rank", col("strategy_signal_rank") * col(label_col)
    ).withColumn(
        "market_return", col(label_col)
    ).withColumn(
        "cum_strategy_return", spark_sum("strategy_return").over(w_symbol)
    ).withColumn(
        "cum_strategy_return_rank", spark_sum("strategy_return_rank").over(w_symbol)
    ).withColumn(
        "cum_market_return", spark_sum("market_return").over(w_symbol)
    ).withColumn(
        "strategy_wealth", exp(col("cum_strategy_return"))
    ).withColumn(
        "strategy_wealth_rank", exp(col("cum_strategy_return_rank"))
    ).withColumn(
        "market_wealth", exp(col("cum_market_return"))
    ).withColumn(
        "strategy_wealth_max", spark_max("strategy_wealth").over(
            w_symbol.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
    ).withColumn(
        "strategy_wealth_rank_max", spark_max("strategy_wealth_rank").over(
            w_symbol.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
    ).withColumn(
        "drawdown", (col("strategy_wealth") - col("strategy_wealth_max")) / col("strategy_wealth_max")
    ).withColumn(
        "drawdown_rank", (col("strategy_wealth_rank") - col("strategy_wealth_rank_max")) / col("strategy_wealth_rank_max")
    )

    output_cols = [
        "symbol", "trade_date", "close",
        label_col, "prediction", "prediction_rank",
        "strategy_signal", "strategy_signal_rank",
        "strategy_return", "strategy_return_rank", "market_return",
        "cum_strategy_return", "cum_strategy_return_rank", "cum_market_return",
        "strategy_wealth", "strategy_wealth_rank", "market_wealth",
        "drawdown", "drawdown_rank"
    ] + features

    path_backtest = f"gs://{BUCKET_NAME}/gold/backtest_results/"
    print(f"[INFO] Export Backtest vers : {path_backtest}")
    df_backtest.select(output_cols).write.mode("overwrite").partitionBy("symbol").parquet(path_backtest)

    trading_days = 252

    df_metrics = df_backtest.groupBy("symbol").agg(
        spark_max("cum_strategy_return").alias("total_strategy_return"),
        spark_max("cum_market_return").alias("total_market_return"),
        spark_max("cum_strategy_return_rank").alias("total_strategy_return_rank"),
        spark_count("trade_date").alias("num_trading_days"),
        spark_avg("strategy_return").alias("avg_daily_strategy_return"),
        spark_avg("market_return").alias("avg_daily_market_return"),
        spark_avg("strategy_return_rank").alias("avg_daily_strategy_return_rank"),
        spark_stddev("strategy_return").alias("stddev_strategy_return"),
        spark_stddev("market_return").alias("stddev_market_return"),
        spark_stddev("strategy_return_rank").alias("stddev_strategy_return_rank"),
        spark_min("drawdown").alias("max_drawdown"),
        spark_min("drawdown_rank").alias("max_drawdown_rank"),
        spark_sum("strategy_signal").alias("num_days_invested"),
        spark_sum("strategy_signal_rank").alias("num_days_invested_rank"),

        spark_sum(
            when((col("prediction") > 0) & (col(label_col) > 0), lit(1)).otherwise(lit(0))
        ).alias("true_positives"),
        spark_sum(
            when(col("prediction") > 0, lit(1)).otherwise(lit(0))
        ).alias("total_positive_predictions")
    ).withColumn(
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
        "model_precision_pct", when(
            col("total_positive_predictions") == 0, lit(0)
        ).otherwise(
            (col("true_positives") / col("total_positive_predictions")) * 100
        )
    ).withColumn(
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

    path_metrics = f"gs://{BUCKET_NAME}/gold/backtest_metrics/"
    print(f"[INFO] Export Métriques vers : {path_metrics}")
    df_metrics.select(metrics_cols).write.mode("overwrite").parquet(path_metrics)

    df_metrics.select(
        "symbol", "total_strategy_return", "total_market_return", "alpha",
        "sharpe_ratio", "sharpe_ratio_market", "max_drawdown", "model_precision_pct"
    ).show(truncate=False)

    print("\n--- Stratégie Ranking (Top 40%) ---")
    df_metrics.select(
        "symbol", "total_strategy_return_rank", "total_market_return", "alpha_rank",
        "sharpe_ratio_rank", "sharpe_ratio_market", "max_drawdown_rank"
    ).show(truncate=False)

    print("\n--- Génération des Prédictions Futures ---")
    max_date_row = df_raw.agg({"trade_date": "max"}).collect()[0]
    last_date = max_date_row[0]
    print(f"[INFO] Date de prédiction : {last_date}")

    df_future = df_raw.filter(col("trade_date") == last_date)
    df_future_clean = df_future.na.drop(subset=features)

    if df_future_clean.count() > 0:
        future_preds = best_model.transform(df_future_clean)

        final_future = future_preds.select(
            "symbol", "trade_date", "close", "prediction"
        ).withColumn(
            "conseil", when(col("prediction") > 0, "ACHETER").otherwise("ATTENDRE")
        ).withColumn(
            "prediction_pct", col("prediction") * 100
        )

        path_future = f"gs://{BUCKET_NAME}/gold/future_predictions/"
        print(f"[INFO] Export Prédictions Futures vers : {path_future}")
        final_future.write.mode("overwrite").parquet(path_future)

        print("[SUCCESS] Prédictions sauvegardées.")
    else:
        print("[WARN] Impossible de faire des prédictions (Features manquantes pour la dernière date).")


if __name__ == "__main__":
    run_ml_analysis()
