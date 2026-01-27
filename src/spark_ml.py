import os
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, when

# --- CONFIGURATION ---
BUCKET_NAME = os.getenv("BUCKET_NAME", "finance_datalake")
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_key.json")
JAR_PATH = "/tmp/gcs-connector.jar"

spark = SparkSession.builder \
    .appName("Finance_Pro_ML_Backtest") \
    .config("spark.jars", JAR_PATH) \
    .config("spark.driver.extraClassPath", JAR_PATH) \
    .config("spark.executor.extraClassPath", JAR_PATH) \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
    .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", KEY_PATH) \
    .getOrCreate()


def run_ml_analysis():
    print("--- Chargement des données Gold ---")
    try:
        df_raw = spark.read.parquet(f"gs://{BUCKET_NAME}/gold/advanced_features/")
    except AnalysisException:
        print(f"[ERREUR] Dossier Gold vide ou introuvable.")
        return

    features = [
        "rsi_14", "macd_line", "pct_b", "pe_ratio", "debt_to_equity", "volume",
        "return_lag_1", "return_lag_2", "return_lag_3", "volatility_10d"
    ]
    label_col = "target_return_next_day"

    print("Préparation des données pour l'entraînement...")

    df_train_ready = df_raw.na.drop(subset=[label_col] + features)

    row_count = df_train_ready.count()
    print(f"Données d'entraînement disponibles : {row_count} lignes")

    if row_count == 0:
        print("Erreur: Pas assez de données pour entraîner.")
        return

    split_date = "2024-06-01"
    train = df_train_ready.filter(col("trade_date") < split_date)
    test = df_train_ready.filter(col("trade_date") >= split_date)

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

    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=parameter_grid,
                        evaluator=evaluator,
                        numFolds=3)

    model = cv.fit(train)
    best_model = model.bestModel
    print(f"RMSE sur train: {min(model.avgMetrics)}")

    print("Génération du Backtest...")
    predictions = best_model.transform(test)

    analysis = predictions.withColumn(
        "strategy_signal", when(col("prediction") > 0, 1).otherwise(0)
    ).withColumn(
        "strategy_return", col("strategy_signal") * col(label_col)
    ).withColumn(
        "market_return", col(label_col)
    )

    output_cols = ["symbol", "trade_date", "close", label_col, "prediction", "strategy_return",
                   "market_return"] + features
    output_viz = analysis.select(output_cols)

    path_backtest = f"gs://{BUCKET_NAME}/gold/backtest_results/"
    print(f"Export Backtest vers : {path_backtest}")
    output_viz.write.mode("overwrite").parquet(path_backtest)

    print("\n--- Génération des Prédictions Futures ---")

    max_date_row = df_raw.agg({"trade_date": "max"}).collect()[0]
    last_date = max_date_row[0]
    print(f"Date de prédiction : {last_date}")

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
        print(f"Export Prédictions Futures vers : {path_future}")

        final_future.write.mode("overwrite").parquet(path_future)

        print("[SUCCESS] Prédictions sauvegardées.")
    else:
        print("[WARN] Impossible de faire des prédictions (Features manquantes pour la dernière date).")


if __name__ == "__main__":
    run_ml_analysis()