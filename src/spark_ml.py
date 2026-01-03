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

    print("Nettoyage des données (Drop Nulls)...")
    df = df_raw.na.drop(subset=[label_col] + features)

    row_count = df.count()
    print(f"Données propres prêtes pour ML : {row_count} lignes")

    if row_count == 0:
        print("Erreur: Plus de données après nettoyage !")
        return

    split_date = "2024-06-01"
    train = df.filter(col("trade_date") < split_date)
    test = df.filter(col("trade_date") >= split_date)

    if train.count() == 0 or test.count() == 0:
        print("Erreur: Train ou Test set vide. Vérifie la date de split.")
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

    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=parameter_grid,
                        evaluator=evaluator,
                        numFolds=3)

    model = cv.fit(train)
    best_model = model.bestModel
    print(f"Meilleur modèle trouvé (RMSE sur train): {min(model.avgMetrics)}")

    gbt_model = best_model.stages[-1]
    print("Importance des Features : ", gbt_model.featureImportances)

    print("Prédictions sur le Test Set...")
    predictions = best_model.transform(test)

    analysis = predictions.withColumn(
        "strategy_signal", when(col("prediction") > 0, 1).otherwise(0)
    ).withColumn(
        "strategy_return", col("strategy_signal") * col(label_col)
    ).withColumn(
        "market_return", col(label_col)
    )

    output_viz = analysis.select(
        "symbol", "trade_date", "close",
        label_col, "prediction",
        "strategy_return", "market_return"
    )

    output_path = f"gs://{BUCKET_NAME}/gold/backtest_results/"
    print(f"Export vers : {output_path}")
    output_viz.write.mode("overwrite").parquet(output_path)
    print("[SUCCESS] Job ML terminé.")


if __name__ == "__main__":
    run_ml_analysis()