from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, when, sum as spark_sum, avg

# --- CONFIGURATION ---
BUCKET_NAME = "xxxxxxx"
spark = SparkSession.builder.appName("Finance_Pro_ML_Backtest").getOrCreate()


def run_ml_analysis():
    df = spark.read.parquet(f"gs://{BUCKET_NAME}/gold/advanced_features/")

    split_date = "2024-06-01"
    train = df.filter(col("trade_date") < split_date)
    test = df.filter(col("trade_date") >= split_date)

    features = ["rsi_14", "macd_line", "pct_b", "pe_ratio", "debt_to_equity", "volume"]

    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    gbt = GBTRegressor(featuresCol="features", labelCol="target_return_next_day", seed=42)
    pipeline = Pipeline(stages=[assembler, scaler, gbt])

    print("Début du Grid Search...")

    parameter_grid = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [5, 10]) \
        .addGrid(gbt.maxIter, [20]) \
        .build()

    evaluator = RegressionEvaluator(labelCol="target_return_next_day", metricName="rmse")

    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=parameter_grid,
                        evaluator=evaluator,
                        numFolds=3)

    model = cv.fit(train)
    best_model = model.bestModel
    print(f"Meilleur modèle trouvé (RMSE sur train): {min(model.avgMetrics)}")

    predictions = best_model.transform(test)


    analysis = predictions.withColumn(
        "strategy_signal", when(col("prediction") > 0.001, 1).otherwise(0)
    ).withColumn(
        "strategy_return", col("strategy_signal") * col("target_return_next_day")
    ).withColumn(
        "market_return", col("target_return_next_day")
    )

    results = analysis.groupBy("symbol").agg(
        spark_sum("market_return").alias("total_market_return"),
        spark_sum("strategy_return").alias("total_model_return"),
        avg("prediction").alias("avg_predicted_return")
    )

    print("--- RÉSULTATS DU BACKTEST ---")
    results.show()

    output_viz = analysis.select(
        "symbol", "trade_date", "close",
        "target_return_next_day", "prediction",
        "strategy_return", "market_return"
    )

    output_viz.write.mode("overwrite").parquet(f"gs://{BUCKET_NAME}/gold/backtest_results/")
    print("Résultats exportés pour visualisation.")


if __name__ == "__main__":
    run_ml_analysis()