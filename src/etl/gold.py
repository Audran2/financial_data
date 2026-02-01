import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, avg, stddev, lag, when, abs as spark_abs,
    lead, row_number, desc, count, isnan, log, lit,
    pandas_udf
)
from pyspark.sql.types import DoubleType

# --- CONFIGURATION ---
BUCKET_NAME = os.getenv("BUCKET_NAME", "finance_datalake")
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_key.json")
JAR_PATH = "/tmp/gcs-connector.jar"

spark = SparkSession.builder \
    .appName("Finance_Gold_Advanced_Features") \
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


@pandas_udf(DoubleType())
def ema_udf(close: pd.Series) -> pd.Series:
    """
    Calcule une EMA sur la série complète du groupe (partitionBy symbol).
    Le span est passé via une colonne auxiliaire pour éviter les closures.
    On utilise span=12 par défaut ici ; on appelle cette UDF deux fois
    avec des colonnes d'input différentes pour avoir ema_12 et ema_26.
    """
    return close.ewm(span=12, adjust=False).mean()


@pandas_udf(DoubleType())
def ema_26_udf(close: pd.Series) -> pd.Series:
    return close.ewm(span=26, adjust=False).mean()


def calculate_technical_indicators(df):
    w_spec = Window.partitionBy("symbol").orderBy("trade_date")

    w_20 = w_spec.rowsBetween(-19, 0)
    df = df.withColumn("sma_20", avg("close").over(w_20)) \
           .withColumn("stddev_20", stddev("close").over(w_20)) \
           .withColumn("bollinger_upper", col("sma_20") + (col("stddev_20") * 2)) \
           .withColumn("bollinger_lower", col("sma_20") - (col("stddev_20") * 2))

    df = df.withColumn("ema_12", ema_udf("close").over(
        Window.partitionBy("symbol").orderBy("trade_date")
              .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    ))
    df = df.withColumn("ema_26", ema_26_udf("close").over(
        Window.partitionBy("symbol").orderBy("trade_date")
              .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    ))
    df = df.withColumn("macd_line", col("ema_12") - col("ema_26"))

    w_12 = w_spec.rowsBetween(-11, 0)
    w_26 = w_spec.rowsBetween(-25, 0)
    df = df.withColumn("sma_diff", avg("close").over(w_12) - avg("close").over(w_26))

    df = df.withColumn("diff", col("close") - lag("close", 1).over(w_spec))
    df = df.withColumn("gain", when(col("diff") > 0, col("diff")).otherwise(0)) \
           .withColumn("loss", when(col("diff") < 0, spark_abs(col("diff"))).otherwise(0))

    w_14 = w_spec.rowsBetween(-13, 0)
    df = df.withColumn("avg_gain", avg("gain").over(w_14)) \
           .withColumn("avg_loss", avg("loss").over(w_14))

    df = df.withColumn("rs", col("avg_gain") / when(col("avg_loss") == 0, 1).otherwise(col("avg_loss"))) \
           .withColumn("rsi_14", 100 - (100 / (1 + col("rs"))))

    df = df.drop("diff", "gain", "loss", "avg_gain", "avg_loss", "rs", "stddev_20")

    return df


def create_advanced_gold():
    try:
        df_prices = spark.read.parquet(f"gs://{BUCKET_NAME}/silver/prices/")
        df_fund = spark.read.parquet(f"gs://{BUCKET_NAME}/silver/fundamentals/")
        print(f"[INFO] Prix chargés : {df_prices.count()} | Fondamentaux chargés : {df_fund.count()}")
    except Exception as e:
        print(f"[ERREUR] Impossible de lire Silver : {e}")
        return

    df_fund_current = df_fund.filter(col("is_current") == True)

    fund_renamed = df_fund_current.select(
        col("symbol").alias("f_symbol"),
        col("report_date").alias("f_date"),
        col("pe_ratio"),
        col("debt_to_equity")
    )

    fund_broadcast = spark.createDataFrame(fund_renamed.collect()).hint("broadcast")

    cond = [
        df_prices.symbol == fund_broadcast.f_symbol,
        fund_broadcast.f_date <= df_prices.trade_date
    ]

    df_merged = df_prices.join(fund_broadcast, cond, "left")

    w_filter = Window.partitionBy("symbol", "trade_date").orderBy(desc("f_date"))
    df_joined = df_merged.withColumn("rn", row_number().over(w_filter)) \
                         .filter(col("rn") == 1) \
                         .drop("rn", "f_symbol", "f_date")

    df_calc = calculate_technical_indicators(df_joined)

    w_spec = Window.partitionBy("symbol").orderBy("trade_date")
    df_calc = df_calc.withColumn("daily_return", log(col("close") / lag("close", 1).over(w_spec)))

    df_calc = df_calc.withColumn("return_lag_1", lag("daily_return", 1).over(w_spec)) \
                     .withColumn("return_lag_2", lag("daily_return", 2).over(w_spec)) \
                     .withColumn("return_lag_3", lag("daily_return", 3).over(w_spec))

    w_vol = w_spec.rowsBetween(-9, 0)
    df_calc = df_calc.withColumn("volatility_10d", stddev("daily_return").over(w_vol))

    w_lead = Window.partitionBy("symbol").orderBy("trade_date")
    bb_width = col("bollinger_upper") - col("bollinger_lower")

    df_final_temp = df_calc.select(
        "symbol", "trade_date", "close", "volume",
        col("rsi_14"),
        col("macd_line"),
        col("sma_diff"),
        col("return_lag_1"), col("return_lag_2"), col("return_lag_3"),
        col("volatility_10d"),
        when(bb_width == 0, 0).otherwise(
            (col("close") - col("bollinger_lower")) / bb_width
        ).alias("pct_b"),
        col("pe_ratio"),
        col("debt_to_equity"),
        when(col("close") == 0, 0)
        .otherwise((lead("close", 1).over(w_lead) - col("close")) / col("close"))
        .alias("target_return_next_day"),
        lit("silver_prices + silver_fundamentals").alias("data_lineage")
    )

    fill_cols = ["pe_ratio", "debt_to_equity", "rsi_14", "macd_line", "sma_diff", "pct_b",
                 "return_lag_1", "return_lag_2", "return_lag_3", "volatility_10d"]
    df_filled = df_final_temp.na.fill(0, subset=fill_cols)

    df_filled.select(
        count(when(col("trade_date").isNull(), col("trade_date"))).alias("nb_null_dates"),
        count(when(col("close").isNull() | isnan(col("close")), col("close"))).alias("nb_null_close")
    ).show()

    df_final = df_filled.dropna(subset=["trade_date", "close"])

    rows = df_final.count()
    print(f"[INFO] Lignes à écrire : {rows}")

    if rows > 0:
        output_path = f"gs://{BUCKET_NAME}/gold/advanced_features/"
        df_final.write.mode("overwrite").partitionBy("symbol").parquet(output_path)
        print("[SUCCESS] Gold Features terminé.")
    else:
        print("[ALERTE] DataFrame vide !")


if __name__ == "__main__":
    create_advanced_gold()
