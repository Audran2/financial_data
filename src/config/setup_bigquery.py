import os
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ID = "tribal-pillar-480213-i1"
DATASET_ID = "finance_analytics"
BUCKET_NAME = os.getenv("BUCKET_NAME", "finance_datalake")

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(project=PROJECT_ID, credentials=credentials)


def create_or_replace_external_table(table_name, gcs_uri, schema):
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    external_config = bigquery.ExternalConfig("PARQUET")
    external_config.source_uris = [gcs_uri]
    external_config.schema = schema

    table = bigquery.Table(table_id, schema=schema)
    table.external_config = external_config

    client.create_table(table, exists_ok=True)
    print(f"[OK] Table '{table_name}' → {gcs_uri}")


def main():
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
    try:
        client.get_dataset(dataset_ref)
        print(f"[INFO] Dataset '{DATASET_ID}' existe déjà")
    except Exception:
        client.create_dataset(dataset_ref)
        print(f"[INFO] Dataset '{DATASET_ID}' créé")

    schema_backtest = [
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("trade_date", "DATE"),
        bigquery.SchemaField("close", "FLOAT"),
        bigquery.SchemaField("target_return_next_day", "FLOAT"),
        bigquery.SchemaField("prediction", "FLOAT"),
        bigquery.SchemaField("strategy_signal", "INTEGER"),
        bigquery.SchemaField("strategy_return", "FLOAT"),
        bigquery.SchemaField("market_return", "FLOAT"),
        bigquery.SchemaField("cum_strategy_return", "FLOAT"),
        bigquery.SchemaField("cum_market_return", "FLOAT"),
        bigquery.SchemaField("strategy_wealth", "FLOAT"),
        bigquery.SchemaField("market_wealth", "FLOAT"),
        bigquery.SchemaField("drawdown", "FLOAT"),
    ]

    create_or_replace_external_table(
        table_name="backtest_results",
        gcs_uri=f"gs://{BUCKET_NAME}/gold/backtest_results/*.parquet",
        schema=schema_backtest
    )

    schema_metrics = [
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("total_strategy_return", "FLOAT"),
        bigquery.SchemaField("total_market_return", "FLOAT"),
        bigquery.SchemaField("alpha", "FLOAT"),
        bigquery.SchemaField("num_trading_days", "INTEGER"),
        bigquery.SchemaField("num_days_invested", "INTEGER"),
        bigquery.SchemaField("avg_daily_strategy_return", "FLOAT"),
        bigquery.SchemaField("avg_daily_market_return", "FLOAT"),
        bigquery.SchemaField("stddev_strategy_return", "FLOAT"),
        bigquery.SchemaField("stddev_market_return", "FLOAT"),
        bigquery.SchemaField("sharpe_ratio", "FLOAT"),
        bigquery.SchemaField("sharpe_ratio_market", "FLOAT"),
        bigquery.SchemaField("max_drawdown", "FLOAT"),
        bigquery.SchemaField("model_precision_pct", "FLOAT"),
    ]

    create_or_replace_external_table(
        table_name="backtest_metrics",
        gcs_uri=f"gs://{BUCKET_NAME}/gold/backtest_metrics/*.parquet",
        schema=schema_metrics
    )

    schema_future = [
        bigquery.SchemaField("symbol", "STRING"),
        bigquery.SchemaField("trade_date", "DATE"),
        bigquery.SchemaField("close", "FLOAT"),
        bigquery.SchemaField("prediction", "FLOAT"),
        bigquery.SchemaField("conseil", "STRING"),
        bigquery.SchemaField("prediction_pct", "FLOAT"),
    ]

    create_or_replace_external_table(
        table_name="future_predictions",
        gcs_uri=f"gs://{BUCKET_NAME}/gold/future_predictions/*.parquet",
        schema=schema_future
    )

    print("\n[SUCCESS] Les trois tables BigQuery sont prêtes pour Looker Studio.")


if __name__ == "__main__":
    main()
