import requests
import json
import os
import logging
from datetime import datetime
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TWELVEDATA_API_KEY = os.getenv("TWELVE_DATA_KEY")
FMP_API_KEY = os.getenv("FMP_KEY")
PROJECT_ID = "tribal-pillar-480213-i1"
BUCKET_NAME = "finance_datalake"

SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN"]

try:
    GCS_CLIENT = storage.Client(project=PROJECT_ID)
except Exception as e:
    logging.warning("Impossible d'initialiser le client GCS localement sans authentification. "
                    "Assure-toi d'avoir fait 'gcloud auth application-default login'.")


def upload_to_gcs(data, path_destination):
    try:
        bucket = GCS_CLIENT.bucket(BUCKET_NAME)
        blob = bucket.blob(path_destination)

        blob.upload_from_string(
            data=json.dumps(data),
            content_type='application/json'
        )
        logging.info(f"[SUCCESS] Uploaded: gs://{BUCKET_NAME}/{path_destination}")
    except Exception as e:
        logging.error(f"[ERROR] Upload GCS failed pour {path_destination}: {e}")


def fetch_twelvedata_prices(symbol):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 500,
        "apikey": TWELVEDATA_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "status" in data and data["status"] == "error":
            logging.error(f"TwelveData API Error pour {symbol}: {data['message']}")
            return None

        return data
    except Exception as e:
        logging.error(f"Erreur connexion TwelveData pour {symbol}: {e}")
        return None


def fetch_fmp_ratios(symbol):
    url = f"https://financialmodelingprep.com/stable/key-metrics-ttm"
    params = {
        "symbol": symbol,
        "apikey": FMP_API_KEY,
        "limit": 1
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and "Error Message" in data:
            logging.error(f"FMP API Error pour {symbol}: {data['Error Message']}")
            return None
        if not data:
            logging.warning(f"FMP: Aucune donnée trouvée pour {symbol}")
            return None

        return data
    except Exception as e:
        logging.error(f"Erreur connexion FMP pour {symbol}: {e}")
        return None


def main():
    today_str = datetime.now().strftime("%Y-%m-%d")
    logging.info(f"--- {today_str} - Début de l'ingestion ---")

    for symbol in SYMBOLS:
        prices_data = fetch_twelvedata_prices(symbol)
        if prices_data:
            path_prices = f"bronze/twelvedata/prices/dt={today_str}/{symbol}.json"
            upload_to_gcs(prices_data, path_prices)

        fund_data = fetch_fmp_ratios(symbol)
        if fund_data:
            path_fund = f"bronze/fmp/ratios/dt={today_str}/{symbol}.json"
            upload_to_gcs(fund_data, path_fund)


if __name__ == "__main__":
    main()