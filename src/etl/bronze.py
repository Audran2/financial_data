import requests
import json
import os
import logging
from datetime import datetime
from google.cloud import storage

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

# API keys and GCP configuration
TWELVEDATA_API_KEY = os.getenv("TWELVE_DATA_KEY")
FMP_API_KEY = os.getenv("FMP_KEY")
PROJECT_ID = "tribal-pillar-480213-i1"
BUCKET_NAME = "finance_datalake"

# Symbols to fetch data for
SYMBOLS = [
    "AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "AMD"
]

# Initialize GCS client
try:
    GCS_CLIENT = storage.Client(project=PROJECT_ID)
    logger.info(f"GCS client initialized successfully for project: {PROJECT_ID}")
except Exception as e:
    logger.error(f"Failed to initialize GCS client: {e}")
    raise


def upload_to_gcs(data, path_destination):
    """
    Upload JSON data to Google Cloud Storage

    Args:
        data: Dictionary or list to upload as JSON
        path_destination: Target path in GCS bucket
    """
    try:
        bucket = GCS_CLIENT.bucket(BUCKET_NAME)
        blob = bucket.blob(path_destination)

        blob.upload_from_string(
            data=json.dumps(data),
            content_type='application/json'
        )
        logger.success(f"Successfully uploaded to gs://{BUCKET_NAME}/{path_destination}")
    except Exception as error:
        logger.error(f"Failed to upload {path_destination}: {error}")
        raise


def blob_already_exists(path_destination):
    """
    Check if a file already exists in GCS

    Args:
        path_destination: Path to check in GCS bucket

    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        bucket = GCS_CLIENT.bucket(BUCKET_NAME)
        blob = bucket.blob(path_destination)
        return blob.exists()
    except Exception as error:
        logger.error(f"Error checking if blob exists at {path_destination}: {error}")
        return False


def fetch_twelvedata_prices(symbol):
    """
    Fetch historical price data from TwelveData API

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')

    Returns:
        dict: Price data or None if request failed
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 500,
        "apikey": TWELVEDATA_API_KEY
    }

    try:
        logger.debug(f"Fetching prices for {symbol} from TwelveData")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Check if API returned an error response
        if "status" in data and data["status"] == "error":
            logger.error(f"TwelveData API error for {symbol}: {data.get('message', 'Unknown error')}")
            return None

        logger.debug(f"Successfully fetched prices for {symbol}")
        return data
    except requests.exceptions.RequestException as error:
        logger.error(f"Network error fetching TwelveData prices for {symbol}: {error}")
        return None
    except Exception as error:
        logger.error(f"Unexpected error fetching TwelveData prices for {symbol}: {error}")
        return None


def fetch_fmp_ratios(symbol):
    """
    Fetch fundamental ratios from Financial Modeling Prep API

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')

    Returns:
        dict: Fundamental ratios data or None if request failed
    """
    url = f"https://financialmodelingprep.com/stable/ratios"
    params = {
        "symbol": symbol,
        "apikey": FMP_API_KEY,
        "period": "annual",
        "limit": 5
    }

    try:
        logger.debug(f"Fetching fundamentals for {symbol} from FMP")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Handle API error responses
        if isinstance(data, dict) and "Error Message" in data:
            logger.error(f"FMP API error for {symbol}: {data['Error Message']}")
            return None

        # Check if data is empty
        if not data:
            logger.warning(f"No fundamental data found for {symbol}")
            return None

        logger.debug(f"Successfully fetched fundamentals for {symbol}")
        return data
    except requests.exceptions.RequestException as error:
        logger.error(f"Network error fetching FMP ratios for {symbol}: {error}")
        return None
    except Exception as error:
        logger.error(f"Unexpected error fetching FMP ratios for {symbol}: {error}")
        return None


def main():
    """Main ETL process to ingest raw data into bronze layer"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info(f"Starting bronze layer ingestion for {today_str}")
    logger.info("=" * 60)

    # Track statistics for the run
    skipped_prices = 0
    skipped_fund = 0
    fetched_prices = 0
    fetched_fund = 0

    # Process each symbol
    for symbol in SYMBOLS:
        logger.info(f"Processing {symbol}...")

        # Handle price data
        path_prices = f"bronze/twelvedata/prices/dt={today_str}/{symbol}.json"

        if blob_already_exists(path_prices):
            logger.info(f"Prices already exist for {symbol}, skipping")
            skipped_prices += 1
        else:
            prices_data = fetch_twelvedata_prices(symbol)
            if prices_data:
                upload_to_gcs(prices_data, path_prices)
                fetched_prices += 1
                logger.info(f"Prices fetched for {symbol}")
            else:
                logger.warning(f"Failed to fetch prices for {symbol}")

        # Handle fundamental data
        path_fund = f"bronze/fmp/ratios/dt={today_str}/{symbol}.json"

        if blob_already_exists(path_fund):
            logger.info(f"Fundamentals already exist for {symbol}, skipping")
            skipped_fund += 1
        else:
            fund_data = fetch_fmp_ratios(symbol)
            if fund_data:
                upload_to_gcs(fund_data, path_fund)
                fetched_fund += 1
                logger.info(f"Fundamentals fetched for {symbol}")
            else:
                logger.warning(f"Failed to fetch fundamentals for {symbol}")

    logger.info("Ingestion Summary:")
    logger.info(f"  Prices       → {fetched_prices} fetched | {skipped_prices} skipped")
    logger.info(f"  Fundamentals → {fetched_fund} fetched | {skipped_fund} skipped")


if __name__ == "__main__":
    main()
