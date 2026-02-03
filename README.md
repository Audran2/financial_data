# Financial Data Pipeline

Pipeline quotidien de données financières (medallion architecture Bronze / Silver / Gold). Ingère des cours boursiers et des ratios fondamentaux, les nettoie et les enrichit avec Spark, puis entraîne un modèle de machine learning pour prédire le rendement du lendemain et tester une stratégie de trading. Les résultats sont exposés dans BigQuery pour être utilisés par Looker Studio.

## Architecture

```
TwelveData API ─┐
                 ├─► bronze.py ──► GCS (JSON brut, partitionné par date)
FMP API ─────────┘

GCS bronze ──► silver.py (PySpark) ──► GCS silver (Parquet)
   - prices        : validation qualité (OHLC, volumes, dates)
   - fundamentals   : SCD Type 2 (historisation des changements)

GCS silver ──► gold.py (PySpark) ──► GCS gold/advanced_features (Parquet)
   - jointure point-in-time prix + fondamentaux (anti-lookahead bias)
   - indicateurs techniques : SMA, Bollinger Bands, EMA 12/26, MACD, RSI 14
   - features ML : lags de rendement, volatilité 10j, target = rendement J+1

GCS gold ──► spark_ml.py (PySpark ML) ──► GCS gold/{backtest_results,backtest_metrics,future_predictions}
   - GBTRegressor + CrossValidator (split temporel train/test)
   - backtest : stratégie "signal" vs stratégie "ranking" (top 6/jour)
   - métriques : Sharpe ratio, alpha, max drawdown, précision du modèle
   - prédictions futures + recommandation BUY/HOLD

GCS gold ──► setup_bigquery.py ──► BigQuery (tables externes sur les Parquet) ──► Looker Studio
```

## Stack technique

- **Ingestion** : Python, `requests`, API [TwelveData](https://twelvedata.com/) (prix) et [Financial Modeling Prep](https://financialmodelingprep.com/) (ratios fondamentaux)
- **Stockage** : Google Cloud Storage (data lake, bucket `finance_datalake`)
- **Transformation** : PySpark (via le connecteur GCS Hadoop, JAR téléchargé en CI)
- **ML** : Spark ML (`GBTRegressor`, `StandardScaler`, `CrossValidator`)
- **Entrepôt / BI** : Google BigQuery (tables externes) → Looker Studio
- **Orchestration** : GitHub Actions (`.github/workflows/daily_pipeline.yml`), cron `0 23 * * 1-5` (23h UTC, jours ouvrés)
- **Symboles suivis** : `AAPL, GOOGL, MSFT, TSLA, NVDA, AMZN, META, AMD`

## Structure du repo

```
src/
├── etl/
│   ├── bronze.py          # Ingestion API → GCS (JSON brut)
│   ├── silver.py          # Nettoyage + SCD2 (PySpark) → GCS Parquet
│   └── gold.py            # Feature engineering ML (PySpark) → GCS Parquet
├── spark_ml.py            # Entraînement du modèle + backtest → GCS Parquet
└── config/
    └── setup_bigquery.py  # Création des tables externes BigQuery
```

## Variables d'environnement

Toutes les variables ci-dessous sont requises (aucune valeur par défaut dans le code). Le workflow CI les fournit en CI, et il faut les exporter soi-même en local.

| Variable | Utilisée dans | Description |
|---|---|---|
| `TWELVE_DATA_KEY` | `bronze.py` | Clé API TwelveData pour les séries de prix |
| `FMP_KEY` | `bronze.py` | Clé API Financial Modeling Prep pour les ratios fondamentaux |
| `PROJECT_ID` | `bronze.py`, `setup_bigquery.py` | ID du projet GCP |
| `DATASET_ID` | `setup_bigquery.py` | Dataset BigQuery cible |
| `BUCKET_NAME` | `bronze.py`, `silver.py`, `gold.py`, `spark_ml.py`, `setup_bigquery.py` | Bucket GCS du data lake |
| `GOOGLE_APPLICATION_CREDENTIALS` | `silver.py`, `gold.py`, `spark_ml.py`, `setup_bigquery.py` | Chemin du fichier JSON de service account GCP |
| `SPARK_JARS` | `silver.py`, `gold.py`, `spark_ml.py` | Chemin du JAR du connecteur GCS pour Spark |
| `SYMBOLS` | `bronze.py` | Liste de tickers à ingérer, séparés par des virgules (ex. `AAPL,GOOGL,MSFT`) |
| `GCP_SA_KEY` | secret GitHub Actions uniquement | Clé de service account GCP encodée en base64, utilisée par le workflow pour s'authentifier et générer `/tmp/gcp_key.json` |

## Setup local

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export TWELVE_DATA_KEY="..."
export FMP_KEY="..."
export GOOGLE_APPLICATION_CREDENTIALS="/chemin/vers/service-account.json"
export SPARK_JARS="/chemin/vers/gcs-connector.jar"
export BUCKET_NAME="finance_datalake"
export PROJECT_ID="tribal-pillar-480213-i1"
export DATASET_ID="finance_analytics"
export SYMBOLS="AAPL,GOOGL,MSFT,TSLA,NVDA,AMZN,META,AMD"

python src/etl/bronze.py
python src/etl/silver.py
python src/etl/gold.py
python src/spark_ml.py
python src/config/setup_bigquery.py
```

Le connecteur GCS pour Spark (`gcs-connector-hadoop3-*.jar`) doit être placé en local sur `/tmp/gcs-connector.jar` (téléchargé automatiquement en CI, voir le workflow).

## CI/CD

Le workflow `.github/workflows/daily_pipeline.yml` s'exécute automatiquement en semaine à 23h UTC (ou manuellement via `workflow_dispatch`) et lance dans l'ordre : `bronze.py` → `silver.py` → `gold.py` → `spark_ml.py`.

Secrets GitHub requis (Settings → Secrets and variables → Actions) :
- `GCP_SA_KEY`: clé de service GCP en base64
- `TWELVE_DATA_KEY`
- `FMP_KEY`

Note : `setup_bigquery.py` n'est **pas** appelé par le workflow. C'est un script à lancer manuellement (ou à ajouter au pipeline) après une mise à jour de schéma.
