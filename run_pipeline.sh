#!/bin/bash

echo "--- [INFO] Démarrage du pipeline ---"

python3 ./src/etl/bronze.py

python3 ./src/etl/silver.

python3 ./src/etl/gold.py

python3 .src/spark_ml.py

echo "--- [INFO] Pipeline terminé ---"