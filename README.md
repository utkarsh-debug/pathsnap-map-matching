# PathSnap: map-matching with ML + HMM/Viterbi
(README is completed at the end; see stages below as they are added.)

## Setup (Windows, Laptop 1 = API)
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
Dataset is read from ../Dataset (override with env var PATHSNAP_DATA).

## Stage 1: EDA
python -m src.eda   -> results/inventory.csv, results/network_stats.csv, results/data_quality.txt, results/figures/
