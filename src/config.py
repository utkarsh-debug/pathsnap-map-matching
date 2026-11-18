"""Project-wide paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Raw dataset location. Override with env var PATHSNAP_DATA if it lives elsewhere.
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT.parent / "Dataset"))
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MAP_DIR = RESULTS_DIR / "maps"
LOG_DIR = RESULTS_DIR / "logs"
ARTIFACT_DIR = ROOT / "api" / "artifacts"

PARTITIONS = ("train", "student_test", "evaluator_test")
# Trips come from different regions (e.g. Germany, Ecuador): a UTM zone is chosen per trip (src/io.py).
SEED = 42

for _d in (PROCESSED_DIR, FIG_DIR, MAP_DIR, LOG_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
