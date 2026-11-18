"""Export the inference artifacts the API loads. Run: python -m api.export_artifacts [model_file_stem]
Copies the trained model into api/artifacts/ and writes pipeline_config.json (features, candidate settings,
model identity and, for a tuned model, its hyper-parameters). Default model: <selected>_tuned if present, else the
untuned selected model."""
import json
import shutil
import sys

from src.config import ARTIFACT_DIR, PROCESSED_DIR, RESULTS_DIR
from src.features import FEATURES

K, RADIUS = 15, 80.0     # candidate generation settings used for training (src/candidates.py defaults)


def main(stem=None):
    sel = json.loads((RESULTS_DIR / "model_selection.json").read_text())["selected_model"]
    mdir = PROCESSED_DIR / "models"
    stem = stem or (f"{sel}_tuned" if (mdir / f"{sel}_tuned.joblib").exists() else sel)
    src = mdir / f"{stem}.joblib"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, ARTIFACT_DIR / "model.joblib")
    cfg = {"model_name": stem, "model_file": "model.joblib", "features": FEATURES,
           "candidates": {"k": K, "radius_m": RADIUS}, "seed": 42}
    bo = RESULTS_DIR / "bo_best_params.json"
    if stem.endswith("_tuned") and bo.exists():
        cfg["hyperparameters"] = json.loads(bo.read_text())["best_params"]
    (ARTIFACT_DIR / "pipeline_config.json").write_text(json.dumps(cfg, indent=2))
    print("exported", stem, "->", ARTIFACT_DIR)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
