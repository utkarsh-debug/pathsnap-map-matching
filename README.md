# PathSnap Map Matching

PathSnap is a Python map-matching project that converts noisy GPS observations into likely road-network arcs. It:

- builds candidate road arcs and point/edge features for each GPS observation;
- compares six classifier families with trip-grouped cross-validation;
- tunes the selected XGBoost model;
- adds a temporal HMM/Viterbi decoder to improve sequence continuity;
- exposes predictions through a FastAPI service; and
- provides a browser UI for selecting a trip and displaying the predicted route.

The system has no JavaScript build step. The UI is plain HTML, CSS, and JavaScript, and serves predictions from the API.

## Repository layout

| Path | Purpose |
|---|---|
| `src/` | raw trip readers, candidate generation, feature engineering, EDA, and dataset construction |
| `training/` | model zoo, trip-grouped training/evaluation, Bayesian tuning, model selection, and sensitivity analysis |
| `hmm/` | HMM/Viterbi decoder, tuning, tests, and result plots |
| `viz/` | model comparison and experiment plots |
| `docs/` | evidence-generation scripts used for report material |
| `api/` | FastAPI server, inference pipeline, scoring, artifact export, and API tests |
| `ui/` | browser interface served with the Python HTTP server |
| `results/` | experiment tables, selected parameters, plots, logs, screenshots, and model-selection records |
| `data/processed/` | generated feature tables, model predictions, trained models, and HMM outputs; ignored by Git |

## Data and file format

By default, `src.config.DATA_DIR` reads `../Dataset` relative to the repository. Set `PATHSNAP_DATA` to use another dataset root.

The data is partitioned into:

- `train/` - 80 trips used for development and cross-validation;
- `student_test/` - 10 held-out trips used for development-stage evaluation;
- `evaluator_test/` - reserved evaluation trips.

Each trip is a folder containing:

- `<trip_id>.track` - longitude, latitude, and timestamp;
- `<trip_id>.nodes` - longitude and latitude for each network node;
- `<trip_id>.arcs` - directed road arc endpoints;
- `<trip_id>.route` - reference arc sequence, when labels are available.

The API loads neither the `.route` file nor its labels during prediction. Reference-only scoring occurs server-side after a prediction is generated.

## Setup

This project is designed for Windows, but Linux commands are included for the browser UI.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Place the raw dataset at `../Dataset`, or override the location:

```powershell
$env:PATHSNAP_DATA="C:\path\to\Dataset"
```

## Reproduce the pipeline

Run commands from the repository root.

### 1. Explore the raw data

```powershell
python -m src.eda
python -m src.check_candidates train 4
```

EDA writes inventory and quality summaries to `results/`. Candidate checks verify candidate generation and reference-label coverage on sample trips.

### 2. Build candidate features

```powershell
python -m src.build_dataset
```

This creates processed tables under `data/processed/` and writes candidate-quality summaries to `results/candidate_stats.csv` and `results/candidate_summary.csv`.

Candidate generation uses up to 15 arcs within an 80 metre radius. The training candidate set contained a reference label for 89.44% of observations; the student-test set contained a reference label for 99.88%.

### 3. Train and compare models

```powershell
python -m training.train_models
python -m training.select_model
```

The comparison uses six models:

- Logistic Regression;
- k-nearest neighbours;
- Random Forest;
- LightGBM;
- XGBoost; and
- MLP.

Evaluation uses five trip-grouped folds over all 80 training trips. XGBoost was selected by mean cross-validation observation-level Top-1 accuracy, with observation log loss as the tie-break metric.

### 4. Tune the selected model

```powershell
python -m training.bayes_opt 40
```

Optuna uses the same trip-grouped folds and training partition. It writes Bayesian-search records and a tuned XGBoost model to `results/` and `data/processed/models/`.

### 5. Run the HMM/Viterbi experiment

```powershell
python -m unittest hmm.test_viterbi
python -m hmm.run_hmm 30 XGBoost_tuned
```

The HMM tunes `alpha`, `beta`, unreachable-transition and backtracking penalties, and maximum time gap using 20 sampled training trips and out-of-fold probabilities. The 10 student-test trips are used only after tuning for the development comparison.

Copy tuned HMM settings to the API:

```powershell
Copy-Item results/hmm_params.json api/artifacts/hmm_params.json
python -m api.export_artifacts XGBoost_tuned
```

## Experiment results

| Result | Development observation |
|---|---|
| Selected untuned model | XGBoost |
| Untuned 5-fold CV Top-1 | 76.82% |
| Tuned 5-fold CV Top-1 | 77.07% |
| Untuned student-test Top-1 | 86.97% |
| Tuned student-test Top-1 | 86.95% |
| Classifier-only student-test Top-1 | 86.95% |
| HMM/Viterbi student-test Top-1 | 87.54% |
| Classifier-only student-test continuity | 94.75% |
| HMM/Viterbi student-test continuity | 96.49% |

The HMM also improved mean arc-set F1 from 89.51% to 91.40% and mean sequence similarity from 75.23% to 76.16% on student-test trips.

These are development-stage measurements. Sequence similarity compares consecutive-deduplicated per-observation arc sequences against aligned observation labels; it is not a claim of exact recovery of every raw route arc.

Key experiment outputs:

- `results/model_comparison.csv` - model comparison;
- `results/model_selection.json` - selected model and rankings;
- `results/bo_trials.csv` - Bayesian tuning trials;
- `results/tuned_vs_untuned.csv` - tuned model comparison;
- `results/hmm_comparison.csv` - classifier-only versus HMM/Viterbi;
- `results/hmm_comparison_per_trip.csv` - per-trip improvements and regressions;
- `results/hmm_params.json` - deployed HMM settings;
- `results/figures/` - EDA and experiment plots.

## Run the API

The API serves `../Dataset/student_test` by default. Point it at the final evaluation folder with:

```powershell
$env:PATHSNAP_API_DATA="C:\path\to\evaluator_test"
```

Start the server:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for interactive Swagger documentation.

| Endpoint | Response |
|---|---|
| `GET /health` | API status, model name, and `hmm_viterbi` or `classifier_only` |
| `GET /trips` | available trip IDs and GPS point counts |
| `POST /predict` | GPS points, predicted arcs, predicted route polyline, timings, and metrics |

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" -Body '{"trip_id":"00000009"}'
```

Run the development-data self-test while the API is running:

```powershell
python -m api.test_api
```

The API exports no reference route or reference labels. If `api/artifacts/hmm_params.json` or `hmm/viterbi.py` is unavailable, it falls back to classifier-only mode and reports that in `/health` and `/predict`.

## Run the browser UI

From the repository root:

```powershell
python -m http.server --bind 0.0.0.0 --directory ui 8500
```

Open `http://localhost:8500` in a browser.

Enter the API address, for example `http://192.168.1.10:8000`, choose a trip, and run a prediction. The address is saved in browser local storage.

## Two-laptop demonstration

On Laptop 1:

1. activate the virtual environment;
2. set `PATHSNAP_API_DATA` if evaluation trips are used;
3. start the API on port `8000`;
4. find the IPv4 address with `ipconfig`; and
5. allow inbound TCP port `8000` through Windows Firewall.

```powershell
New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound `
  -LocalPort 8000 -Protocol TCP -Action Allow
```

Set both laptops to the same network and use a private Wi-Fi profile.

On Laptop 2:

1. start the UI;
2. enter `http://<laptop-1-ip>:8000`; and
3. run `GET /health`, select a trip, and run `POST /predict`.

Keep the API console visible during the demonstration: it logs client addresses, request status, decoded trip, metrics, and timing.

## Validation and caveats

- Two of the 80 training trips have poor reference-route/GPS agreement and are excluded only from robustness-check evaluations, not from the primary 80-trip CV ranking.
- The API only validates trips found under `PATHSNAP_API_DATA`; unknown IDs return HTTP 404.
- Training and student-test partitions are development partitions. Do not use evaluator-trip labels or routes for tuning or testing.
- HMM jumps can remain disconnected after map-matching, so route continuity is reported from predictions rather than assumed.
- The generated requirements file pins the environment; the project does not define a separate package metadata or build system.
