# PathSnap Map Matching

PathSnap is a Python map-matching project that converts noisy GPS observations into likely road-network arcs. It:

- builds candidate road arcs and point/edge features for each GPS observation;
- compares six classifier families with trip-grouped cross-validation;
- tunes the selected XGBoost model with Bayesian optimisation (Optuna);
- adds a temporal HMM/Viterbi decoder to improve sequence continuity;
- exposes predictions through a FastAPI service;
- provides a browser UI for selecting a trip and displaying the predicted route; and
- generates interactive Folium maps comparing GPS, road network, reference route and predicted route.

The system has no JavaScript build step. The UI is plain HTML, CSS, and JavaScript, and serves predictions from the API.

**Team:** [Team name] | **Members:** [Names]

## Pipeline at a glance

```
raw trip files -> candidate arcs (<=15 within 80 m) -> 19 features per (observation, arc)
   -> classifier (tuned XGBoost) -> arc probabilities
   -> HMM/Viterbi decoding -> predicted arc per observation -> stitched route
   -> FastAPI (/predict) -> browser UI and Folium maps
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/` | raw trip readers, candidate generation, feature engineering, EDA, and dataset construction |
| `training/` | model zoo, trip-grouped training/evaluation, Bayesian tuning, model selection, and sensitivity analysis |
| `hmm/` | HMM/Viterbi decoder, parameter tuning, unit tests, and result plots |
| `viz/` | Folium map generation (`folium_map.py`) and model-comparison plots (`plots_models.py`) |
| `docs/` | evidence-generation scripts and fact sheets used for report material |
| `api/` | FastAPI server, inference pipeline, server-side scoring, artifact export, and API tests |
| `api/artifacts/` | **saved model weights and configuration used by the API**: `model.joblib` (tuned XGBoost, about 2.8 MB), `pipeline_config.json`, `hmm_params.json` |
| `ui/` | browser interface served with the Python HTTP server |
| `results/` | experiment tables, selected parameters, plots, logs, screenshots, maps, and model-selection records |
| `results/figures/` | EDA, model-comparison, Bayesian-optimisation and HMM plots |
| `results/histories/` | real epoch/iteration training histories (XGBoost, LightGBM, MLP) |
| `results/logs/` | training, Bayesian-optimisation, HMM and API logs |
| `results/maps/` | generated interactive Folium maps (HTML) |
| `results/screenshots/` | UI and API-console evidence images (see the README in that folder) |
| `data/processed/` | generated feature tables, model predictions, trained models, and HMM outputs; ignored by Git |

The raw dataset is not stored in this repository. No passwords, API keys or other secrets are committed (`.env` is git-ignored).

## Data and file format

By default, `src.config.DATA_DIR` reads `../Dataset` relative to the repository. Set `PATHSNAP_DATA` to use another dataset root.

The data is partitioned into:

- `train/` - 80 trips used for development and cross-validation;
- `student_test/` - 10 held-out trips used for development-stage evaluation;
- `evaluator_test/` - reserved evaluation trips (never used for training, tuning or decisions).

Each trip is a folder containing:

- `<trip_id>.track` - longitude, latitude, and timestamp;
- `<trip_id>.nodes` - longitude and latitude for each network node;
- `<trip_id>.arcs` - directed road arc endpoints;
- `<trip_id>.route` - reference arc sequence, when labels are available.

Arc IDs are row indices of each trip's own `.arcs` file, so a road segment is identified by (trip, arc ID).

The API loads neither the `.route` file nor its labels during prediction. Reference-only scoring occurs server-side after a prediction is generated.

## Setup

This project is designed for Windows, with Linux/macOS equivalents shown where they differ. Use a recent Python 3 interpreter [state the exact version the team used].

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
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

```bash
export PATHSNAP_DATA=/path/to/Dataset
```

All experiments use random seed 42.

## Reproduce the pipeline

Run commands from the repository root.

### 1. Explore the raw data

```powershell
python -m src.eda
python -m src.check_candidates train 4
```

EDA writes inventory, network and data-quality summaries (`results/inventory.csv`, `results/network_stats.csv`, `results/data_quality.txt`) and EDA plots (`results/figures/eda_*.png`). Candidate checks verify candidate generation and reference-label coverage on sample trips.

Findings recorded in `results/data_quality.txt`: no missing values, no non-monotonic timestamps and no isolated nodes; 94.3% of time steps are exactly 1 s; two training trips (78 and 90) contain very large time gaps.

### 2. Build candidate features

```powershell
python -m src.build_dataset
```

This creates processed tables under `data/processed/` and writes candidate-quality summaries to `results/candidate_stats.csv` and `results/candidate_summary.csv`.

Candidate generation uses up to 15 arcs within an 80 metre radius. The training candidate set contained a reference label for 89.44% of observations; the student-test set contained a reference label for 99.88%. Each row (observation, candidate arc) has 19 features covering distance, position on the arc, arc geometry and connectivity, heading and motion, and temporal/sequence context (`src/features.py`).

### 3. Train and compare models

```powershell
python -m training.train_models
python -m training.select_model
python -m viz.plots_models
```

The comparison uses six models:

- Logistic Regression;
- k-nearest neighbours;
- Random Forest;
- LightGBM;
- XGBoost; and
- MLP.

Evaluation uses five trip-grouped folds over all 80 training trips, then a refit on all 80 trips and one evaluation on the 10 student-test trips. XGBoost was selected by mean cross-validation observation-level Top-1 accuracy, with observation log loss as the tie-break metric.

`python -m viz.plots_models` regenerates the comparison plots (cross-validation F1/Top-1, student-test Top-1/F1, log loss, confusion matrices, training curves, overfitting diagnostics) from the recorded results. Training curves exist only for XGBoost, LightGBM and the MLP, which produce training histories.

Optional robustness check for the two training trips whose reference route does not match their GPS track:

```powershell
python -m training.sensitivity_exclusion
```

This writes `results/sensitivity_exclusion.csv`.

### 4. Tune the selected model

```powershell
python -m training.bayes_opt 40
```

Optuna (TPE sampler, 40 trials) uses the same trip-grouped folds and training partition. The search space is recorded in `results/bo_search_space.json`, trials in `results/bo_trials.csv`, the best parameters in `results/bo_best_params.json`, and the tuned-versus-untuned comparison in `results/tuned_vs_untuned.csv`. The tuned XGBoost model is saved to `data/processed/models/`.

### 5. Run the HMM/Viterbi experiment

```powershell
python -m unittest hmm.test_viterbi
python -m hmm.run_hmm 30 XGBoost_tuned
python -m hmm.plot_results
```

The HMM tunes `alpha`, `beta`, unreachable-transition and backtracking penalties, and maximum time gap using 20 sampled training trips and out-of-fold probabilities. The 10 student-test trips are used only after tuning for the development comparison.

Copy tuned HMM settings to the API and export the model artifacts:

```powershell
Copy-Item results/hmm_params.json api/artifacts/hmm_params.json
python -m api.export_artifacts XGBoost_tuned
```

## Experiment results

All values below are development-stage measurements from `results/`.

### Model comparison (student-test partition unless stated)

| Model | CV Macro F1 | Top-1 | Route continuity | Macro F1 | Log loss | Inference (ms/obs) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7130 | 0.8427 | 0.9242 | 0.7105 | 0.6042 | 0.0019 |
| KNN | 0.7277 | 0.8469 | 0.9384 | 0.7329 | 0.7003 | 6.83 |
| Random Forest | 0.7674 | 0.8618 | 0.9395 | 0.7606 | 0.3816 | 0.0079 |
| LightGBM | 0.7572 | 0.8658 | 0.9355 | 0.7630 | 0.3729 | 0.0057 |
| **XGBoost (selected)** | **0.7747** | **0.8697** | **0.9442** | **0.7728** | **0.3646** | 0.0037 |
| MLP | 0.7681 | 0.8587 | 0.9442 | 0.7618 | 0.4052 | 0.0043 |

Macro F1 and log loss are observation-level. Source: `results/model_comparison.csv`.

Cross-validation Top-1 is lower than student-test Top-1 mainly because one fold contains two unreliable training trips (33 and 54); without them the XGBoost CV Top-1 is about 0.847.

### Tuning and HMM decoding

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

The HMM also improved mean arc-set F1 from 89.51% to 91.40% and mean sequence similarity from 75.23% to 76.16% on student-test trips. Per trip, HMM Top-1 was better on 5 and worse on 5 of the 10 student-test trips, so the gain is a modest average effect, not a uniform one (`results/hmm_comparison_per_trip.csv`).

These are development-stage measurements. Sequence similarity compares consecutive-deduplicated per-observation arc sequences against aligned observation labels; it is not a claim of exact recovery of every raw route arc.

Key experiment outputs:

- `results/model_comparison.csv` and `results/cv_folds.csv` - model comparison and per-fold results;
- `results/model_selection.json` - selected model and rankings;
- `results/bo_search_space.json`, `results/bo_trials.csv`, `results/bo_best_params.json` - Bayesian tuning;
- `results/tuned_vs_untuned.csv` - tuned model comparison;
- `results/hmm_params.json`, `results/hmm_tuning_trials.csv` - deployed HMM settings and search record;
- `results/hmm_comparison.csv` - classifier-only versus HMM/Viterbi;
- `results/hmm_comparison_per_trip.csv` - per-trip improvements and regressions;
- `results/sensitivity_exclusion.csv` - effect of excluding the two unreliable training trips;
- `results/decisions.md` - data-quality issues, assumptions and design decisions;
- `results/logs/` - training, tuning, HMM and API logs;
- `results/figures/` - EDA and experiment plots.

## Folium maps

`viz/folium_map.py` runs the same pipeline as the API (tuned XGBoost plus HMM/Viterbi) and writes an interactive HTML map with layer controls and a legend. Layers: road network, GPS observations, reference route and predicted route.

```powershell
python -m viz.folium_map --trip-id 00000009 --data-dir "C:\path\to\Dataset\student_test" --output results/maps/student_test_00000009.html
```

- `--trip-id` is a trip folder name inside `--data-dir`.
- `--data-dir` is a folder that directly contains trip folders (for example `student_test` or `evaluator_test`). If omitted, it uses `PATHSNAP_API_DATA`, else `../Dataset/evaluator_test`.
- `--output` is optional; the default is `results/maps/evaluator_test_<trip-id>.html` (use `--output` when mapping a different partition so the file name matches).

Open the generated HTML file in a browser. The decoder shown in the layer name (`hmm_viterbi` or `classifier_only`) tells you which mode produced the route.

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

The model (`api/artifacts/model.joblib`) and decoder settings are loaded once at start-up. Open `http://localhost:8000/docs` for interactive Swagger documentation.

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

Every request is logged to the console, for example:

```
>>> POST /predict from 192.168.1.20
trip 00000009: 1049 points | decoder=hmm_viterbi | Top-1 ... | continuity ... | pipeline ... ms
<<< POST /predict -> 200 in ... ms
```

The API exports no reference route or reference labels. If `api/artifacts/hmm_params.json` or `hmm/viterbi.py` is unavailable, it falls back to classifier-only mode and reports that in `/health` and `/predict`.

## Run the browser UI

From the repository root:

```powershell
python -m http.server --bind 0.0.0.0 --directory ui 8500
```

(On Linux/macOS use `python3 -m http.server ...`.) Open `http://localhost:8500` in a browser. Serve the page over HTTP rather than opening `index.html` directly, because browsers can block API requests from `file://` pages.

Enter the API address, for example `http://192.168.1.10:8000`, choose **Connect**, select a trip, and choose **Run prediction**. The address is saved in browser local storage. The UI shows the predicted route, the model and decoder in use, and server-side metrics (Top-1, classifier-only Top-1, route continuity, candidate miss rate).

## Two-laptop demonstration

On Laptop 1 (API):

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

On Laptop 2 (UI):

1. start the UI;
2. enter `http://<laptop-1-ip>:8000`; and
3. run `GET /health`, select a trip, and run `POST /predict`.

Keep the API console visible during the demonstration: it logs client addresses, request status, decoded trip, metrics, and timing.

## Tests

| Command | What it checks |
|---|---|
| `python -m unittest hmm.test_viterbi` | HMM/Viterbi decoder behaviour |
| `python -m unittest api.test_route_geometry` | route geometry and segment splitting on synthetic data |
| `python -m api.test_api` | end-to-end API test over every served trip (API must be reachable through the test client; uses `PATHSNAP_API_DATA`, default student-test): HTTP 200, no reference data in responses, metrics present, bad IDs refused |
| `node ui/test-route-segments.cjs` | UI route-segment logic (requires Node.js) |

## Validation and caveats

- Two of the 80 training trips (00000033 and 00000054) have poor reference-route/GPS agreement and are excluded only from robustness-check evaluations, not from the primary 80-trip CV ranking.
- Reference labels come from aligning GPS points to an untimed reference route, so they are approximate ground truth.
- The API only validates trips found under `PATHSNAP_API_DATA`; unknown IDs return HTTP 404.
- Training and student-test partitions are development partitions. Do not use evaluator-trip labels or routes for tuning or testing.
- HMM jumps can remain disconnected after map-matching, so route continuity is reported from predictions rather than assumed.
- HMM parameters were tuned on 20 sampled training trips, and classifier hyperparameters were chosen on the training partition, so training-partition comparisons are development results, not an unbiased nested estimate.
- The generated requirements file pins the environment; the project does not define a separate package metadata or build system.

## Troubleshooting

- **UI cannot reach the API:** check the IPv4 address, the firewall rule for port 8000, that both laptops share a private network, and that `GET /health` works from Laptop 2 (`curl http://<laptop-1-ip>:8000/health`).
- **Unknown trip ID (404):** the trip folder must be inside the folder set in `PATHSNAP_API_DATA`.
- **`/health` says `classifier_only`:** `api/artifacts/hmm_params.json` or `hmm/viterbi.py` was not found or is incompatible; re-copy `results/hmm_params.json` and restart.
- **Dataset not found:** place it at `../Dataset` or set `PATHSNAP_DATA`.
