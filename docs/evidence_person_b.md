# Person B evidence notes: XGBoost, MLP, and UI deployment

This is a facts-only working sheet derived from the checked-in experiment outputs and implementation. Use the values and prompts to write and explain the report in your own words. It is not report prose. Results below use the 80 training trips and the 10 student-testing trips; no evaluator-testing data is represented here.

## Model comparison facts

All models were trained on the complete training candidate rows. Cross-validation uses five `GroupKFold` splits grouped by trip, with seed 42. Student-test results are from the ten development-stage student-test trips. The comparison includes six models; the table below focuses on the two assigned to Person B.

| Measure | XGBoost (untuned) | MLP |
|---|---:|---:|
| Family | Gradient boosting | Neural network |
| CV observation Top-1, mean | 0.7682 | 0.7628 |
| CV observation Top-1, fold standard deviation | 0.1951 | 0.1942 |
| CV observation macro F1, mean | 0.7747 | 0.7681 |
| CV observation log loss, mean | 0.4159 | 0.4613 |
| CV route continuity, mean | 0.9381 | 0.9359 |
| Student-test observation Top-1 | 0.8697 | 0.8587 |
| Student-test observation macro precision | 0.7971 | 0.7860 |
| Student-test observation macro recall | 0.7662 | 0.7550 |
| Student-test observation macro F1 | 0.7728 | 0.7618 |
| Student-test observation log loss | 0.3646 | 0.4052 |
| Student-test candidate macro F1 | 0.9245 | 0.9178 |
| Student-test candidate accuracy | 0.9751 | 0.9727 |
| Student-test candidate log loss | 0.0624 | 0.0691 |
| Student-test route continuity | 0.9442 | 0.9442 |
| Student-test reference coverage | 0.9988 | 0.9988 |
| Final model training time recorded | 7.57 s | 82.87 s |
| Inference time per GPS observation recorded | 0.00369 ms | 0.00429 ms |

The recorded train and inference times are from the experiment environment in `results/model_comparison.csv`; the repository does not record the machine specification alongside them. Observation log loss is computed over the reference candidate's model probability normalized across candidates when that reference candidate exists. Candidate-level log loss is over the binary candidate rows.

## XGBoost configuration and results

- Untuned comparison configuration: 300 estimators, learning rate 0.05, maximum depth 6, histogram tree method, seed 42, `eval_metric="logloss"`.
- Classifier objective: binary logistic classification; the fitted model predicts a probability for each (GPS observation, candidate arc) row.
- CV folds (Top-1, all trips): 0.8202, 0.9080, 0.8696, 0.8173, 0.4257.
- CV folds (Top-1, excluding the two unreliable-reference trips from evaluation): 0.8202, 0.9080, 0.8696, 0.8173, 0.8209.
- Train candidate log loss mean: 0.0603; held-out CV candidate log loss mean: 0.0800.
- The API artifact is the Bayesian-tuned XGBoost model, named `XGBoost_tuned` in `api/artifacts/pipeline_config.json`.

### Tuned XGBoost comparison

| Measure | Untuned | Tuned |
|---|---:|---:|
| CV Top-1 | 0.7682 | 0.7707 |
| CV observation macro F1 | 0.7747 | 0.7786 |
| CV observation log loss | 0.4159 | 0.4120 |
| Student Top-1 | 0.8697 | 0.8695 |
| Student observation macro F1 | 0.7728 | 0.7746 |
| Student observation log loss | 0.3646 | 0.3600 |
| Student route continuity | 0.9442 | 0.9475 |
| Final training time | 7.12 s | 12.24 s |
| Inference time per GPS observation | 0.00331 ms | 0.00478 ms |

The search procedure and complete tuned parameter set are recorded in `results/bo_search_space.json`, `results/bo_best_params.json`, and `results/bo_trials.csv`; Bayesian-optimisation reporting is assigned to Person A.

## MLP configuration and curve evidence

- Configuration: `StandardScaler`, hidden layers `(64, 32)`, ReLU activation, Adam, initial learning rate `1e-3`, L2 regularization `alpha=1e-4`, batch size 512, maximum 30 epochs, seed 42.
- Classifier loss: binary cross-entropy/log loss with the estimator's L2 weight penalty.
- CV folds (Top-1, all trips): 0.8148, 0.9024, 0.8608, 0.8143, 0.4216.
- CV folds (Top-1, excluding the two unreliable-reference trips from evaluation): 0.8148, 0.9024, 0.8608, 0.8143, 0.8171.
- Train candidate log loss mean: 0.0658; held-out CV candidate log loss mean: 0.0869.
- Fold-0 epoch history in `results/histories/MLP_history.csv`: train log loss was 0.07722 at epoch 1 and 0.06275 at epoch 30. Validation log loss reached 0.09400 at epoch 4 and was 0.10032 at epoch 30.
- The fold-0 history helper estimates training loss on a fixed random subset of up to 300,000 in-fold rows; validation loss is calculated on the held-out fold. This curve is evidence from that fold, not an average over five folds.

## Data-quality caveat in interpreting the folds

Trips `00000033` and `00000054` have reference routes that do not align with most of their GPS observations. They are in fold 4. For the untuned XGBoost, fold-4 Top-1 is 0.4257 over all fold-4 observations and 0.8209 after those two trips are excluded from evaluation. For the MLP, the corresponding values are 0.4216 and 0.8171. The all-trip metric remains the primary recorded CV result; the reliable-trip value is a secondary diagnostic.

## UI and deployment evidence to capture

The UI in `ui/` reads the trip list from the API and sends the selected trip ID to `POST /predict`. The API returns model and decoder names, GPS coordinates, predicted route coordinates, timings, and server-calculated metrics. The page does not receive reference arcs or labels.

For the report's deployment section, capture genuine evidence when Laptop 1 is running the API with the intended dataset:

1. A Laptop 2 browser screenshot showing the connected API, a selected trip, the returned predicted path, and metrics.
2. A Laptop 1 terminal screenshot showing the incoming `POST /predict` request and successful response status.
3. Record the model, decoder, trip ID, API address, date, and outcome used for those screenshots.

The checked-out workspace now includes the HMM/Viterbi decoder and an initial parameter file. That file is explicitly marked `initial_defaults_not_tuned`; it must not be described as experimentally tuned. The configured workspace does not currently contain the raw training and student-test partitions or their processed feature tables, so no HMM tuning metrics or classifier-versus-HMM results have been generated. A real end-to-end screenshot must be taken after the permitted data partitions and Laptop 1 API are available; do not substitute a fabricated response or screenshot.

## Source files for checking these facts

- `results/model_comparison.csv`
- `results/cv_folds.csv`
- `results/histories/MLP_history.csv`
- `results/histories/XGBoost_history.csv`
- `results/tuned_vs_untuned.csv`
- `results/decisions.md`
- `api/artifacts/pipeline_config.json`
- `training/models.py`
- `training/common.py`
