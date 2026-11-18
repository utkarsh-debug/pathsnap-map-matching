# HMM/Viterbi reconstruction

Run tests: `python -m unittest hmm.test_viterbi`.

Run the experiment: `python -m hmm.run_hmm 30 XGBoost_tuned`.
This requires the regenerated feature/label tables, tuned model predictions, and
`results/bo_best_params.json`. It does not read the evaluator partition.

## Model

- Hidden state: a directed candidate arc at each GPS observation. Candidate indices
  are explicitly mapped back to the original trip-specific arc IDs.
- Emission score: classifier probabilities normalised within each observation's
  candidate set, clipped before taking logs, and weighted by `alpha`.
  These are discriminative compatibility scores used as emission potentials,
  not an independently fitted generative GPS observation distribution.
- Transition score: `-|network_distance - GPS_distance| / beta`. Network distance
  includes the remaining fraction of the first arc, a directed shortest path,
  and the initial fraction of the next arc. Same-arc forward motion is handled directly.
- Unreachable transitions receive a finite penalty. Therefore a decoded sequence can
  still contain disconnected jumps; continuity must be measured, not assumed.
- Large time gaps or GPS jumps remove transition information at that step.
- Empty candidate sets produce `-1` for that observation; decoding resumes at the next
  nonempty observation. Normal candidate generation supplies a nearest-arc fallback.
- Standard log-space Viterbi dynamic programming and backtracking choose the sequence.

## Experiment protocol

Five trip-grouped classifier folds generate out-of-fold probabilities for all 80
training trips. A seeded sample of 20 training trips is used to tune decoder parameters
for 30 Optuna trials. The student-test partition is used only for the subsequent comparison.
Training comparisons are development results, not an unbiased nested-CV estimate:
the classifier hyperparameters were previously selected using the training partition.

## Evidence outputs

- `results/hmm_params.json`: chosen settings, tuning trip IDs and fixed settings.
- `results/hmm_tuning_trials.csv`: search record.
- `results/hmm_comparison.csv`: classifier-only versus decoder aggregate metrics.
- `results/hmm_comparison_per_trip.csv`: improvements and regressions by trip.
- `data/processed/hmm/<trip>.parquet`: per-observation classifier and HMM arc IDs,
  for maps (regenerable, gitignored).

Sequence performance is reported both as arc-set F1 and an order-sensitive
SequenceMatcher similarity on consecutive-deduplicated per-observation arc sequences.
The latter compares to aligned observation labels, not to every unsampled arc in the raw
reference route. Neither metric should be described as exact route recovery.

## API handoff

After results exist, copy `results/hmm_params.json` to
`api/artifacts/hmm_params.json`. The API imports the decoder and loads these settings
at startup. Restart it, verify `/health` says `hmm_viterbi`, and run
`python -m api.test_api` on student-test data before the two-laptop demonstration.
