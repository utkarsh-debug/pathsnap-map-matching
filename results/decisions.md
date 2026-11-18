# Data-quality issues, assumptions and decisions (Dataset Exploration PDF, section 8, item 6)

## Data
- Partitions available on the development machine: train = 80 trips, student_test = 10 trips. The 10 evaluator
  trips are not present and are never used for training, tuning, feature selection or any decision.
- Each trip ships its own road network (`.nodes`/`.arcs`); arc ids are row indices of that trip's `.arcs`,
  so a road segment is identified by (trip, arc_id).
- The trips lie in different parts of the world (33 different UTM zones), so a metric CRS (UTM) is chosen per
  trip from the trip's median coordinate (`src/io.py`). A single fixed zone produced infinite coordinates for
  trip 57 (Ecuador).
- No missing values or non-monotonic timestamps. 94.3% of time steps are exactly 1 s; trips 78 and 90 have gaps of
  52,810 s and 153,064 s (see `results/data_quality.txt`).

## Reference route (`.route`) used as labels - assumption
- `.route` is an ordered arc sequence without timestamps. Labels per GPS point come from a monotonic alignment of the
  points to that sequence (`src/candidates.py::label_points`). The reference route is treated as approximate truth,
  not independently verified truth.
- For two training trips (33 and 54) the reference route does not describe the GPS track: 66% / 87% of their
  reference arcs lie outside the candidate set (trip 54: nearest road is a median 84 m from the points; trip 33:
  the reference route is ~122 m from the points while the nearest road is ~7 m away).

## Decisions
1. **Training set = all 80 training trips**, as the Models-and-Training PDF specifies (model fitting, CV, Bayesian
   optimisation). The 2 unreliable trips stay in training.
2. **Second evaluation view.** Because those 2 trips fall into one CV fold (fold 4) and drag that fold's Top-1 down
   for every model, CV metrics are also reported with those trips removed from the *evaluation* (`cv_rel_*`
   columns). `training/sensitivity_exclusion.py` additionally measures what happens when they are removed from
   *training*. The ranking of models is checked under both views (`results/model_selection.json`).
3. **Candidate set**: the 15 nearest arcs within 80 m of each GPS point. Candidate miss rate (reference arc absent
   from the candidate set) is reported in `results/candidate_summary.csv`.
4. **Primary model-selection metric: CV observation-level Top-1 accuracy** (held-out trips). It is exactly the
   per-GPS-point decision PathSnap has to make, and a missing reference arc counts as wrong. Tie-break: CV
   observation-level log loss, because the HMM uses the probabilities as emission scores.
5. **Metrics.** Observation-level: Top-1, macro precision/recall/F1 (classes = reference road segments), log loss
   (probability of the reference arc normalised over the candidates; only observations whose reference arc is in the
   candidate set), route continuity. Candidate-level: binary macro precision/recall/F1, accuracy, log loss for
   every (observation, candidate) row.
6. **All six models are trained on 100% of the training rows**, with the same 5 trip-grouped folds (GroupKFold),
   seed 42. Models are not class-weighted, so probabilities stay calibrated for the HMM.
7. **Bayesian optimisation** (Optuna TPE, seed 42) maximises the same primary metric over the same folds.
