"""Build candidate/feature tables for every trip. Run: python -m src.build_dataset
Outputs data/processed/{partition}_features.parquet and {partition}_labels.parquet,
results/candidate_stats.csv (per-trip candidate miss rate and label quality)."""
import numpy as np
import pandas as pd

from .candidates import build_index, label_points, _project
from .config import PROCESSED_DIR, RESULTS_DIR
from .features import build_features
from .io import list_trips, load_trip


MAX_MISS_RATE = 0.5


def process_trip(part, tid):
    tr = load_trip(part, tid)
    idx = build_index(tr)
    feats = build_features(tr, idx)
    lab_arc, jpos = label_points(tr, idx[2])
    n = len(tr.track)
    labels = pd.DataFrame({"trip": tid, "obs": np.arange(n), "label_arc": lab_arc, "route_pos": jpos})
    feats["trip"] = tid
    feats = feats.merge(labels[["obs", "label_arc"]], on="obs")
    feats["y"] = (feats["arc"] == feats["label_arc"]).astype(np.int8)
    hit = feats.groupby("obs")["y"].max().reindex(range(n)).fillna(0).values > 0
    c = idx[2][lab_arc]
    ld, _ = _project(tr.track.x.values, tr.track.y.values, c[:, 0], c[:, 1])
    labels["label_dist"] = ld
    labels["in_cands"] = hit
    stat = dict(partition=part, trip=tid, n_obs=n, rows=len(feats), miss_rate=float((~hit).mean()),
                label_dist_med=float(np.median(ld)), label_dist_p95=float(np.percentile(ld, 95)),
                nearest_is_label=float((feats[feats["rank"] == 0]["y"]).mean()))
    return feats, labels, stat


def main():
    stats = []
    for part in ("train", "student_test"):
        fs, ls = [], []
        for tid in list_trips(part):
            f, l, s = process_trip(part, tid)
            fs.append(f); ls.append(l); stats.append(s)
        pd.concat(fs, ignore_index=True).to_parquet(PROCESSED_DIR / f"{part}_features.parquet")
        pd.concat(ls, ignore_index=True).to_parquet(PROCESSED_DIR / f"{part}_labels.parquet")
        print(part, "done", sum(len(f) for f in fs), "candidate rows")
    st = pd.DataFrame(stats)
    # Data-quality rule (documented in report): when >50% of a trip's reference labels lie outside
    # the candidate set the reference route does not describe the GPS track -> exclude from training.
    st["usable_for_training"] = st["miss_rate"] <= MAX_MISS_RATE
    st.to_csv(RESULTS_DIR / "candidate_stats.csv", index=False)
    g = st.groupby("partition").apply(lambda d: pd.Series({
        "obs": d.n_obs.sum(), "rows": d.rows.sum(),
        "miss_rate_pooled": (d.miss_rate * d.n_obs).sum() / d.n_obs.sum(),
        "miss_rate_median_trip": d.miss_rate.median(),
        "nearest_is_label_mean": d.nearest_is_label.mean()}), include_groups=False)
    print(g.to_string())
    print(st.nlargest(6, "miss_rate")[["partition", "trip", "n_obs", "miss_rate", "label_dist_p95"]].to_string(index=False))
    g.to_csv(RESULTS_DIR / "candidate_summary.csv")


if __name__ == "__main__":
    main()
