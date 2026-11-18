"""Sanity check of candidate generation + labelling on a few trips: python -m src.check_candidates"""
import sys
import time
import numpy as np
from .candidates import build_index, generate_candidates, label_points
from .io import list_trips, load_trip

part = sys.argv[1] if len(sys.argv) > 1 else "train"
ids = list_trips(part)[: int(sys.argv[2]) if len(sys.argv) > 2 else 4]
for tid in ids:
    t0 = time.time()
    tr = load_trip(part, tid)
    idx = build_index(tr)
    cand = generate_candidates(tr, idx)
    lab, jpos = label_points(tr, idx[2])
    # miss rate: label arc absent from candidate set
    m = cand.merge(
        __import__("pandas").DataFrame({"obs": np.arange(len(lab)), "arc": lab, "y": 1}),
        on=["obs", "arc"], how="left")
    hit = m.groupby("obs")["y"].sum().reindex(range(len(lab))).fillna(0) > 0
    nearest_is_label = (cand[cand["rank"] == 0].set_index("obs")["arc"].reindex(range(len(lab))).values == lab).mean()
    print(f"{part}/{tid} n={len(lab)} cands/pt={len(cand)/len(lab):.1f} miss_rate={(~hit).mean():.4f} "
          f"nearest==label={nearest_is_label:.3f} monotone_viol={(np.diff(jpos)<0).sum()} {time.time()-t0:.1f}s")
