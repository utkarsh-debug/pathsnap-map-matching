"""Plot actual HMM comparisons: python -m hmm.plot_results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import RESULTS_DIR, FIG_DIR

d = pd.read_csv(RESULTS_DIR / "hmm_comparison_per_trip.csv", dtype={"trip": str})
s = pd.read_csv(RESULTS_DIR / "hmm_comparison.csv").set_index("partition")
student = d[d.partition == "student_test"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(len(student))
ax[0].bar(x - .2, student.clf_top1, .4, label="Classifier")
ax[0].bar(x + .2, student.top1, .4, label="HMM")
ax[0].set_xticks(x, student.trip.str[-2:], rotation=45)
ax[0].set(title="Student-test Top-1 by trip", xlabel="Trip suffix", ylim=(0, 1))
ax[0].legend()
r = s.loc["student_test"]
labels = ["Top-1", "Continuity", "Arc-set F1", "Ordered sequence similarity"]
before = [r.clf_top1, r.clf_continuity, r.clf_arc_f1, r.clf_sequence_similarity]
after = [r.hmm_top1, r.hmm_continuity, r.hmm_arc_f1, r.hmm_sequence_similarity]
x = np.arange(4)
ax[1].bar(x-.2, before, .4, label="Classifier")
ax[1].bar(x+.2, after, .4, label="HMM")
ax[1].set_xticks(x, labels, rotation=20)
ax[1].set(title="Student-test aggregate comparison", ylim=(0, 1))
ax[1].legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "hmm_comparison.png", dpi=150)
plt.close(fig)
