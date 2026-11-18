"""Comparison plots from real experiment outputs only. Run: python -m viz.plots_models"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config import FIG_DIR, PROCESSED_DIR, RESULTS_DIR

comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
folds = pd.read_csv(RESULTS_DIR / "cv_folds.csv")
names = list(comp.model)
x = np.arange(len(names))


def labels(ax):
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")


def bar(ax, vals, title, err=None, fmt="{:.3f}"):
    b = ax.bar(x, vals, yerr=err, capsize=3, color="tab:blue")
    labels(ax); ax.set_title(title)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=8)


def grouped(ax, a, b, la, lb, title):
    w = 0.38
    ax.bar(x - w / 2, a, w, label=la); ax.bar(x + w / 2, b, w, label=lb)
    labels(ax); ax.legend(); ax.set_title(title)


# 1. Cross-validation Macro F1 (observation-level and candidate-level) across all models
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
bar(ax[0], comp.cv_obs_macro_f1_mean, "CV observation-level macro F1 (5 trip-grouped folds, mean±std)", err=comp.cv_obs_macro_f1_std)
bar(ax[1], comp.cv_cand_macro_f1_mean, "CV candidate-level macro F1 (mean±std)", err=comp.cv_cand_macro_f1_std)
ax[1].set_ylim(max(0, comp.cv_cand_macro_f1_mean.min() - 0.05), 1)
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_cv_macro_f1.png", dpi=140); plt.close()

# 2. Primary metric: CV observation-level Top-1, all 80 trips vs reliable trips only
fig, ax = plt.subplots(figsize=(8, 4))
grouped(ax, comp.cv_top1_mean, comp.cv_rel_top1_mean, "all 80 trips", "excluding the 2 unreliable trips", "CV observation-level Top-1 (primary selection metric)")
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_cv_top1.png", dpi=140); plt.close()

# 3. Student-test Top-1 and macro F1
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
grouped(ax[0], comp.student_top1, comp.student_obs_macro_f1, "Top-1 accuracy", "observation-level macro F1", "Student-test (observation-level)")
grouped(ax[1], comp.student_top1, comp.student_cand_macro_f1, "Top-1 accuracy", "candidate-level macro F1", "Student-test (Top-1 and candidate-level macro F1)")
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_student_top1_f1.png", dpi=140); plt.close()

# 4. Log loss (observation-level and candidate-level; CV and student-test)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
grouped(ax[0], comp.cv_obs_log_loss_mean, comp.student_obs_log_loss, "CV", "student-test", "Observation-level log loss (lower is better)")
grouped(ax[1], comp.cv_cand_log_loss_mean, comp.student_cand_log_loss, "CV", "student-test", "Candidate-level log loss (lower is better)")
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_log_loss.png", dpi=140); plt.close()

# 5. Fold stability, route continuity, inference time
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
ax[0].boxplot([folds[folds.model == n].top1 for n in names], tick_labels=names)
ax[0].set_title("Fold-to-fold Top-1 (stability; fold 4 holds the 2 unreliable trips)"); ax[0].tick_params(axis="x", rotation=30)
bar(ax[1], comp.student_route_continuity, "Student-test route continuity")
ax[1].set_ylim(max(0, comp.student_route_continuity.min() - 0.05), 1)
bar(ax[2], comp.infer_ms_per_obs, "Inference time (ms per GPS observation)", fmt="{:.4f}")
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_stability_continuity_time.png", dpi=140); plt.close()

# 6. Over/underfitting evidence: train vs held-out
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
grouped(ax[0], comp.cv_train_top1_mean, comp.cv_rel_top1_mean, "train (in-fold)", "validation (held-out trips, reliable)", "Top-1: train vs validation")
grouped(ax[1], comp.cv_train_cand_log_loss_mean, comp.cv_rel_cand_log_loss_mean, "train (in-fold)", "validation (held-out trips, reliable)", "Candidate log loss: train vs validation")
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_overfitting.png", dpi=140); plt.close()

# 7. Confusion matrices (candidate-level, student-test, threshold 0.5)
n = len(names); cols = 3; rows = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows)); axs = np.atleast_1d(axs).ravel()
for a, nm in zip(axs, names):
    d = pd.read_parquet(PROCESSED_DIR / "models" / f"preds_{nm}.parquet")
    cm = confusion_matrix(d.y, (d.p >= 0.5).astype(int), labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["not ref arc", "ref arc"]).plot(ax=a, colorbar=False, values_format="d")
    a.set_title(nm)
for a in axs[n:]:
    a.axis("off")
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_confusion_matrices.png", dpi=130); plt.close()

# 8. Epoch-wise histories ONLY for models that produce them (LightGBM, XGBoost, MLP)
hist = {p.stem.replace("_history", ""): pd.read_csv(p) for p in sorted((RESULTS_DIR / "histories").glob("*_history.csv"))}
if hist:
    fig, axs = plt.subplots(1, len(hist), figsize=(5 * len(hist), 3.8)); axs = np.atleast_1d(axs)
    for a, (nm, h) in zip(axs, hist.items()):
        xs = h.iloc[:, 0]
        a.plot(xs, h.train_loss, label="train"); a.plot(xs, h.val_loss, label="validation (held-out trips)")
        a.set_title(f"{nm} (CV fold 0)"); a.set_xlabel(h.columns[0]); a.set_ylabel("log loss"); a.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_training_curves.png", dpi=140); plt.close()
print("figures written to", FIG_DIR)
