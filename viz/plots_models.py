"""Comparison plots from real experiment outputs. Run: python -m viz.plots_models"""
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


def bar(ax, vals, title, err=None, fmt="{:.3f}"):
    b = ax.bar(x, vals, yerr=err, capsize=3, color="tab:blue")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_title(title)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=8)


# 1. CV macro F1 (mean +/- std over group folds)
fig, ax = plt.subplots(figsize=(7, 4))
bar(ax, comp.cv_macro_f1_mean, "Group-CV macro F1 (5 folds, mean±std)", err=comp.cv_macro_f1_std)
ax.set_ylim(max(0, comp.cv_macro_f1_mean.min() - 0.05), 1)
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_cv_macro_f1.png", dpi=140); plt.close()

# 2. student-test top-1 + macro F1
fig, ax = plt.subplots(figsize=(8, 4)); w = 0.38
ax.bar(x - w / 2, comp.student_top1, w, label="Top-1 accuracy")
ax.bar(x + w / 2, comp.student_macro_f1, w, label="Macro F1")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.legend()
ax.set_title("Student-test: observation-level Top-1 and candidate macro F1")
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_student_top1_f1.png", dpi=140); plt.close()

# 3. log loss
fig, ax = plt.subplots(figsize=(8, 4)); w = 0.38
ax.bar(x - w / 2, comp.cv_log_loss_mean, w, label="CV")
ax.bar(x + w / 2, comp.student_log_loss, w, label="Student-test")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.legend(); ax.set_title("Log loss (lower is better)")
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_log_loss.png", dpi=140); plt.close()

# 4. fold stability (boxplot of fold macro F1) and route continuity / inference time
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].boxplot([folds[folds.model == n].macro_f1 for n in names], tick_labels=names); ax[0].set_title("Fold-to-fold macro F1 (stability)")
ax[0].tick_params(axis="x", rotation=30)
bar(ax[1], comp.student_route_continuity, "Student-test route continuity")
ax[1].set_ylim(max(0, comp.student_route_continuity.min() - 0.05), 1)
bar(ax[2], comp.infer_ms_per_obs, "Inference time (ms per GPS observation)", fmt="{:.4f}")
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_stability_continuity_time.png", dpi=140); plt.close()

# 5. overfitting evidence: train vs validation log loss
fig, ax = plt.subplots(figsize=(8, 4)); w = 0.38
ax.bar(x - w / 2, comp.cv_train_log_loss_mean, w, label="train (in-fold)")
ax.bar(x + w / 2, comp.cv_log_loss_mean, w, label="validation (held-out trips)")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.legend(); ax.set_title("Train vs validation log loss")
plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_overfitting.png", dpi=140); plt.close()

# 6. confusion matrices on student-test candidate rows (threshold 0.5)
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

# 7. epoch-wise histories ONLY for models that produce them (LightGBM, XGBoost, MLP)
hist = {p.stem.replace("_history", ""): pd.read_csv(p) for p in sorted((RESULTS_DIR / "histories").glob("*_history.csv"))}
if hist:
    fig, axs = plt.subplots(1, len(hist), figsize=(5 * len(hist), 3.8)); axs = np.atleast_1d(axs)
    for a, (nm, h) in zip(axs, hist.items()):
        xs = h.iloc[:, 0]
        a.plot(xs, h.train_loss, label="train"); a.plot(xs, h.val_loss, label="validation (held-out trips)")
        a.set_title(f"{nm} (CV fold 0)"); a.set_xlabel(h.columns[0]); a.set_ylabel("log loss"); a.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "cmp_training_curves.png", dpi=140); plt.close()
print("figures written to", FIG_DIR)
