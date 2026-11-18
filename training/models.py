"""Model zoo: six models from different algorithm families (initial/untuned configs).
Every model is trained on 100% of the training rows (train_frac = 1.0)."""
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED

MODELS = {
    "LogisticRegression": dict(
        family="Linear / probabilistic",
        config="C=1.0, L2, lbfgs, max_iter=300, scaled",
        make=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, random_state=SEED)),
        train_frac=1.0),
    "KNN": dict(
        family="Instance-based",
        config="k=25, distance-weighted, euclidean, scaled",
        make=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        train_frac=1.0),
    "RandomForest": dict(
        family="Ensemble (bagging)",
        config="n_estimators=150, max_depth=14, min_samples_leaf=5",
        make=lambda: RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=5,
                                            n_jobs=-1, random_state=SEED),
        train_frac=1.0),
    "LightGBM": dict(
        family="Boosting",
        config="n_estimators=300, lr=0.05, num_leaves=31",
        make=lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                        random_state=SEED, n_jobs=-1, verbose=-1),
        train_frac=1.0),
    "XGBoost": dict(
        family="Boosting",
        config="n_estimators=300, lr=0.05, max_depth=6, hist",
        make=lambda: xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                       tree_method="hist", random_state=SEED, n_jobs=-1,
                                       eval_metric="logloss"),
        train_frac=1.0),
    "MLP": dict(
        family="Neural network",
        config="hidden=(64,32), relu, adam lr=1e-3, alpha=1e-4, 30 epochs, batch 512, scaled",
        make=lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3, max_iter=30,
            batch_size=512, random_state=SEED)),
        train_frac=1.0),
}
