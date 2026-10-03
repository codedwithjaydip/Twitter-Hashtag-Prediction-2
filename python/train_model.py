"""
Stage 5: train and evaluate models that predict a hashtag's count on the NEXT day.

Chronological split on dates (no random shuffling -> no future data leaks into training):
    first 60% of days  -> train
    next  20% of days  -> validation (used only to choose the best model)
    last  20% of days  -> test       (final, untouched evaluation)
The chosen model is refitted on train+validation and evaluated once on the test days.

Run:  python python/train_model.py
"""
import json
import sys
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import (accuracy_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score)
except ImportError as exc:
    sys.exit(f"Missing Python package ({exc.name}). Run:  pip install -r requirements.txt")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp  # noqa: E402

ROOT = pp.ROOT
MODEL_FILE = ROOT / "models" / "hashtag_model.joblib"
METRICS_FILE = ROOT / "models" / "metrics.json"
LABELS = ["Low", "Medium", "High"]


def to_category(counts, thresholds):
    """Popularity class from a daily count: Low < t1 <= Medium < t2 <= High."""
    return np.select([counts < thresholds[0], counts < thresholds[1]], [0, 1], default=2)


def reg_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


class Persistence:
    """Baseline: tomorrow = today. Any real model must beat this."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["lag0"].to_numpy()


def main():
    daily = pp.load_daily()
    matrix = pp.build_matrix(daily)
    totals = matrix.sum()
    eligible = totals[totals >= pp.MIN_TAG_TOTAL].index
    feats = pp.make_features(matrix[eligible])
    data = feats.dropna(subset=["target"]).copy()
    if data.empty:
        sys.exit("ERROR: no training rows. Check the daily MapReduce output.")

    dates = np.sort(data["date"].unique())
    if len(dates) < 10:
        sys.exit(f"ERROR: only {len(dates)} usable days; need at least 10 for a chronological split.")
    n = len(dates)
    train_end, val_end = dates[int(n * 0.6) - 1], dates[int(n * 0.8) - 1]
    train = data[data["date"] <= train_end]
    val = data[(data["date"] > train_end) & (data["date"] <= val_end)]
    test = data[data["date"] > val_end]
    print(f"Hashtags: {len(eligible)} | rows train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print(f"Train days up to {pd.Timestamp(train_end).date()}, validation up to {pd.Timestamp(val_end).date()}, "
          f"test {pd.Timestamp(test['date'].min()).date()} -> {pd.Timestamp(test['date'].max()).date()}")

    Xtr, Xva, Xte = (pp.model_input(d) for d in (train, val, test))
    ytr, yva, yte = (np.log1p(d["target"].to_numpy()) for d in (train, val, test))

    candidates = {
        "Persistence (baseline)": Persistence(),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=42),
    }

    def predict_counts(model, X, name):
        if name.startswith("Persistence"):
            return np.expm1(model.predict(X))          # lag0 is already log1p -> back to counts
        return np.clip(np.expm1(model.predict(X)), 0, None)

    # ---- model selection on the validation days
    val_results = {}
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        val_results[name] = reg_metrics(val["target"].to_numpy(), predict_counts(model, Xva, name))
        print(f"[validation] {name:24s} MAE={val_results[name]['MAE']:.3f}  RMSE={val_results[name]['RMSE']:.3f}  R2={val_results[name]['R2']:.3f}")
    real = {k: v for k, v in val_results.items() if not k.startswith("Persistence")}
    best_name = min(real, key=lambda k: real[k]["RMSE"])
    print(f"Selected model (lowest validation RMSE): {best_name}")

    # ---- popularity categories: thresholds from the TRAINING targets only
    t1, t2 = np.percentile(train["target"], [75, 95])
    thresholds = [float(max(t1, 1.0)), float(max(t2, t1 + 1.0))]
    print(f"Popularity classes: Low < {thresholds[0]:.0f} <= Medium < {thresholds[1]:.0f} <= High (train 75th / 95th percentile of next-day counts)")

    # ---- final: refit on train+validation, evaluate once on test
    trainval = pd.concat([train, val])
    Xtv, ytv = pp.model_input(trainval), np.log1p(trainval["target"].to_numpy())
    test_results, preds = {}, {}
    for name, model in candidates.items():
        model.fit(Xtv, ytv)
        preds[name] = predict_counts(model, Xte, name)
        test_results[name] = reg_metrics(test["target"].to_numpy(), preds[name])
        print(f"[test]       {name:24s} MAE={test_results[name]['MAE']:.3f}  RMSE={test_results[name]['RMSE']:.3f}  R2={test_results[name]['R2']:.3f}")

    y_true_cat = to_category(test["target"].to_numpy(), thresholds)
    y_pred_cat = to_category(preds[best_name], thresholds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_cat, y_pred_cat, labels=[0, 1, 2], average="macro", zero_division=0)
    class_metrics = {"accuracy": float(accuracy_score(y_true_cat, y_pred_cat)),
                     "precision_macro": float(prec), "recall_macro": float(rec), "f1_macro": float(f1)}
    print(f"[test] popularity class (Low/Medium/High) with {best_name}: {class_metrics}")

    importance = None
    best = candidates[best_name]
    if hasattr(best, "feature_importances_"):
        importance = dict(zip(pp.FEATURES, map(float, best.feature_importances_)))
    elif hasattr(best, "coef_"):
        importance = dict(zip(pp.FEATURES, map(float, best.coef_)))

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best, "model_name": best_name, "features": pp.FEATURES, "thresholds": thresholds}, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps({
        "selected_model": best_name,
        "split": {"train_days_until": str(pd.Timestamp(train_end).date()), "validation_days_until": str(pd.Timestamp(val_end).date()),
                  "test_from": str(pd.Timestamp(test["date"].min()).date()), "test_to": str(pd.Timestamp(test["date"].max()).date()),
                  "rows": {"train": len(train), "validation": len(val), "test": len(test)}},
        "validation": val_results, "test": test_results, "test_popularity_class": class_metrics,
        "thresholds": thresholds, "feature_importance_or_coef": importance,
        "hashtags_used": int(len(eligible)),
    }, indent=2))
    print(f"Saved model   -> {MODEL_FILE}\nSaved metrics -> {METRICS_FILE}")


if __name__ == "__main__":
    main()
