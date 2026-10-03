"""
Stage 6: predict the next-day popularity of one hashtag.

Run:  python python/predict.py            (asks for a hashtag)
      python python/predict.py ios11
"""
import re
import sys
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing Python package ({exc.name}). Run:  pip install -r requirements.txt")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp  # noqa: E402

MODEL_FILE = pp.ROOT / "models" / "hashtag_model.joblib"
LABELS = ["Low", "Medium", "High"]


class PredictionError(Exception):
    pass


def load_model():
    if not MODEL_FILE.exists():
        raise PredictionError(f"Model not found: {MODEL_FILE}. Train it first:  python python/train_model.py")
    return joblib.load(MODEL_FILE)


def normalise(raw: str) -> str:
    tag = (raw or "").strip().lstrip("#").lower()
    if not tag:
        raise PredictionError("Hashtag is empty.")
    if not re.fullmatch(r"\w+", tag):
        raise PredictionError("Invalid hashtag: use letters, digits and underscore only (no spaces or symbols).")
    return tag


def predict_hashtag(tag: str, matrix: pd.DataFrame, bundle: dict) -> dict:
    """Uses the last day in the data as 'today' and predicts the following day."""
    tag = normalise(tag)
    if tag not in matrix.columns:
        raise PredictionError(f"Hashtag '#{tag}' does not occur in the active period of the dataset.")
    feats = pp.make_features(matrix[[tag]])
    today = feats[feats["date"] == matrix.index.max()]
    if today.empty:
        raise PredictionError("Not enough history to build features for this hashtag.")
    X = pp.model_input(today)[bundle["features"]]
    count = float(max(np.expm1(bundle["model"].predict(X))[0], 0.0))
    t1, t2 = bundle["thresholds"]
    label = LABELS[0 if count < t1 else 1 if count < t2 else 2]
    return {"hashtag": tag, "last_date": matrix.index.max(), "predicted_date": matrix.index.max() + pd.Timedelta(days=1),
            "last_count": int(matrix[tag].iloc[-1]), "predicted_count": count, "category": label}


def main():
    bundle = load_model()
    matrix = pp.build_matrix(pp.load_daily())
    raw = sys.argv[1] if len(sys.argv) > 1 else input("Enter hashtag: ")
    try:
        r = predict_hashtag(raw, matrix, bundle)
    except PredictionError as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"\nHashtag                : #{r['hashtag']}")
    print(f"Last observed ({r['last_date'].date()}) : {r['last_count']}")
    print(f"Predicted popularity for {r['predicted_date'].date()}:\n{r['category']}")
    print(f"Predicted frequency:\n{round(r['predicted_count'])}")


if __name__ == "__main__":
    main()
