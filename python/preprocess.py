"""
Stage 4: read the Hadoop MapReduce output, clean it and build ML features.

Inputs  (created by `hdfs dfs -getmerge`, see README):
    output/daily_counts.tsv     lines:  date,hashtag <TAB> count      (Job 2)
    output/hashtag_totals.tsv   lines:  hashtag <TAB> count          (Job 1)
Outputs:
    output/features.csv         one row per (hashtag, day) with the features below

Run:  python python/preprocess.py
"""
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(f"Missing Python package ({exc.name}). Run:  pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parents[1]
DAILY_FILE = ROOT / "output" / "daily_counts.tsv"
TOTAL_FILE = ROOT / "output" / "hashtag_totals.tsv"
FEATURES_FILE = ROOT / "output" / "features.csv"

# The CSV contains a few tweets from 2008-2016, but real volume only starts in Oct 2017.
# Days whose total hashtag count is below this value are treated as "dataset not active yet".
MIN_DAILY_TOTAL = 1000
# Only hashtags seen at least this many times in the active period are used to train the model.
MIN_TAG_TOTAL = 30

# ---------------------------------------------------------------- features
# Every feature is for a hashtag H on day t; the target is H's count on day t+1.
FEATURES = [
    "lag0",       # count on day t (today's popularity: strongest signal for tomorrow)
    "lag1",       # count on day t-1
    "lag2",       # count on day t-2
    "lag3",       # count on day t-3
    "ma3",        # average of last 3 days (smooths daily noise)
    "ma7",        # average of last 7 days (weekly level)
    "growth",     # (lag0 - lag1) / (lag1 + 1): is the hashtag rising or falling?
    "dow",        # weekday (0=Mon) of the day being predicted: weekends have less activity
    "hist_mean",  # mean daily count from the first active day up to day t (historical popularity)
]
COUNT_FEATURES = ["lag0", "lag1", "lag2", "lag3", "ma3", "ma7", "hist_mean"]


def _read_tsv(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"ERROR: {what} not found: {path}\n"
            "Run the Hadoop jobs and copy the result with `hdfs dfs -getmerge` (see README, section 'MapReduce')."
        )
    if path.stat().st_size == 0:
        sys.exit(f"ERROR: {path} is empty. Did the MapReduce job produce output?")
    try:
        return pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    except Exception as exc:  # malformed file
        sys.exit(f"ERROR: could not read {path}: {exc}")


def load_daily(path: Path = DAILY_FILE) -> pd.DataFrame:
    """Daily MapReduce output -> DataFrame[date, hashtag, count] (cleaned)."""
    raw = _read_tsv(path, "daily MapReduce output")
    if raw.shape[1] != 2:
        sys.exit(f"ERROR: {path} should have 2 tab-separated columns (date,hashtag<TAB>count), got {raw.shape[1]}.")
    key = raw[0].str.split(",", n=1, expand=True)
    if key.shape[1] != 2:
        sys.exit(f"ERROR: {path} keys must look like 'date,hashtag'.")
    df = pd.DataFrame({
        "date": pd.to_datetime(key[0], format="%Y-%m-%d", errors="coerce"),
        "hashtag": key[1].str.strip().str.lower(),
        "count": pd.to_numeric(raw[1], errors="coerce"),
    })
    before = len(df)
    df = df.dropna()
    df = df[(df["hashtag"] != "") & (df["count"] > 0)]
    df = df[~df["hashtag"].str.fullmatch(r"\d+")]      # '#1', '#2' ... are numbering, not topics
    dropped = before - len(df)
    if dropped:
        print(f"[clean] removed {dropped} invalid / empty / numeric-only rows")
    if df.empty:
        sys.exit("ERROR: no valid rows left after cleaning.")
    df["count"] = df["count"].astype(int)
    # the same (date, hashtag) must be unique; sum in case of duplicates
    return df.groupby(["date", "hashtag"], as_index=False)["count"].sum()


def load_totals(path: Path = TOTAL_FILE) -> pd.DataFrame:
    raw = _read_tsv(path, "total MapReduce output")
    df = pd.DataFrame({"hashtag": raw[0].str.strip().str.lower(), "count": pd.to_numeric(raw[1], errors="coerce")})
    return df.dropna().query("hashtag != ''").sort_values("count", ascending=False).reset_index(drop=True)


def build_matrix(daily: pd.DataFrame, min_daily_total: int = MIN_DAILY_TOTAL) -> pd.DataFrame:
    """Matrix [continuous active dates x hashtags]; days without the hashtag = 0."""
    totals = daily.groupby("date")["count"].sum()
    active = totals[totals >= min_daily_total]
    if active.empty:
        sys.exit(f"ERROR: no day has at least {min_daily_total} hashtag occurrences.")
    start, end = active.index.min(), active.index.max()
    daily = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    matrix = daily.pivot(index="date", columns="hashtag", values="count").fillna(0.0)
    return matrix.reindex(pd.date_range(start, end, freq="D"), fill_value=0.0)


def make_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """Long table [date, hashtag, features..., target]. target = count on the next day (NaN on last day)."""
    parts = {
        "lag0": matrix,
        "lag1": matrix.shift(1),
        "lag2": matrix.shift(2),
        "lag3": matrix.shift(3),
        "ma3": matrix.rolling(3, min_periods=3).mean(),
        "ma7": matrix.rolling(7, min_periods=3).mean(),
        "hist_mean": matrix.expanding().mean(),
        "target": matrix.shift(-1),
    }
    parts["growth"] = (matrix - matrix.shift(1)) / (matrix.shift(1) + 1)
    frames = []
    for name, frame in parts.items():
        s = frame.stack(future_stack=True)
        s.name = name
        frames.append(s)
    out = pd.concat(frames, axis=1)
    out.index.names = ["date", "hashtag"]
    out = out.reset_index()
    out["dow"] = (out["date"] + pd.Timedelta(days=1)).dt.dayofweek
    out = out.dropna(subset=FEATURES)          # drops the first days that have no 3-day history
    return out[["date", "hashtag"] + FEATURES + ["target"]]


def model_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric matrix given to the models: counts on log scale (counts are very skewed)."""
    x = frame[FEATURES].copy()
    x[COUNT_FEATURES] = np.log1p(x[COUNT_FEATURES])
    x["growth"] = x["growth"].clip(-1, 10)
    return x


def main() -> None:
    daily = load_daily()
    matrix = build_matrix(daily)
    totals = matrix.sum()
    eligible = totals[totals >= MIN_TAG_TOTAL].index
    feats = make_features(matrix[eligible])
    FEATURES_FILE.parent.mkdir(exist_ok=True)
    feats.to_csv(FEATURES_FILE, index=False)
    print(f"Active period          : {matrix.index.min().date()} -> {matrix.index.max().date()} ({len(matrix)} days)")
    print(f"Hashtag occurrences    : {int(matrix.values.sum())}")
    print(f"Unique hashtags        : {matrix.shape[1]}")
    print(f"Hashtags used for ML   : {len(eligible)} (seen >= {MIN_TAG_TOTAL} times)")
    print(f"Feature rows saved     : {len(feats)} -> {FEATURES_FILE}")


if __name__ == "__main__":
    main()
