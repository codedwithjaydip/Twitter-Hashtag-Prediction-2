"""Streamlit dashboard.  Run from the project root:  streamlit run dashboard/app.py"""
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import streamlit as st
except ImportError as exc:
    sys.exit(f"Missing Python package ({exc.name}). Run:  pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import predict as pr      # noqa: E402
import preprocess as pp   # noqa: E402

st.set_page_config(page_title="Twitter Hashtag Popularity Prediction", layout="wide")
st.title("Twitter Hashtag Popularity Prediction using Big Data")
st.caption("Java ingestion -> HDFS -> Hadoop MapReduce -> Python ML. Source: public customer-support Twitter dataset (twcs.csv), not live X API data.")


@st.cache_data(show_spinner="Loading Hadoop output...")
def load():
    if not pp.DAILY_FILE.exists():
        return None
    daily = pp.load_daily()
    return daily, pp.build_matrix(daily)


try:
    loaded = load()
except SystemExit as exc:           # preprocess.py reports problems with sys.exit(message)
    st.error(str(exc))
    st.stop()
if loaded is None:
    st.error(f"Hadoop output not found: {pp.DAILY_FILE}\n\nRun the MapReduce jobs and `hdfs dfs -getmerge` first (see README).")
    st.stop()
daily, matrix = loaded
totals = matrix.sum().sort_values(ascending=False)

# ------------------------------------------------------------------ overview
st.header("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total hashtag occurrences", f"{int(totals.sum()):,}")
c2.metric("Unique hashtags", f"{len(totals):,}")
c3.metric("Date range", f"{matrix.index.min().date()} to {matrix.index.max().date()}")
c4.metric("Days analysed", len(matrix))
st.caption(f"Only days with at least {pp.MIN_DAILY_TOTAL} hashtag occurrences are used (the dataset has very few tweets before Oct 2017).")

# ------------------------------------------------------------------ charts
st.header("Charts")
left, right = st.columns(2)
with left:
    st.subheader("Top 10 hashtags")
    st.bar_chart(totals.head(10))
with right:
    st.subheader("Frequency distribution (number of hashtags per total-count range)")
    bins = [1, 2, 6, 11, 51, 101, 501, 10**9]
    labels = ["1", "2-5", "6-10", "11-50", "51-100", "101-500", "500+"]
    dist = pd.cut(totals, bins=bins, labels=labels, right=False).value_counts().reindex(labels)
    dist.index = dist.index.astype(str)
    st.bar_chart(dist)
st.subheader("Daily hashtag activity (all hashtags)")
st.line_chart(matrix.sum(axis=1).rename("hashtag occurrences per day"))

# ------------------------------------------------------------------ selected hashtag + prediction
st.header("Hashtag trend and prediction")
default = totals.index[0]
choice = st.selectbox("Select a hashtag (type to search)", list(totals.index[:2000]), index=0)
typed = st.text_input("...or type any hashtag", "")
tag = typed.strip().lstrip("#").lower() if typed.strip() else choice

try:
    bundle = pr.load_model()
    result = pr.predict_hashtag(tag, matrix, bundle)
except pr.PredictionError as exc:
    st.warning(str(exc))
    st.stop()

hist = matrix[tag].rename("historical frequency")
pred_series = pd.Series(np.nan, index=list(hist.index) + [result["predicted_date"]], name="predicted next day")
pred_series.iloc[-2] = hist.iloc[-1]
pred_series.iloc[-1] = result["predicted_count"]
chart = pd.concat([hist.reindex(pred_series.index), pred_series], axis=1)
st.line_chart(chart)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total occurrences", int(hist.sum()))
m2.metric(f"Last day ({result['last_date'].date()})", result["last_count"])
m3.metric(f"Predicted frequency ({result['predicted_date'].date()})", round(result["predicted_count"]))
m4.metric("Predicted popularity", result["category"])
t1, t2 = bundle["thresholds"]
st.caption(f"Classes use thresholds learned from the training data: Low < {t1:.0f} <= Medium < {t2:.0f} <= High (daily mentions). Model: {bundle['model_name']}.")

# ------------------------------------------------------------------ model quality
metrics_file = ROOT / "models" / "metrics.json"
if metrics_file.exists():
    with st.expander("Model evaluation (chronological test set)"):
        m = json.loads(metrics_file.read_text())
        st.write(f"Test days: {m['split']['test_from']} to {m['split']['test_to']} | selected model: **{m['selected_model']}**")
        st.dataframe(pd.DataFrame(m["test"]).T.round(3))
        st.write("Popularity class (Low/Medium/High):", {k: round(v, 3) for k, v in m["test_popularity_class"].items()})
