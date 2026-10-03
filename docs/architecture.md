# Architecture

```
twcs.csv (public customer-support Twitter dataset, 2.8M tweets)
   |  PowerShell:  java TwitterDataIngestion
   v
Java ingestion (real CSV parser: quotes, commas, multi-line tweets)
   -> output/hashtags.csv        one line per hashtag occurrence:  date,hashtag
   |  WSL:  hdfs dfs -put
   v
HDFS  /twitter-hashtag/input/hashtags.csv
   |  WSL:  hadoop jar
   v
MapReduce Job 1  HashtagCount        -> /twitter-hashtag/output        hashtag <TAB> count
MapReduce Job 2  DailyHashtagCount   -> /twitter-hashtag/daily-output  date,hashtag <TAB> count
   |  WSL:  hdfs dfs -getmerge
   v
output/hashtag_totals.tsv, output/daily_counts.tsv
   |  python/preprocess.py   (clean, continuous daily matrix, features)
   v
output/features.csv
   |  python/train_model.py  (chronological split, 4 models compared)
   v
models/hashtag_model.joblib + models/metrics.json
   |  python/predict.py  /  dashboard/app.py (Streamlit)
   v
Next-day frequency + Low/Medium/High popularity
```

## Data decisions
* One tweet can span several physical lines, so the Java program uses a character-level CSV parser (quote aware) instead of `readLine()`.
* Dates are converted to UTC `yyyy-MM-dd`.
* The CSV holds a few tweets from 2008-2016, but real volume starts on 2017-10-04. Python only uses days with at least 1000 hashtag occurrences (`MIN_DAILY_TOTAL` in `preprocess.py`).
* Hashtags seen fewer than 30 times in the active period are not used for training (`MIN_TAG_TOTAL`) because their daily series is almost all zeros. They can still be predicted.
* Purely numeric tags (`#1`, `#2`) are removed in cleaning.

## ML design
* Target: the hashtag's count on the next day (regression). The count is also mapped to Low / Medium / High using thresholds computed from the training data only (75th and 95th percentile of next-day counts).
* Split by time: first 60% of days train, next 20% validation (model choice), last 20% test.
* Models: persistence baseline (tomorrow = today), Linear Regression, Random Forest, Gradient Boosting.
* Counts are log-transformed because they are very skewed.

## Data source note
The X/Twitter API returned `402 Payment Required` (`credits depleted`), so a public dataset is used. The system does **not** use live data and cannot predict live Twitter trends.
