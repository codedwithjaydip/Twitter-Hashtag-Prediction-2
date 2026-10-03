# Twitter Hashtag Popularity Prediction using Big Data

## 1. Objective
Extract hashtags from a Twitter dataset, store and process them with **HDFS + Hadoop MapReduce**, and use **Python machine learning** to predict next-day hashtag popularity, shown in a **Streamlit dashboard**.

## 2. Problem statement
Hashtag usage changes day by day. Counting hashtags only describes the past; this project uses historical daily counts to predict how often a hashtag will be used tomorrow and whether that is Low, Medium or High popularity.

## 3. Technologies
Java (ingestion, MapReduce), Hadoop 3.5.0 (HDFS + MapReduce, in WSL Ubuntu), Python (pandas, NumPy, scikit-learn, joblib), Streamlit.

## 4. Architecture
See `docs/architecture.md`.
`twcs.csv -> Java ingestion -> HDFS -> MapReduce (total + daily) -> Python preprocessing -> ML -> prediction -> dashboard`

## 5. Dataset
`data/twcs.csv` is the public *Customer Support on Twitter* dataset. Columns: `tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`. `created_at` looks like `Tue Oct 31 22:10:47 +0000 2017`. Text fields are quoted, contain commas, and some tweets span several lines, so a real CSV parser is required (the Java program has one).
**Data-source decision / limitation:** the X API returned `402 Payment Required (credits depleted)`, so this project uses an offline public dataset. It is not live X data.

## 6. Folder structure
```
data/twcs.csv                      dataset
java/TwitterDataIngestion.java     CSV parsing -> output/hashtags.csv (date,hashtag)
hadoop/HashtagCount.java           MapReduce job 1: total count per hashtag
hadoop/DailyHashtagCount.java      MapReduce job 2: count per (date,hashtag)
python/preprocess.py               cleaning + features
python/train_model.py              chronological split, model comparison, saves model
python/predict.py                  command-line prediction
dashboard/app.py                   Streamlit dashboard
models/  output/  docs/            generated models/data, documentation
```

## 7. Requirements
Windows 11, WSL Ubuntu with Hadoop 3.5.0 at `/mnt/c/hadoop/hadoop-3.5.0` (already installed, nothing to reinstall), a JDK (`javac`), Python 3.10+.

## 8. Stage 1: Java ingestion (Windows PowerShell)
```powershell
cd D:\Twitter-Hashtag-Prediction-2
javac -encoding UTF-8 -d java java\TwitterDataIngestion.java
java -Xmx2g -cp java TwitterDataIngestion
```
Creates `output\hashtags.csv` with lines like `2017-10-31,halloween` (old file is overwritten). Progress prints every 100,000 tweets.

## 9. Stage 2: HDFS (WSL Ubuntu)
Start Hadoop if it is not running (`start-dfs.sh`; `start-yarn.sh` if you use YARN), then:
```bash
cd /mnt/d/Twitter-Hashtag-Prediction-2
hdfs dfs -mkdir -p /twitter-hashtag/input
hdfs dfs -put -f output/hashtags.csv /twitter-hashtag/input/hashtags.csv
hdfs dfs -ls -h /twitter-hashtag/input
hdfs dfs -cat /twitter-hashtag/input/hashtags.csv | head
# remove old MapReduce output before re-running (a job fails if its output folder exists)
hdfs dfs -rm -r -f /twitter-hashtag/output /twitter-hashtag/daily-output
```

## 10. Stage 3: MapReduce (WSL Ubuntu)
Compile both jobs into one jar:
```bash
mkdir -p build
javac --release 11 -cp "$(hadoop classpath)" -d build hadoop/HashtagCount.java hadoop/DailyHashtagCount.java
jar cf build/hashtag-mr.jar -C build .
```
Run:
```bash
hadoop jar build/hashtag-mr.jar HashtagCount      /twitter-hashtag/input/hashtags.csv /twitter-hashtag/output
hadoop jar build/hashtag-mr.jar DailyHashtagCount /twitter-hashtag/input/hashtags.csv /twitter-hashtag/daily-output
```
Check and copy the results to Windows (the folder is shared via `/mnt/d`):
```bash
hdfs dfs -cat /twitter-hashtag/output/part-r-00000 | sort -k2 -nr | head
hdfs dfs -getmerge /twitter-hashtag/output       output/hashtag_totals.tsv
hdfs dfs -getmerge /twitter-hashtag/daily-output output/daily_counts.tsv
```
Sanity check: the sum of the counts in `hashtag_totals.tsv` must equal the number of lines in `output/hashtags.csv` (`wc -l`).

## 11. Stage 4-6: Python (Windows PowerShell)
```powershell
cd D:\Twitter-Hashtag-Prediction-2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python python\preprocess.py
python python\train_model.py
python python\predict.py ios11
```

## 12. Dashboard
```powershell
streamlit run dashboard\app.py
```
Open http://localhost:8501.

## 13. Features (all explained in `python/preprocess.py`)
`lag0..lag3` (last 4 daily counts), `ma3`, `ma7` (moving averages), `growth` (change vs yesterday), `dow` (weekday of the predicted day), `hist_mean` (historical average). Target: next-day count. Popularity class thresholds come from the training data only.

## 14. Expected output
* Ingestion: a summary with tweets parsed, tweets with hashtags, hashtag occurrences.
* MapReduce: `part-r-00000` files with `hashtag<TAB>count` and `date,hashtag<TAB>count`.
* `train_model.py`: validation and test metrics for 4 models and `models/hashtag_model.joblib`, `models/metrics.json`.
* `predict.py`: last observed count, predicted next-day frequency and Low/Medium/High.

## 15. Results (fill in from your own run; see `models/metrics.json`)
| Item | Value |
|---|---|
| Tweets parsed | _from ingestion output_ |
| Hashtag occurrences / unique hashtags | _from preprocess.py output_ |
| Selected model | _from train_model.py_ |
| Test MAE / RMSE / R2 | _from metrics.json_ |
| Class accuracy / F1 | _from metrics.json_ |

## 16. Limitations
* Customer-support tweets, not general Twitter; hashtags are few (~8% of tweets).
* Only about 2 months (2017-10-04 to 2017-12-03) have enough data; no seasonality can be learned.
* Daily counts are small and noisy; external events cannot be predicted.
* Offline dataset (X API 402 error): no live prediction.
* Hadoop runs on a single laptop (pseudo-distributed), so no real cluster speed-up.

## 17. Future scope
More data and longer history, hashtag co-occurrence/graph features, sentiment features, time-series models (ARIMA/LSTM) if more data is available, live API streaming (Kafka/Spark) when API access is available.

## 18. Troubleshooting
* `Output directory ... already exists`: run the `hdfs dfs -rm -r -f` command in section 9.
* `UnsupportedClassVersionError` when running the jar: recompile with the same Java that Hadoop uses (`--release 8` or `11`).
* `hadoop: command not found` in WSL: `export HADOOP_HOME=/mnt/c/hadoop/hadoop-3.5.0; export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin`.
* `ERROR: ... not found: output/daily_counts.tsv`: run the `getmerge` commands in section 10.
* Do not upload `.env` (it contains your X token) to GitHub or share it with the zip.
