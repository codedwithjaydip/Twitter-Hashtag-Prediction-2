# Viva questions and simple answers

1. **What is Big Data?** Data that is too large, too fast or too varied for one normal program/computer to handle easily. Here: 2.8 million tweets (about 500 MB).
2. **Why Hadoop?** It stores data across machines (HDFS) and processes it in parallel (MapReduce), so the same program scales from a laptop to a cluster. It is also the standard Big Data tool taught in the course.
3. **What is HDFS?** Hadoop Distributed File System. It splits files into blocks, stores them (with copies) on many nodes and gives one file-system view.
4. **What is MapReduce?** A programming model with two steps: Map turns each input record into key-value pairs, Reduce combines all values of the same key.
5. **What does the Mapper do?** It reads a line `date,hashtag` and emits `hashtag, 1` (Job 1) or `date,hashtag, 1` (Job 2).
6. **What does the Reducer do?** It receives one key with all its 1s and adds them to give the total count.
7. **Why Java?** Hadoop is written in Java and its MapReduce API is Java. Java is also fast for reading the 500 MB CSV.
8. **Why Python?** It has pandas and scikit-learn for data preparation and machine learning, and Streamlit for the dashboard.
9. **Why Machine Learning?** Counting only tells the past. ML learns patterns in past daily counts to estimate tomorrow's count.
10. **What is hashtag popularity?** How many times a hashtag is used in tweets in a time period (here, per day).
11. **How is popularity calculated?** MapReduce counts every `(date, hashtag)`. The model predicts tomorrow's count; it is labelled Low / Medium / High using thresholds from the training data (75th and 95th percentile).
12. **Why is time important?** Popularity rises and falls; yesterday's counts help predict tomorrow's. Weekday also matters.
13. **Why chronological split?** A random split would put future days in training and test on the past, which leaks information and gives unrealistically good results. We train on older days and test on newer days.
14. **What features are used?** lag0-lag3 (counts of the last 4 days), ma3 and ma7 (moving averages), growth (rise/fall vs yesterday), dow (weekday of predicted day), hist_mean (average historical popularity).
15. **What is the target variable?** The hashtag's count on the next day.
16. **Which ML model and why?** Four models are compared on a validation period: persistence baseline, Linear Regression, Random Forest, Gradient Boosting. The one with the lowest validation RMSE is saved (see `models/metrics.json`). Tree models capture non-linear patterns such as sudden spikes.
17. **What metrics are used?** MAE, RMSE and R2 for the count; accuracy, precision, recall and F1 for the Low/Medium/High class.
18. **What are the limitations?** Only about 2 months of dense data; customer-support tweets, not general Twitter; counts are small; spikes caused by outside events are not predictable; no live data; only single-machine (pseudo-distributed) Hadoop.
19. **Why wasn't the X API used?** It returned HTTP 402 Payment Required (credits depleted), so a public dataset was used instead.
20. **Role of HDFS?** It stores the extracted `date,hashtag` file and is the input/output location for MapReduce jobs.
