# readmate-book-recommendation-system
Big Data Analytics and Personalized Book Recommendation System Using Apache Spark

# 📚 ReadMate: Big Data Analytics and Personalized Book Recommendation System Using Apache Spark

A Big Data Analytics + Recommendation System project built using **Apache Spark (PySpark)**, **Python**, and **Streamlit**, using the Goodreads Fantasy & Paranormal datasets.

This project was developed for a **Big Data Analytics Mini Project Assignment** using Apache Spark for large-scale data processing and collaborative filtering recommendation generation.

---

# Project Overview

This system combines:

- **Big Data Analytics**
- **Data Cleaning & Transformation**
- **Exploratory Data Analysis (EDA)**
- **Collaborative Filtering Recommendation System**
- **Interactive Web UI**

The system processes over **3.4 million Goodreads reviews** and **258,000+ books** using Apache Spark.

---

# Features

## Big Data Analytics
✔ Load large Goodreads datasets with PySpark  
✔ Data cleaning and preprocessing  
✔ Rating distribution analysis  
✔ Review activity analysis  
✔ Publication year trend analysis  
✔ Book popularity analysis  
✔ Most active users analysis  
✔ Review text length analysis  

## Recommendation System
✔ Collaborative Filtering using Spark MLlib ALS  
✔ Personalized Top-N book recommendations  
✔ Model evaluation using RMSE  

## UI
✔ Streamlit dashboard  
✔ Analytics visualizations  
✔ Recommendation browser  
✔ Model metrics viewer  

---

# Dataset Information

This project uses the Goodreads Fantasy & Paranormal datasets.

Dataset source:

**UCSD Book Graph Dataset**
https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home

Download source:

https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html

Required datasets:

### Books Dataset
goodreads_books_fantasy_paranormal.json.gz

Direct link:
https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books_fantasy_paranormal.json.gz

---

### Reviews Dataset
goodreads_reviews_fantasy_paranormal.json.gz

Direct link:
https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_reviews_fantasy_paranormal.json.gz

---

# Dataset Statistics

## Books Dataset
- Total books: **258,585**
- Domain: Fantasy / Paranormal
- JSON format

Contains:
- book_id
- title
- authors
- publication_year
- average_rating
- ratings_count
- description
- genres

---

## Reviews Dataset
- Total reviews: **3,424,641**
- JSON format

Contains:
- user_id
- book_id
- rating
- review_text
- date_added
- date_updated

---

# Technologies Used

- Python 3.10+
- Apache Spark / PySpark
- Spark MLlib
- Pandas
- NumPy
- Matplotlib
- Streamlit
- Java JDK 17
- Hadoop Windows binaries (for Windows execution)

---

# Project Structure

```bash
readmate-book-recommendation-system/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── goodreads_books_fantasy_paranormal.json
│   └── goodreads_reviews_fantasy_paranormal.json
│
├── outputs/
│   ├── models/
│   ├── plots/
│   └── tables/
│
├── src/
│   ├── 01_load_clean_data.py
│   ├── 02_big_data_analytics.py
│   ├── 03_train_recommender.py
│   ├── 04_generate_recommendations.py
│   ├── test_spark_write.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

---

# Installation Guide

## 1. Clone Repository

```bash
git clone https://github.com/VidushanAvishka/readmate-book-recommendation-system
cd readmate-book-recommendation-system
```

---

## 2. Create Virtual Environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements file is missing:

```bash
pip install pyspark==3.5.1 pandas numpy matplotlib streamlit pyarrow
```

---

# Windows Spark Setup

This project was tested on Windows.

## Install Java

Install:

Java JDK 17

Example:

https://adoptium.net/

Verify:

```powershell
java -version
```

---

## Install Hadoop Windows Binaries

Create:

```text
C:\hadoop\bin
```

Download:

winutils.exe
hadoop.dll

From:

https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6/bin

Copy files into:

```text
C:\hadoop\bin
```

Final structure:

```text
C:\hadoop
└── bin
    ├── winutils.exe
    └── hadoop.dll
```

---

## Environment Variables

Set:

```text
HADOOP_HOME = C:\hadoop
hadoop.home.dir = C:\hadoop
```

Add to PATH:

```text
C:\hadoop\bin
```

---

# Data Preparation

Download:

```text
goodreads_books_fantasy_paranormal.json.gz
goodreads_reviews_fantasy_paranormal.json.gz
```

Extract `.gz` files.

Place extracted files in:

```text
data/
```

Final:

```text
data/goodreads_books_fantasy_paranormal.json
data/goodreads_reviews_fantasy_paranormal.json
```

---

# How to Run Project

Run scripts in this order.

---

## Step 1: Test Spark Setup

```powershell
python src/test_spark_write.py
```

Expected:

```text
SUCCESS
```

---

## Step 2: Data Cleaning & Transformation

```powershell
python src/01_load_clean_data.py
```

This will:

- Load raw JSON datasets
- Clean invalid rows
- Filter ratings
- Transform columns
- Save cleaned CSV outputs

Generated:

```text
outputs/tables/books_clean_csv
outputs/tables/reviews_clean_csv
```

---

## Step 3: Big Data Analytics

```powershell
python src/02_big_data_analytics.py
```

Generates:

### Plots
```text
rating_distribution.png
top_reviewed_books.png
books_by_publication_year.png
review_count_vs_average_rating.png
review_length_distribution.png
predicted_most_rateable_books.png
```

### Analytics Tables
```text
rating_distribution_csv
top_reviewed_books_csv
books_by_year_csv
most_active_users_csv
book_popularity_score_csv
review_length_summary_csv
```

---

## Step 4: Train Recommendation Model

```powershell
python src/03_train_recommender.py
```

Uses:

Spark MLlib ALS collaborative filtering

Outputs:

```text
outputs/models/als_model
outputs/models/user_indexer_model
outputs/models/book_indexer_model
outputs/tables/indexed_ratings_csv
outputs/tables/model_metrics.txt
```

---

## Step 5: Generate Recommendations

```powershell
python src/04_generate_recommendations.py
```

Outputs:

```text
outputs/tables/final_recommendations_csv
```

---

## Step 6: Run Web UI

```powershell
streamlit run app/streamlit_app.py
```

Open browser:

```text
http://localhost:8501
```

---

# Recommendation Model

Algorithm used:

## ALS (Alternating Least Squares)

Collaborative Filtering using Spark MLlib.

Why ALS?

- Scalable for big data
- Designed for recommendation systems
- Distributed training with Spark
- Handles sparse user-item matrices efficiently

Configuration:

```python
rank = 20
maxIter = 10
regParam = 0.1
coldStartStrategy = "drop"
```

Evaluation metric:

RMSE

---

# Example Outputs

## Analytics
- Rating distribution
- Top reviewed books
- Publication trends
- User activity insights
- Popularity analysis

## Recommendations
Top 10 personalized books for selected users.

---

# Known Issues

## Windows Spark Temp Folder Warning

You may see:

```text
ERROR ShutdownHookManager
Exception while deleting Spark temp dir
```

This is harmless and can be ignored.

---

## Streamlit Warning

You may see:

```text
use_container_width will be removed
```

Non-breaking warning.

---

# Future Improvements

- Content-based recommendation
- Hybrid recommender
- Author similarity recommendations
- Search by title
- Book cover integration
- Deployment to cloud
- Better UI filtering
- Hyperparameter tuning

---

# Assignment Requirements Coverage

This project satisfies:

✔ Apache Spark / PySpark big data processing  
✔ Dataset loading and inspection  
✔ Data cleaning and preprocessing  
✔ Big data analytics  
✔ Visualizations  
✔ Recommendation model  
✔ Spark MLlib implementation  
✔ Result generation  
✔ Interactive system UI  

---

# Author

258803D PHN Muthumali
258832N RDAV Thennakoon

Big Data Analytics Mini Project
Apache Spark Recommendation System