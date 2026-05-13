import os
import shutil

import pandas as pd
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from utils import create_spark_session


spark = create_spark_session("ReadMate ALS Recommendation Model")

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

books = spark.read.option("header", True).option("inferSchema", True).csv(
    "outputs/tables/books_clean_csv"
)

reviews = spark.read.option("header", True).option("inferSchema", True).csv(
    "outputs/tables/reviews_clean_csv"
)

reviews = reviews.select(
    "user_id",
    "book_id",
    "rating"
).dropna()

reviews = reviews.filter((reviews.rating >= 1) & (reviews.rating <= 5))

# Ensure rating is numeric for ALS
reviews = reviews.withColumn("rating", col("rating").cast("float"))

# Full dataset is loaded and processed by Spark.
# For local laptop training, this limits ALS workload while keeping Spark processing.
SAMPLE_FRACTION = 0.10

reviews_sample = reviews.sample(
    withReplacement=False,
    fraction=SAMPLE_FRACTION,
    seed=42
)

print("Full reviews count:", reviews.count())
print("Training sample count:", reviews_sample.count())

user_indexer = StringIndexer(
    inputCol="user_id",
    outputCol="user_index",
    handleInvalid="skip"
)

book_indexer = StringIndexer(
    inputCol="book_id",
    outputCol="book_index",
    handleInvalid="skip"
)

user_indexer_model = user_indexer.fit(reviews_sample)
reviews_indexed = user_indexer_model.transform(reviews_sample)

book_indexer_model = book_indexer.fit(reviews_indexed)
reviews_indexed = book_indexer_model.transform(reviews_indexed)

ratings = reviews_indexed.select(
    "user_id",
    "book_id",
    "user_index",
    "book_index",
    "rating"
).dropna()

train_data, test_data = ratings.randomSplit([0.8, 0.2], seed=42)

als = ALS(
    userCol="user_index",
    itemCol="book_index",
    ratingCol="rating",
    rank=20,
    maxIter=10,
    regParam=0.1,
    coldStartStrategy="drop",
    nonnegative=True
)

model = als.fit(train_data)

predictions = model.transform(test_data)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)

print("Model RMSE:", rmse)

paths_to_remove = [
    "outputs/models/als_model",
    "outputs/models/user_indexer_model",
    "outputs/models/book_indexer_model",
    "outputs/tables/indexed_ratings_csv",
]

for path in paths_to_remove:
    if os.path.exists(path):
        shutil.rmtree(path)

model.write().overwrite().save("outputs/models/als_model")
user_indexer_model.write().overwrite().save("outputs/models/user_indexer_model")
book_indexer_model.write().overwrite().save("outputs/models/book_indexer_model")

ratings.write.mode("overwrite").option("header", True).csv(
    "outputs/tables/indexed_ratings_csv"
)

# Export book index map for cold-start / new user recommendations
book_index_map = reviews_indexed.select(
    "book_id",
    "book_index"
).distinct().join(
    books.select(
        "book_id",
        "title",
        "average_rating",
        "ratings_count",
        "publication_year"
    ),
    "book_id",
    "left"
)

book_index_map_pdf = book_index_map.toPandas()
book_index_map_pdf["book_index"] = book_index_map_pdf["book_index"].astype(int)
book_index_map_pdf.to_csv(
    "outputs/tables/book_index_map.csv",
    index=False,
    encoding="utf-8"
)

# Save item factor vectors for cold-start inference in the app
item_factors_pdf = model.itemFactors.toPandas()
item_factors_pdf["book_index"] = item_factors_pdf["id"].astype(int)
item_factors_pdf["features"] = item_factors_pdf["features"].apply(lambda vec: list(vec))
item_factors_pdf[["book_index", "features"]].to_csv(
    "outputs/tables/item_factors.csv",
    index=False,
    encoding="utf-8"
)

with open("outputs/tables/model_metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Full reviews count: {reviews.count()}\n")
    f.write(f"Training sample count: {reviews_sample.count()}\n")
    f.write(f"Sample fraction: {SAMPLE_FRACTION}\n")
    f.write(f"ALS RMSE: {rmse}\n")
    f.write("Model: Spark MLlib ALS Collaborative Filtering\n")

print("Model saved successfully.")

spark.stop()