import os
import shutil

from pyspark.sql.functions import col, size, split, regexp_replace, lower, when, trim
from utils import create_spark_session


spark = create_spark_session("ReadMate Load and Clean Data")

BOOKS_PATH = "data/goodreads_books_fantasy_paranormal.json"
REVIEWS_PATH = "data/goodreads_reviews_fantasy_paranormal.json"

OUTPUT_TABLES = "outputs/tables"
BOOKS_OUTPUT = "outputs/tables/books_clean_csv"
REVIEWS_OUTPUT = "outputs/tables/reviews_clean_csv"

os.makedirs(OUTPUT_TABLES, exist_ok=True)

for path in [BOOKS_OUTPUT, REVIEWS_OUTPUT]:
    if os.path.exists(path):
        shutil.rmtree(path)

books = spark.read.json(BOOKS_PATH)
reviews = spark.read.json(REVIEWS_PATH)

print("Raw books count:", books.count())
print("Raw reviews count:", reviews.count())

books_clean = books.select(
    col("book_id").cast("string").alias("book_id"),
    col("title").cast("string").alias("title"),
    col("authors").cast("string").alias("authors"),

    when(trim(col("average_rating")) == "", None)
        .otherwise(col("average_rating"))
        .cast("double")
        .alias("average_rating"),

    when(trim(col("ratings_count")) == "", None)
        .otherwise(col("ratings_count"))
        .cast("int")
        .alias("ratings_count"),

    when(trim(col("text_reviews_count")) == "", None)
        .otherwise(col("text_reviews_count"))
        .cast("int")
        .alias("text_reviews_count"),

    when(trim(col("publication_year")) == "", None)
        .otherwise(col("publication_year"))
        .cast("int")
        .alias("publication_year"),

    col("description").cast("string").alias("description")
).dropna(subset=["book_id", "title"])

books_clean = books_clean.withColumn(
    "description_clean",
    lower(regexp_replace(col("description"), "[^a-zA-Z0-9 ]", ""))
)

books_clean = books_clean.withColumn(
    "description_word_count",
    size(split(col("description_clean"), " "))
)

reviews_clean = reviews.select(
    col("user_id").cast("string").alias("user_id"),
    col("book_id").cast("string").alias("book_id"),

    when(trim(col("rating")) == "", None)
        .otherwise(col("rating"))
        .cast("double")
        .alias("rating"),

    col("review_text").cast("string").alias("review_text"),
    col("date_added").cast("string").alias("date_added")
).dropna(subset=["user_id", "book_id", "rating"])

reviews_clean = reviews_clean.filter(
    (col("rating") >= 1) & (col("rating") <= 5)
)

books_clean.write.mode("overwrite") \
    .option("header", True) \
    .csv(BOOKS_OUTPUT)

reviews_clean.write.mode("overwrite") \
    .option("header", True) \
    .csv(REVIEWS_OUTPUT)

print("Clean books count:", books_clean.count())
print("Clean reviews count:", reviews_clean.count())

print("Books clean saved to:", BOOKS_OUTPUT)
print("Reviews clean saved to:", REVIEWS_OUTPUT)

books_clean.show(5, truncate=False)
reviews_clean.show(5, truncate=False)

spark.stop()