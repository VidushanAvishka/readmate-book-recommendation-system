import datetime
import os
import shutil
import matplotlib.pyplot as plt

from pyspark.sql.functions import (
    col,
    count,
    avg,
    desc,
    when,
    split,
    size,
    trim,
    regexp_replace,
    lower,
    min as spark_min,
    max as spark_max,
    stddev,
    round
)
from utils import create_spark_session


def log_progress(message: str):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {message}")


def sanitize_text(value):
    if value is None:
        return ""
    return "".join(ch for ch in str(value) if ord(ch) >= 32)


def normalize(df, column_name, normalized_name):
    bounds = df.select(
        spark_min(col(column_name)).alias("min_value"),
        spark_max(col(column_name)).alias("max_value")
    ).collect()[0]
    min_value = bounds["min_value"]
    max_value = bounds["max_value"]

    if min_value is None or max_value is None or min_value == max_value:
        return df.withColumn(normalized_name, when(col(column_name).isNull(), 0.0).otherwise(0.0))

    return df.withColumn(
        normalized_name,
        ((col(column_name) - min_value) / (max_value - min_value)).cast("double")
    )


def safe_to_pandas(df, limit=10000):
    """Convert a Spark DataFrame to pandas safely by limiting row count for local plotting."""
    return df.limit(limit).toPandas()


spark = create_spark_session("ReadMate Big Data Analytics")

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# Step 1: Load Data
books = spark.read.options(
    header=True,
    inferSchema=True,
    multiLine=True,
    escape="\\",
    quote='"',
    mode="DROPMALFORMED"
).csv("outputs/tables/books_clean_csv")
reviews = spark.read.options(
    header=True,
    inferSchema=True,
    multiLine=True,
    escape="\\",
    quote='"',
    mode="DROPMALFORMED"
).csv("outputs/tables/reviews_clean_csv")

books = books.withColumn("average_rating", col("average_rating").cast("double")) \
    .withColumn("ratings_count", col("ratings_count").cast("int")) \
    .withColumn("text_reviews_count", col("text_reviews_count").cast("int")) \
    .withColumn("publication_year", col("publication_year").cast("int"))

reviews = reviews.withColumn("rating", col("rating").cast("double"))

books = books.filter(
    col("book_id").isNotNull() &
    (trim(col("book_id")) != "") &
    col("title").isNotNull() &
    (trim(col("title")) != "")
)

reviews = reviews.filter(
    col("user_id").isNotNull() &
    (trim(col("user_id")) != "") &
    col("book_id").isNotNull() &
    (trim(col("book_id")) != "") &
    col("rating").isNotNull()
)

log_progress("STEP 1: DATA LOADING START")
log_progress("Reading book and review CSV data")
print("Books schema:")
books.printSchema()
print("Sample books records:")
books.show(5, truncate=False)
print("Reviews schema:")
reviews.printSchema()
print("Sample reviews records:")
reviews.show(5, truncate=False)

# Step 1b: Clean Data
log_progress("STEP 1B: DATA CLEANING START")
books_clean = books.dropDuplicates(["book_id"]).filter(col("book_id").isNotNull())
reviews_clean = reviews.dropDuplicates(["user_id", "book_id", "review_text", "date_added"]).filter(
    col("user_id").isNotNull() &
    col("book_id").isNotNull() &
    col("rating").isNotNull()
)

log_progress(f"Clean books count: {books_clean.count()}")
log_progress(f"Clean reviews count: {reviews_clean.count()}")
log_progress("STEP 1B: DATA CLEANING COMPLETE")

# Step 2: Join Reviews with Book Metadata
books_meta = books_clean.select(
    col("book_id"),
    col("title").alias("book_title"),
    col("authors").alias("book_authors"),
    col("average_rating"),
    col("ratings_count"),
    col("text_reviews_count"),
    col("publication_year"),
    col("description").alias("book_description")
)

joined = reviews_clean.alias("r").join(
    books_meta.alias("b"),
    on=["book_id"],
    how="inner"
)

log_progress("STEP 2: JOIN REVIEWS WITH BOOK METADATA START")
print("Joined schema:")
joined.printSchema()
print("Sample joined records:")
joined.select(
    "user_id",
    "book_id",
    "book_title",
    "rating",
    "publication_year",
    "review_text"
).show(5, truncate=False)
log_progress("STEP 2: JOIN COMPLETE")

# Step 3: Descriptive Analytics - Most Reviewed Books
most_reviewed_books = joined.groupBy(
    "book_id",
    "book_title",
    "book_authors",
    "publication_year",
    "average_rating",
    "ratings_count"
).agg(
    count("*").alias("review_count"),
    avg("rating").alias("avg_user_rating")
).orderBy(desc("review_count"))

top_reviewed_books = most_reviewed_books.limit(10)
log_progress("STEP 3: MOST REVIEWED BOOKS ANALYSIS START")
print("Top 10 most reviewed books:")
top_reviewed_books.show(10, truncate=False)
log_progress("STEP 3: MOST REVIEWED BOOKS ANALYSIS COMPLETE")

# Step 4: Visualization - Top 10 Most Reviewed Books
top_reviewed_books_pdf = safe_to_pandas(top_reviewed_books)
top_reviewed_books_pdf["book_title"] = top_reviewed_books_pdf["book_title"].apply(sanitize_text)

plt.figure(figsize=(12, 7))
plt.barh(
    top_reviewed_books_pdf["book_title"].iloc[::-1],
    top_reviewed_books_pdf["review_count"].iloc[::-1]
)
plt.title("Top 10 Most Reviewed Books")
plt.xlabel("Review Count")
plt.ylabel("Book Title")
plt.tight_layout()
plt.savefig("outputs/plots/top_reviewed_books.png", bbox_inches="tight")
plt.close()
log_progress("Saved top reviewed books plot")

# Step 5: Descriptive Analytics - Review Length Analysis
review_text_clean = when(
    col("review_text").isNull() | (trim(col("review_text")) == ""),
    ""
).otherwise(regexp_replace(trim(col("review_text")), "\\s+", " "))

review_length_df = joined.withColumn("review_text_clean", review_text_clean).withColumn(
    "review_length",
    when(col("review_text_clean") == "", 0).otherwise(size(split(col("review_text_clean"), " ")))
)

log_progress("STEP 5: REVIEW LENGTH ANALYSIS START")
review_length_summary = review_length_df.agg(
    round(avg("review_length"), 2).alias("avg_review_length"),
    spark_min("review_length").alias("min_review_length"),
    spark_max("review_length").alias("max_review_length"),
    round(stddev("review_length"), 2).alias("stddev_review_length")
)
review_length_summary.show(truncate=False)

review_length_categories = review_length_df.withColumn(
    "review_length_category",
    when(col("review_length") <= 20, "Short")
    .when(col("review_length") <= 50, "Medium")
    .otherwise("Long")
)

review_length_category_counts = review_length_categories.groupBy("review_length_category").count().orderBy(desc("count"))
log_progress("Review length categories computed")
review_length_category_counts.show(truncate=False)

# Step 6: Visualization - Review Length Distribution
review_length_pdf = safe_to_pandas(review_length_df.select("review_length"), limit=10000)
plt.figure(figsize=(10, 6))
plt.hist(review_length_pdf["review_length"].dropna(), bins=30, color="#1f77b4", edgecolor="black", alpha=0.8)
plt.title("Review Length Distribution")
plt.xlabel("Review Length (words)")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/plots/review_length_distribution.png", bbox_inches="tight")
plt.close()
log_progress("Saved review length distribution plot")

# Step 7: Diagnostic Analytics - Review Count vs Average Rating
review_count_vs_rating = joined.groupBy("book_id", "book_title").agg(
    count("*").alias("review_count"),
    round(avg("rating"), 2).alias("average_rating")
).orderBy(desc("review_count"))

log_progress("STEP 7: REVIEW COUNT VS AVERAGE RATING START")
review_count_vs_rating.show(10, truncate=False)
log_progress("STEP 7: REVIEW COUNT VS AVERAGE RATING COMPLETE")

# Step 8: Visualization - Review Count vs Average Rating
review_count_vs_rating_pdf = safe_to_pandas(review_count_vs_rating, limit=1000)
plt.figure(figsize=(10, 6))
plt.scatter(
    review_count_vs_rating_pdf["review_count"],
    review_count_vs_rating_pdf["average_rating"],
    alpha=0.6,
    edgecolor="k"
)
plt.title("Review Count vs Average Rating")
plt.xlabel("Review Count")
plt.ylabel("Average Rating")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("outputs/plots/review_count_vs_average_rating.png", bbox_inches="tight")
plt.close()
log_progress("Saved review count vs average rating plot")

# Step 9: Diagnostic Analytics - Most Active Users
most_active_users = reviews_clean.groupBy("user_id").agg(
    count("*").alias("total_reviews"),
    round(avg("rating"), 2).alias("average_rating")
).orderBy(desc("total_reviews")).limit(10)

log_progress("STEP 9: MOST ACTIVE USERS ANALYSIS START")
most_active_users.show(truncate=False)
log_progress("STEP 9: MOST ACTIVE USERS ANALYSIS COMPLETE")

# Step 10: Publication Year Trend Analytics
VALID_PUBLICATION_YEAR_MIN = 1800
VALID_PUBLICATION_YEAR_MAX = 2025
publication_year_trends = joined.withColumn("publication_year", col("publication_year").cast("int")).filter(
    col("publication_year").between(VALID_PUBLICATION_YEAR_MIN, VALID_PUBLICATION_YEAR_MAX)
).groupBy("publication_year").agg(
    count("*").alias("review_count"),
    round(avg("rating"), 2).alias("average_rating")
).orderBy("publication_year")

log_progress("STEP 10: PUBLICATION YEAR TREND ANALYTICS START")
publication_year_trends.show(10, truncate=False)
log_progress("STEP 10: PUBLICATION YEAR TREND ANALYTICS COMPLETE")

# Step 11: Visualization - Reviews by Book Publication Year
publication_year_pdf = publication_year_trends.toPandas()
plt.figure(figsize=(12, 6))
plt.plot(
    publication_year_pdf["publication_year"],
    publication_year_pdf["review_count"],
    marker="o",
    linestyle="-",
    linewidth=1,
    markersize=4
)
plt.title("Reviews by Book Publication Year")
plt.xlabel("Publication Year")
plt.ylabel("Review Count")
plt.tight_layout()
plt.savefig("outputs/plots/review_count_by_publication_year.png", bbox_inches="tight")
plt.close()
log_progress("Saved publication year trend plot")

# Step 12: Predictive Analytics - Most Rateable Books in the Future
rateability_base = joined.groupBy(
    "book_id",
    "book_title",
    "book_authors",
    "average_rating",
    "ratings_count",
    "text_reviews_count"
).agg(
    count("*").alias("review_count")
)

rateability_base = rateability_base.fillna({
    "ratings_count": 0,
    "text_reviews_count": 0,
    "average_rating": 0.0,
    "review_count": 0
})

rateability_norm = normalize(rateability_base, "ratings_count", "norm_ratings_count")
rateability_norm = normalize(rateability_norm, "text_reviews_count", "norm_text_reviews_count")
rateability_norm = normalize(rateability_norm, "review_count", "norm_review_count")

predicted_rateable_books = rateability_norm.withColumn(
    "weighted_rateability_score",
    round(
        col("average_rating") * 0.4 +
        col("norm_ratings_count") * 0.25 +
        col("norm_text_reviews_count") * 0.2 +
        col("norm_review_count") * 0.15,
        4
    )
).orderBy(desc("weighted_rateability_score"))

top_predicted_rateable = predicted_rateable_books.limit(10)
log_progress("STEP 12: PREDICTIVE RATEABILITY ANALYTICS START")
top_predicted_rateable.select(
    "book_title",
    "book_authors",
    "weighted_rateability_score",
    "average_rating",
    "ratings_count",
    "text_reviews_count",
    "review_count"
).show(10, truncate=False)
log_progress("STEP 12: PREDICTIVE RATEABILITY ANALYTICS COMPLETE")

# Step 13: Visualization - Top 10 Predicted Most Rateable Books
top_predicted_rateable_pdf = top_predicted_rateable.toPandas()
top_predicted_rateable_pdf["book_title"] = top_predicted_rateable_pdf["book_title"].apply(sanitize_text)

plt.figure(figsize=(12, 7))
plt.barh(
    top_predicted_rateable_pdf["book_title"].iloc[::-1],
    top_predicted_rateable_pdf["weighted_rateability_score"].iloc[::-1],
    color="#2ca02c"
)
plt.title("Top 10 Predicted Most Rateable Books")
plt.xlabel("Weighted Rateability Score")
plt.ylabel("Book Title")
plt.tight_layout()
plt.savefig("outputs/plots/predicted_most_rateable_books.png", bbox_inches="tight")
plt.close()
log_progress("Saved predicted most rateable books plot")

outputs = {
    "top_reviewed_books_csv": top_reviewed_books,
    "review_length_summary_csv": review_length_summary,
    "review_length_categories_csv": review_length_category_counts,
    "review_count_vs_rating_csv": review_count_vs_rating,
    "most_active_users_csv": most_active_users,
    "publication_year_trends_csv": publication_year_trends,
    "predicted_rateable_books_csv": top_predicted_rateable
}

log_progress("STEP 14: SAVING ANALYTICS OUTPUTS START")
for folder_name, dataframe in outputs.items():
    output_path = f"outputs/tables/{folder_name}"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    dataframe.write.mode("overwrite").option("header", True).csv(output_path)
    log_progress(f"Saved output table: {folder_name}")
log_progress("STEP 14: SAVING ANALYTICS OUTPUTS COMPLETE")

log_progress("Analytics pipeline completed successfully")

spark.stop()
