import os
import shutil

from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import col, explode
from utils import create_spark_session


def main():
    spark = create_spark_session("ReadMate Generate Recommendations")

    books = spark.read.option("header", True).option("inferSchema", True).csv(
        "outputs/tables/books_clean_csv"
    )

    indexed_ratings = spark.read.option("header", True).option("inferSchema", True).csv(
        "outputs/tables/indexed_ratings_csv"
    )

    model = ALSModel.load("outputs/models/als_model")

    users = indexed_ratings.select("user_index").distinct().limit(100)

    recommendations = model.recommendForUserSubset(users, 10)

    recommendations_flat = recommendations.select(
        "user_index",
        explode("recommendations").alias("recommendation")
    ).select(
        "user_index",
        col("recommendation.book_index").alias("book_index"),
        col("recommendation.rating").alias("predicted_rating")
    )

    book_lookup = indexed_ratings.select("book_index", "book_id").distinct()

    final_recommendations = recommendations_flat \
        .join(book_lookup, "book_index") \
        .join(
            books.select(
                "book_id",
                "title",
                "average_rating",
                "ratings_count",
                "publication_year"
            ),
            "book_id"
        ) \
        .orderBy("user_index", col("predicted_rating").desc())

    output_path = "outputs/tables/final_recommendations_csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    final_recommendations.write.mode("overwrite").option("header", True).csv(output_path)

    final_recommendations.show(50, truncate=False)

    print("Recommendations generated successfully.")
    print("Saved to:", output_path)

    spark.stop()


if __name__ == "__main__":
    main()