import ast
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="ReadMate Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
WORKFLOW_IMAGE = os.path.join(ASSETS_DIR, "mermaid-diagram.png")


def load_spark_csv_folder(folder_path):
    if not os.path.exists(folder_path):
        return pd.DataFrame()

    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]

    if not files:
        return pd.DataFrame()

    frames = []

    for file in files:
        try:
            frames.append(pd.read_csv(file))
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_new_user_data():
    book_index_map_path = os.path.join(TABLES_DIR, "book_index_map.csv")
    item_factors_path = os.path.join(TABLES_DIR, "item_factors.csv")

    if not os.path.exists(book_index_map_path) or not os.path.exists(item_factors_path):
        return None, None

    book_index_map = pd.read_csv(book_index_map_path)
    item_factors = pd.read_csv(
        item_factors_path,
        converters={"features": ast.literal_eval}
    )

    book_index_map["book_index"] = book_index_map["book_index"].astype(int)
    item_factors["book_index"] = item_factors["book_index"].astype(int)

    return book_index_map, item_factors


def compute_new_user_recommendations(
    selected_titles,
    selected_ratings,
    book_index_map,
    item_factors,
    top_n=10
):
    if not selected_titles:
        return pd.DataFrame()

    title_index_map = (
        book_index_map
        .drop_duplicates(subset=["title"])
        .set_index("title")[["book_id", "book_index"]]
    )

    selected_rows = []

    for title, rating in zip(selected_titles, selected_ratings):
        if title in title_index_map.index:
            row = title_index_map.loc[title]
            selected_rows.append({
                "title": title,
                "rating": float(rating),
                "book_id": int(row["book_id"]),
                "book_index": int(row["book_index"])
            })

    if not selected_rows:
        return pd.DataFrame()

    selected_df = pd.DataFrame(selected_rows)
    selected_factors = selected_df.merge(item_factors, on="book_index", how="inner")

    if selected_factors.empty:
        return pd.DataFrame()

    feature_matrix = np.vstack(selected_factors["features"].values)
    weights = selected_factors["rating"].to_numpy(dtype=float)

    user_vector = (feature_matrix * weights[:, None]).sum(axis=0)
    user_vector = user_vector / max(weights.sum(), 1.0)

    all_factors = np.vstack(item_factors["features"].values)
    scores = all_factors.dot(user_vector)

    scored_items = item_factors.copy()
    scored_items["score"] = scores

    selected_book_ids = selected_df["book_id"].unique().tolist()

    recs = scored_items.merge(book_index_map, on="book_index", how="left")
    recs = recs[~recs["book_id"].isin(selected_book_ids)]
    recs = recs.sort_values("score", ascending=False)

    return (
        recs[
            [
                "book_id",
                "title",
                "score",
                "average_rating",
                "ratings_count",
                "publication_year"
            ]
        ]
        .drop_duplicates(subset=["book_id"])
        .head(top_n)
    )


st.title("📚 ReadMate: Personalized Book Recommendation System")
st.write("Big Data Analytics and Recommendation System using Apache Spark")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Dataset Explorer",
        "Data Preprocessing",
        "Big Data Analytics",
        "Diagnostic Analytics",
        "Feature Engineering",
        "ALS Model Training",
        "Recommendation Results",
        "New User Recommendations",
        "Model Evaluation"
    ]
)


if page == "Project Overview":
    st.header("Project Overview")

    st.write("""
    This project builds a personalized book recommendation system using Apache Spark
    and the Goodreads Fantasy & Paranormal dataset. The system performs big data
    processing, data cleaning, exploratory analytics, diagnostic analytics, feature
    engineering, collaborative filtering using ALS, recommendation generation, and
    model evaluation.
    """)

    st.subheader("Project Objectives")
    st.markdown("""
    - Analyze a large-scale Goodreads dataset using Apache Spark.
    - Perform data preprocessing and transformation using PySpark.
    - Generate meaningful visual insights from book rating and review data.
    - Build a personalized book recommendation system.
    - Use ALS collaborative filtering for recommendation generation.
    - Evaluate the recommendation model using RMSE.
    - Provide an interactive Streamlit dashboard for demonstration.
    """)

    st.subheader("Datasets Used")
    st.markdown("""
    - `goodreads_books_fantasy_paranormal.json.gz`
    - `goodreads_reviews_fantasy_paranormal.json.gz`
    """)

    st.subheader("Workflow")

    if os.path.exists(WORKFLOW_IMAGE):
        st.image(Image.open(WORKFLOW_IMAGE), use_container_width=True)
    else:
        st.warning("Workflow image not found. Please place mermaid-diagram.png inside the assets folder.")


elif page == "Dataset Explorer":
    st.header("Dataset Explorer")

    st.write("""
    This section describes the datasets used in the project. The project uses the
    Goodreads Fantasy & Paranormal subset, which contains book metadata and detailed
    user reviews.
    """)

    dataset_info = pd.DataFrame({
        "Dataset": [
            "goodreads_books_fantasy_paranormal.json.gz",
            "goodreads_reviews_fantasy_paranormal.json.gz"
        ],
        "Description": [
            "Contains book metadata such as title, authors, publication year, average rating, and ratings count.",
            "Contains detailed user reviews and rating information used for analytics and recommendation modeling."
        ],
        "Project Usage": [
            "Book information, metadata analysis, and recommendation output display.",
            "Rating analysis, review analytics, and ALS recommendation model training."
        ]
    })

    st.dataframe(dataset_info, use_container_width=True)

    st.subheader("Dataset Scale")
    st.markdown("""
    - Books: approximately **258,585**
    - Detailed reviews: approximately **3,424,641**
    """)

    st.info("Dataset source: UCSD Goodreads Book Graph dataset.")


elif page == "Data Preprocessing":
    st.header("Data Preprocessing")

    st.write("""
    Data preprocessing prepares the raw Goodreads JSON files for analytics and
    recommendation model training. Since the dataset is large, Apache Spark is used
    to load, clean, transform, and save processed data efficiently.
    """)

    st.subheader("Preprocessing Steps")
    st.markdown("""
    - Load raw JSON dataset files using PySpark.
    - Select required columns from books and reviews.
    - Remove null, missing, duplicate, or invalid records.
    - Convert fields such as rating, average rating, ratings count, and publication year into proper numeric types.
    - Clean book titles and metadata.
    - Join book information with review and rating data where required.
    - Save cleaned datasets for analytics and model training.
    """)

    st.subheader("Expected Outputs")
    st.markdown("""
    - Cleaned book metadata
    - Cleaned rating and review data
    - Prepared tables for analytics
    - Prepared input data for ALS model training
    """)

    st.code("Run: spark-submit src/01_data_preprocessing.py", language="bash")


elif page == "Big Data Analytics":
    st.header("Big Data Analytics")

    st.write("""
    This section presents all analytics visualizations generated from the Goodreads
    Fantasy & Paranormal dataset using Apache Spark.
    """)

    if not os.path.exists(PLOTS_DIR):
        st.warning("Plots directory not found. Run src/02_big_data_analytics.py first.")
    else:
        plot_files = sorted(
            [
                file for file in os.listdir(PLOTS_DIR)
                if file.lower().endswith(".png")
            ]
        )

        if not plot_files:
            st.warning("No plot images found. Run src/02_big_data_analytics.py first.")
        else:
            cols = st.columns(2)

            for i, plot_file in enumerate(plot_files):
                plot_path = os.path.join(PLOTS_DIR, plot_file)

                plot_title = (
                    plot_file
                    .replace(".png", "")
                    .replace("_", " ")
                    .title()
                )

                with cols[i % 2]:
                    st.subheader(plot_title)
                    st.image(Image.open(plot_path), use_container_width=True)


elif page == "Diagnostic Analytics":
    st.header("Diagnostic Analytics")

    st.write("""
    Diagnostic analytics explains patterns and relationships in the dataset.
    It helps identify whether popular books also have higher ratings, how review
    activity changes over time, and how publication year affects user engagement.
    """)

    review_vs_rating_plot = os.path.join(PLOTS_DIR, "review_count_vs_average_rating.png")
    review_count_plot = os.path.join(PLOTS_DIR, "review_count_by_publication_year.png")
    year_plot = os.path.join(PLOTS_DIR, "books_by_publication_year.png")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Review Count vs Average Rating")
        if os.path.exists(review_vs_rating_plot):
            st.image(Image.open(review_vs_rating_plot), use_container_width=True)
        else:
            st.warning("Run src/02_big_data_analytics.py first.")

    with col2:
        st.subheader("Review Count by Publication Year")
        if os.path.exists(review_count_plot):
            st.image(Image.open(review_count_plot), use_container_width=True)
        else:
            st.warning("Run src/02_big_data_analytics.py first.")

    st.subheader("Publication Trend Analysis")
    if os.path.exists(year_plot):
        st.image(Image.open(year_plot), use_container_width=True)
    else:
        st.warning("Run src/02_big_data_analytics.py first.")

    st.subheader("Interpretation")
    st.markdown("""
    - Books with higher review counts are usually more popular or widely read.
    - Higher popularity does not always guarantee a higher average rating.
    - Publication year trends help identify growth patterns in the selected genre.
    - Diagnostic analytics supports better understanding before model training.
    """)


elif page == "Feature Engineering":
    st.header("Feature Engineering")

    st.write("""
    Feature engineering converts raw Goodreads data into a format suitable for
    machine learning. Spark MLlib ALS requires numeric user indexes, numeric item
    indexes, and rating values.
    """)

    feature_df = pd.DataFrame({
        "Feature": [
            "user_index",
            "book_index",
            "rating",
            "book_id",
            "title",
            "average_rating",
            "ratings_count",
            "publication_year",
            "item_factors"
        ],
        "Purpose": [
            "Numeric user identifier used by ALS.",
            "Numeric book identifier used by ALS.",
            "User rating used as model input.",
            "Original Goodreads book identifier.",
            "Book title used for displaying recommendations.",
            "Book-level average rating used for interpretation.",
            "Popularity indicator based on number of ratings.",
            "Used for trend analysis and display.",
            "Latent book feature vectors learned by ALS."
        ]
    })

    st.dataframe(feature_df, use_container_width=True)

    st.subheader("ALS Input Format")
    st.code(
        """
user_id  → user_index
book_id  → book_index
rating   → ALS input rating
        """,
        language="text"
    )

    st.subheader("Why Feature Engineering is Important")
    st.markdown("""
    - Converts raw IDs into numeric indexes.
    - Reduces data into useful modeling features.
    - Allows Spark MLlib to train the ALS model.
    - Enables recommendation outputs to be mapped back to real book titles.
    - Supports recommendations for a new user using item factor vectors.
    """)


elif page == "ALS Model Training":
    st.header("ALS Model Training")

    st.write("""
    The recommendation engine uses Alternating Least Squares, a collaborative
    filtering algorithm from Spark MLlib. ALS learns hidden user preference factors
    and book feature factors from user-book rating interactions.
    """)

    st.subheader("Training Workflow")
    st.markdown("""
    - Load cleaned review and rating data.
    - Convert users and books into numeric indexes.
    - Split data into training and testing sets.
    - Train the ALS collaborative filtering model.
    - Generate Top-N recommendations for selected users.
    - Extract and save item factors for new-user recommendation.
    - Evaluate model performance using RMSE.
    """)

    model_config = pd.DataFrame({
        "Parameter": [
            "Algorithm",
            "Library",
            "User Column",
            "Item Column",
            "Rating Column",
            "Evaluation Metric"
        ],
        "Value": [
            "Alternating Least Squares",
            "Spark MLlib",
            "user_index",
            "book_index",
            "rating",
            "RMSE"
        ]
    })

    st.dataframe(model_config, use_container_width=True)

    st.code("Run: spark-submit src/03_train_recommender.py", language="bash")


elif page == "Recommendation Results":
    st.header("Recommendation Results")

    st.write("""
    This section displays personalized Top-N recommendations generated by the trained
    ALS model. Select a user index to view the books recommended for that user.
    """)

    rec_folder = os.path.join(TABLES_DIR, "final_recommendations_csv")
    rec_df = load_spark_csv_folder(rec_folder)

    if rec_df.empty:
        st.warning("Run src/04_generate_recommendations.py first.")
    else:
        user_ids = sorted(rec_df["user_index"].unique())
        selected_user = st.selectbox("Select User Index", user_ids)

        user_recs = rec_df[rec_df["user_index"] == selected_user]

        st.dataframe(
            user_recs[
                [
                    "title",
                    "predicted_rating",
                    "average_rating",
                    "ratings_count",
                    "publication_year"
                ]
            ],
            use_container_width=True
        )


elif page == "New User Recommendations":
    st.header("New User Recommendations")

    st.write("""
    This section allows a new user to receive personalized recommendations.
    The user selects a few books they have read and provides ratings. The system
    then builds a temporary user preference vector using trained ALS item factors
    and recommends similar books.
    """)

    book_index_map, item_factors = load_new_user_data()

    if book_index_map is None or item_factors is None:
        st.warning("Run src/03_train_recommender.py first to generate book index and item factor files.")
    else:
        title_options = sorted(book_index_map["title"].dropna().unique().tolist())

        st.write(
            "Select up to 5 books you've read and rate them. The system will infer your preferences "
            "from the trained ALS item factors."
        )

        selected_titles = st.multiselect(
            "Select books for your profile",
            title_options,
            help="Choose a few books you know and rate them.",
            max_selections=5
        )

        ratings = []

        for i, title in enumerate(selected_titles):
            rating = st.slider(
                f"Rating for: {title}",
                min_value=1.0,
                max_value=5.0,
                step=0.5,
                value=4.0,
                key=f"rating_{i}"
            )
            ratings.append(rating)

        if not selected_titles:
            st.info("Select a few books above to get personalized recommendations.")
        else:
            if st.button("Generate Top 5 Recommendations"):
                recommendations = compute_new_user_recommendations(
                    selected_titles,
                    ratings,
                    book_index_map,
                    item_factors,
                    top_n=5
                )

                if recommendations.empty:
                    st.warning("No recommendations could be generated from the selected books.")
                else:
                    st.dataframe(
                        recommendations.rename(columns={"score": "predicted_score"}),
                        use_container_width=True
                    )


elif page == "Model Evaluation":
    st.header("Model Evaluation")

    st.write("""
    Model evaluation measures how well the ALS model predicts user ratings.
    RMSE is used as the main evaluation metric.
    """)

    metrics_file = os.path.join(TABLES_DIR, "model_metrics.txt")

    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            st.code(f.read())
    else:
        st.warning("Run src/03_train_recommender.py first.")

    st.subheader("RMSE Explanation")
    st.write("""
    RMSE measures the difference between actual user ratings and predicted ratings.
    A lower RMSE value indicates better prediction accuracy.
    """)

    st.subheader("Evaluation Summary")
    st.markdown("""
    - The ALS model is evaluated using test rating data.
    - Predicted ratings are compared with actual ratings.
    - RMSE is calculated to measure prediction error.
    - The trained model is then used to generate Top-N recommendations.
    """)