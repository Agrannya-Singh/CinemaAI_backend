# train.py
import sqlite3
import pandas as pd
import logging
import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
DB_PATH = "movies.db"
ASSETS_PATH = "assets"
os.makedirs(ASSETS_PATH, exist_ok=True)

# --- Main Training Logic ---
def train_and_save_models():
    """
    Loads data, trains all models, and saves them as artifacts to disk.
    This is the "heavy lifting" part that should be done offline.
    """
    logging.info("Connecting to database...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            movies_df = pd.read_sql_query("SELECT * FROM movies", conn, index_col='id')
            ratings_df = pd.read_sql_query("SELECT * FROM ratings", conn)
    except Exception as e:
        logging.error(f"Failed to load data from database: {e}")
        return

    if movies_df.empty:
        logging.error("No movies found in the database. Cannot train models.")
        return

    # --- 1. Content-Based Model ---
    logging.info("Building and saving content-based model...")
    corpus = movies_df["combined_features"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    content_matrix = cosine_similarity(tfidf_matrix)

    joblib.dump(vectorizer, os.path.join(ASSETS_PATH, "tfidf_vectorizer.joblib"))
    save_npz(os.path.join(ASSETS_PATH, "tfidf_matrix.npz"), tfidf_matrix)
    joblib.dump(content_matrix, os.path.join(ASSETS_PATH, "content_matrix.joblib"))
    logging.info("Content-based model saved.")

    # --- 2. Collaborative Filtering Model ---
    if not ratings_df.empty and len(ratings_df['movie_id'].unique()) > 1:
        logging.info("Building and saving collaborative model...")
        user_movie_matrix = ratings_df.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
        user_movie_matrix = user_movie_matrix.reindex(movies_df.index, fill_value=0)
        
        movie_features_sparse = csr_matrix(user_movie_matrix.values)
        collab_model = NearestNeighbors(metric='cosine', algorithm='brute')
        collab_model.fit(movie_features_sparse)
        
        user_movie_matrix.to_pickle(os.path.join(ASSETS_PATH, "user_movie_matrix.pkl"))
        joblib.dump(collab_model, os.path.join(ASSETS_PATH, "collab_model.joblib"))
        logging.info("Collaborative model saved.")
    else:
        logging.warning("Not enough data to build collaborative model. Skipping.")

    # --- 3. Save the main movies dataframe ---
    movies_df.to_pickle(os.path.join(ASSETS_PATH, "movies_df.pkl"))
    logging.info("Movies DataFrame saved.")

    logging.info("All models and data artifacts have been successfully built and saved!")

if __name__ == "__main__":
    train_and_save_models()
