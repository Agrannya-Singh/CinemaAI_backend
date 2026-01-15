# train.py (Robust Version)
import sqlite3
import pandas as pd
import logging
import joblib
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
DB_PATH = "movies.db"
ASSETS_PATH = "assets"
os.makedirs(ASSETS_PATH, exist_ok=True)

# Choose a robust model (Small & Fast: 'all-MiniLM-L6-v2', Larger: 'all-mpnet-base-v2')
TRANSFORMER_MODEL_NAME = 'all-MiniLM-L6-v2' 

def train_and_save_models():
    """
    Trains a robust semantic recommendation engine using Sentence Transformers.
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
        logging.error("No movies found. Please populate the database first.")
        return

    # --- 1. Robust Content-Based Model (The "Transformer" Part) ---
    logging.info(f"Loading Transformer model: {TRANSFORMER_MODEL_NAME}...")
    model = SentenceTransformer(TRANSFORMER_MODEL_NAME)

    logging.info("Generating semantic embeddings (this may take a moment)...")
    # Combine features just like before, but now the model "reads" them
    corpus = movies_df["combined_features"].fillna("").astype(str).tolist()
    
    # Encode the corpus into dense vectors (embeddings)
    # This captures synonyms, context, and mood (e.g., "scary" ~= "horror")
    movie_embeddings = model.encode(corpus, show_progress_bar=True)

    logging.info("Calculating cosine similarity matrix...")
    # Calculate similarity between every movie pair based on deep meaning
    content_matrix = cosine_similarity(movie_embeddings)

    # Save artifacts (Compatible with your existing recommender.py)
    joblib.dump(content_matrix, os.path.join(ASSETS_PATH, "content_matrix.joblib"))
    # We don't need the vectorizer anymore, but saving the model can be useful if you do real-time encoding later
    # model.save(os.path.join(ASSETS_PATH, "transformer_model")) 
    logging.info("Content-based model saved successfully.")

    # --- 2. Collaborative Filtering Model (Unchanged) ---
    if not ratings_df.empty and len(ratings_df['movie_id'].unique()) > 1:
        logging.info("Building and saving collaborative model...")
        user_movie_matrix = ratings_df.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
        # Ensure alignment with movies_df
        user_movie_matrix = user_movie_matrix.reindex(movies_df.index, fill_value=0)
        
        movie_features_sparse = csr_matrix(user_movie_matrix.values)
        collab_model = NearestNeighbors(metric='cosine', algorithm='brute')
        collab_model.fit(movie_features_sparse)
        
        user_movie_matrix.to_pickle(os.path.join(ASSETS_PATH, "user_movie_matrix.pkl"))
        joblib.dump(collab_model, os.path.join(ASSETS_PATH, "collab_model.joblib"))
        logging.info("Collaborative model saved.")
    else:
        logging.warning("Not enough ratings data for collaborative model. Skipping.")

    # --- 3. Save Dataframe ---
    movies_df.to_pickle(os.path.join(ASSETS_PATH, "movies_df.pkl"))
    logging.info("Movies DataFrame saved.")

    logging.info("Training complete! Robust models are ready.")

if __name__ == "__main__":
    train_and_save_models()