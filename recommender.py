# recommender.py

import logging
import pandas as pd
import numpy as np
import joblib
import os
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler
from fastapi import HTTPException

# Import settings from your new config file
import config

# --- Load Pre-trained Models and Data (Happens only once on startup) ---
try:
    logging.info("Loading pre-trained model artifacts...")
    movies_df = pd.read_pickle(os.path.join(config.ASSETS_PATH, "movies_df.pkl"))
    content_matrix = joblib.load(os.path.join(config.ASSETS_PATH, "content_matrix.joblib"))
    
    collab_model_path = os.path.join(config.ASSETS_PATH, "collab_model.joblib")
    user_movie_matrix_path = os.path.join(config.ASSETS_PATH, "user_movie_matrix.pkl")
    
    collab_model = joblib.load(collab_model_path) if os.path.exists(collab_model_path) else None
    user_movie_matrix = pd.read_pickle(user_movie_matrix_path) if os.path.exists(user_movie_matrix_path) else None

    logging.info("Artifacts loaded successfully.")
except FileNotFoundError:
    logging.error("Model artifacts not found! Please run train.py first and commit the /assets folder.")
  
    # Set to empty structures to prevent crash, but endpoints will fail gracefully
    movies_df = pd.DataFrame()
    content_matrix = None
    collab_model = None
    user_movie_matrix = None

# --- Recommendation Logic ---
def get_recommendations(selected_movie_ids: list[str], num_recs: int = 10) -> list[dict]:
    """Generates hybrid recommendations using pre-loaded models."""
    if content_matrix is None or movies_df.empty:
        raise HTTPException(status_code=503, detail="Recommendation models are not ready.")
    if not all(mid in movies_df.index for mid in selected_movie_ids):
        raise HTTPException(status_code=404, detail="One or more selected movies not found.")

    # --- Content-Based Scores ---
    indices = [movies_df.index.get_loc(mid) for mid in selected_movie_ids]
    content_scores = np.mean(content_matrix[indices, :], axis=0)

    # --- Collaborative Scores ---
    collab_scores = np.zeros(len(movies_df))
    if collab_model and user_movie_matrix is not None:
        # (Your collaborative logic would go here, this is a placeholder)
        # In a real scenario, you'd calculate the collaborative scores
        pass

    # --- Combine and Rank ---
    scaler = MinMaxScaler()
    content_scores_norm = scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
    collab_scores_norm = scaler.fit_transform(collab_scores.reshape(-1, 1)).flatten()
    
    hybrid_scores = (config.CONTENT_WEIGHT * content_scores_norm) + \
                    (config.COLLAB_WEIGHT * collab_scores_norm)

    recs_df = pd.DataFrame({'score': hybrid_scores, 'id': movies_df.index})
    recs_df = recs_df[~recs_df['id'].isin(selected_movie_ids)]
    recs_df = recs_df.sort_values('score', ascending=False).head(num_recs)

    return movies_df.loc[recs_df['id'].tolist()].reset_index().to_dict(orient="records")

def get_all_movies() -> list[dict]:
    """Returns the list of all movies from the pre-loaded dataframe."""
    if movies_df.empty:
        return []
    return movies_df.reset_index().to_dict(orient="records")

# (find_or_add_movie logic would also go here)
