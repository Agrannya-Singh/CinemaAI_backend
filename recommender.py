# recommender.py

import logging
import pandas as pd
import numpy as np
import joblib
import os
import requests
import sqlite3
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler
from fastapi import HTTPException

# Import settings f config file
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
    logging.error("Model artifacts not found! Run train.py and commit the /assets folder.")
    movies_df = pd.DataFrame()
    content_matrix, collab_model, user_movie_matrix = None, None, None


# --- Core Logic Functions ---

def get_recommendations(selected_movie_ids: list[str], num_recs: int = 10) -> list[dict]:
    """Generates hybrid recommendations using pre-loaded models."""
    if content_matrix is None or movies_df.empty:
        raise HTTPException(status_code=503, detail="Recommendation models are not ready.")
    if not all(mid in movies_df.index for mid in selected_movie_ids):
        raise HTTPException(status_code=404, detail="One or more selected movies not found.")

    indices = [movies_df.index.get_loc(mid) for mid in selected_movie_ids]
    content_scores = np.mean(content_matrix[indices, :], axis=0)
    collab_scores = np.zeros(len(movies_df)) # Placeholder for your collab logic

    scaler = MinMaxScaler()
    content_scores_norm = scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
    collab_scores_norm = scaler.fit_transform(collab_scores.reshape(-1, 1)).flatten()
    
    hybrid_scores = (config.CONTENT_WEIGHT * content_scores_norm) + (config.COLLAB_WEIGHT * collab_scores_norm)

    recs_df = pd.DataFrame({'score': hybrid_scores, 'id': movies_df.index})
    recs_df = recs_df[~recs_df['id'].isin(selected_movie_ids)]
    recs_df = recs_df.sort_values('score', ascending=False).head(num_recs)

    return movies_df.loc[recs_df['id'].tolist()].reset_index().to_dict(orient="records")

def get_all_movies() -> list[dict]:
    """Returns the list of all movies from the pre-loaded dataframe."""
    if movies_df.empty:
        return []
    return movies_df.reset_index().to_dict(orient="records")

def find_or_add_movie(title: str) -> dict:
    """
    Finds a movie in the local DB/cache. If not found, fetches from OMDb API,
    adds it to the database, but does NOT update the live models.
    """
    if not movies_df.empty:
        # Case-insensitive search on the pre-loaded dataframe
        local_match = movies_df[movies_df['title'].str.lower() == title.lower()]
        if not local_match.empty:
            logging.info(f"Found '{title}' in local cache.")
            return local_match.reset_index().iloc[0].to_dict()

    logging.info(f"'{title}' not in local cache. Fetching from OMDb API.")
    raw_data = _fetch_movie_from_api(title)
    
    if raw_data and (movie_data := _process_api_data(raw_data)):
        _add_movie_to_db(movie_data)
        # Note: The in-memory dataframe and models are NOT updated.
        # This new movie will only be included after the next run of train.py
        return movie_data

    raise HTTPException(status_code=404, detail=f"Movie '{title}' not found in OMDb.")


# --- Helper Functions for Search ---

def _fetch_movie_from_api(title: str) -> dict | None:
    """Fetches raw movie data from the OMDb API."""
    params = {"apikey": config.OMDB_API_KEY, "t": title, "plot": "full"}
    try:
        resp = requests.get(config.OMDB_API_URL, params=params, timeout=config.API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        return data if data.get("Response") == "True" else None
    except requests.RequestException as e:
        logging.error(f"Error fetching '{title}' from OMDb: {e}")
        return None

def _process_api_data(data: dict) -> dict:
    """Processes raw API data into a structured movie dictionary."""
    director = data.get("Director", "")
    genres = data.get("Genre", "")
    actors = data.get("Actors", "")
    plot = data.get("Plot", "")
    
    director_clean = "".join(director.split()) if director else ""
    genres_clean = "".join(genres.split(",")) if genres else ""
    actors_clean = "".join(actors.split(",")[:3]) if actors else ""
    
    combined_features = f"{director_clean} {genres_clean} {actors_clean} {plot}"
    
    return {
        "id": data.get("imdbID"), "title": data.get("Title"), "overview": plot,
        "genres": genres, "director": director, "cast": actors,
        "poster_path": data.get("Poster"),
        "vote_average": float(data.get("imdbRating", 0.0) if data.get("imdbRating") != "N/A" else 0.0),
        "release_date": data.get("Year"), "combined_features": combined_features
    }

def _add_movie_to_db(movie_data: dict):
    """Adds a new movie to the SQLite database."""
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            #  INSERT OR IGNORE to prevent adding duplicates if another process adds it first
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO movies (id, title, overview, genres, director, cast, poster_path, vote_average, release_date, combined_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                movie_data["id"], movie_data["title"], movie_data["overview"], movie_data["genres"],
                movie_data["director"], movie_data["cast"], movie_data["poster_path"],
                movie_data["vote_average"], movie_data["release_date"], movie_data["combined_features"]
            ))
            conn.commit()
            logging.info(f"Successfully added '{movie_data['title']}' to the database.")
    except sqlite3.Error as e:
        logging.error(f"Database error while adding movie: {e}")
        # Not raising an exception here to allow the API to return the movie data
        # even if the database write fails.
