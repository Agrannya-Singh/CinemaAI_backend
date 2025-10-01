# recommender.py

import logging
import numpy as np
import joblib
import os
import requests
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fastapi import HTTPException
{{ ... }}
    collab_model = joblib.load(collab_model_path) if os.path.exists(collab_model_path) else None
    user_movie_matrix = pd.read_pickle(user_movie_matrix_path) if os.path.exists(user_movie_matrix_path) else None

except Exception as e:
    logging.error(f"Error loading models: {e}")
# --- Global Variables ---
movies_df = pd.DataFrame()
content_matrix = None
collab_model = None
user_movie_matrix = None

# --- Retraining State ---
retrain_lock = threading.Lock()
last_retrain_time = datetime.min
last_movie_addition = datetime.min
pending_retrain = False
retrain_thread: Optional[threading.Thread] = None
new_movie_count = 0

# --- Core Logic Functions ---

def get_recommendations(selected_movie_ids: list[str], num_recs: int = 10) -> list[dict]:
{{ ... }}
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

def _schedule_retrain() -> None:
    """
    Schedule a model retraining after a delay if conditions are met.
    Uses a background thread to avoid blocking the main application.
    """
    global pending_retrain, retrain_thread, last_movie_addition, new_movie_count

    with retrain_lock:
        current_time = datetime.now()
        last_movie_addition = current_time
        new_movie_count += 1

        # Check if we should schedule a retrain
        time_since_last_retrain = (current_time - last_retrain_time).total_seconds()
        should_retrain = (
            (new_movie_count >= config.MIN_MOVIES_FOR_RETRAIN and 
             time_since_last_retrain > config.MODEL_UPDATE_DELAY) or
            time_since_last_retrain > config.MAX_RETRAIN_INTERVAL
        )

        if should_retrain and not pending_retrain:
            pending_retrain = True
            logging.info(f"Scheduling model retraining in {config.MODEL_UPDATE_DELAY} seconds...")

            def delayed_retrain():
                global pending_retrain, last_retrain_time, new_movie_count

                try:
                    while True:
                        time.sleep(config.MODEL_UPDATE_DELAY)
                        with retrain_lock:
                            if not pending_retrain:
                                return
                            quiet_for = (datetime.now() - last_movie_addition).total_seconds()
                            if quiet_for >= config.MODEL_UPDATE_DELAY:
                                logging.info(
                                    f"Starting scheduled model retraining with {new_movie_count} new movies"
                                )
                                _retrain_models()
                                last_retrain_time = datetime.now()
                                new_movie_count = 0
                                pending_retrain = False
                                return
                except Exception as e:
                    logging.error(f"Error in retrain scheduler: {e}")
                    with retrain_lock:
                        pending_retrain = False

            # Start the retrain thread
            retrain_thread = threading.Thread(target=delayed_retrain, daemon=True)
            retrain_thread.start()


def _retrain_models() -> Dict[str, Any]:
    """
    Retrain the recommendation models using the latest data from the database.
    Saves models both locally and to Supabase Storage.
    
    Returns:
        Dict containing retraining status and metadata
    """
    global movies_df, content_matrix, collab_model, user_movie_matrix, last_retrain_time
    
    try:
        logging.info("Starting model retraining...")
        
        # 1. Fetch all movies from Supabase
        supabase = get_supabase_client()
        response = supabase.table(config.MODELS_TABLE).select("*").execute()
        
        if not response.data:
            logging.warning("No movies found in the database for retraining")
            return False
            
        movies = pd.DataFrame(response.data)
        
        # 2. Preprocess and create features
        movies['combined_features'] = movies.apply(
            lambda x: f"{x.get('director', '')} {x.get('genres', '')} {x.get('cast', '')} {x.get('overview', '')}",
            axis=1
        )
        
        # 3. Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['combined_features'].fillna(''))
        
        # 4. Compute cosine similarity
        new_content_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # 5. Save updated models and data
        os.makedirs(config.ASSETS_PATH, exist_ok=True)
        movies.set_index('id', inplace=True)
        
        # Define paths
        movies_path = os.path.join(config.ASSETS_PATH, "movies_df.pkl")
        matrix_path = os.path.join(config.ASSETS_PATH, "content_matrix.joblib")
        collab_path = os.path.join(config.ASSETS_PATH, "collab_model.joblib")
        user_matrix_path = os.path.join(config.ASSETS_PATH, "user_movie_matrix.pkl")
        
        # Save locally
        movies.to_pickle(movies_path)
        joblib.dump(new_content_matrix, matrix_path)
        
        # Save collaborative filtering models if they exist
        if collab_model is not None:
            joblib.dump(collab_model, collab_path)
        if user_movie_matrix is not None:
            user_movie_matrix.to_pickle(user_matrix_path)
        
        # Upload to Supabase Storage
        from storage import storage
        storage.upload_file(movies_path, "movies_df.pkl")
        storage.upload_file(matrix_path, "content_matrix.joblib")
        
        if os.path.exists(collab_path):
            storage.upload_file(collab_path, "collab_model.joblib")
        if os.path.exists(user_matrix_path):
            storage.upload_file(user_matrix_path, "user_movie_matrix.pkl")
        
        # Update in-memory variables
        movies_df = movies
        content_matrix = new_content_matrix
        
        logging.info("Model retraining completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error during model retraining: {e}")
        return False

def _add_movie_to_db(movie_data: dict) -> Dict[str, Any]:
    """
    Adds a new movie to the Supabase database and schedules model retraining.
    
    Args:
        movie_data: Dictionary containing movie details
        
    Returns:
        Dict containing operation status and metadata
    """
    try:
        supabase = get_supabase_client()
        
        # Check if movie already exists
        existing_movie = supabase.table(config.MOVIES_TABLE)\
            .select("*")\
            .eq("id", movie_data["id"])\
            .execute()
            
        is_new_movie = len(existing_movie.data) == 0
        
        if not is_new_movie:
            return {
                "status": "exists",
                "message": "Movie already exists in database",
                "movie_id": movie_data["id"]
            }
        
        # Prepare data for Supabase insert
        insert_data = {
            "id": movie_data["id"],
            "title": movie_data["title"],
            "overview": movie_data["overview"],
            "genres": movie_data["genres"],
            "director": movie_data["director"],
            "cast": movie_data["cast"],
            "poster_path": movie_data["poster_path"],
            "vote_average": movie_data["vote_average"],
            "release_date": movie_data["release_date"],
            "combined_features": movie_data.get("combined_features", ""),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Insert the new movie
        response = supabase.table(config.MOVIES_TABLE).insert(insert_data).execute()
        
        if not response.data:
            raise Exception("Failed to add movie to database")
        
        # Schedule model retraining
        _schedule_retrain()
        
        logging.info(f"Successfully added '{movie_data['title']}' to the database")
        return {
            "status": "success",
            "message": "Movie added successfully",
            "movie_id": movie_data["id"],
            "retrain_scheduled": True,
            "new_movie_count": new_movie_count + 1
        }
        
    except Exception as e:
        error_msg = f"Error adding movie: {str(e)}"
        logging.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "movie_id": movie_data.get("id", "unknown")
        }
