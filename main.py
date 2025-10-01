# main.py

#streamlined and optimzed for quick start up

import logging
import threading
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Import the logic functions from your new recommender.py file
import recommender

 # Import authentication utilities
 from auth import (
     UserSignUp, UserLogin, Token,
     sign_up_user, login_user,
     get_current_user, require_auth
 )

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- FastAPI Application ---
app = FastAPI(title="CinemaAI Recommendation API", description="A hybrid movie recommendation engine.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # potential security risk. To Do:  restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas (API Data Models) ---
class MovieRequest(BaseModel):
    movie_ids: List[str]
    num_recommendations: int = 10

class Movie(BaseModel):
    id: str
    title: str
    overview: Optional[str] = None
    genres: Optional[str] = None
    director: Optional[str] = None
    cast: Optional[str] = None
    poster_path: Optional[str] = None
    vote_average: Optional[float] = None
    release_date: Optional[str] = None
# --- API Endpoints ---

@app.get("/", summary="Health Check")
def read_root():
    """Health check endpoint to confirm the API is running."""
    return {"status": "API is running"}

# --- Admin Endpoints ---

_retrain_lock = threading.Lock()
_retrain_in_progress = False

@app.post("/admin/retrain-models", response_model=Dict[str, Any], tags=["admin"], status_code=202)
async def retrain_models(current_user: dict = Depends(require_auth)):
    """
    Manually trigger retraining of the recommendation models.
    Requires authentication.
    """
    global _retrain_in_progress
    with _retrain_lock:
        if _retrain_in_progress:
            return {
                "status": "in_progress",
                "message": "Retraining already running."
            }
        _retrain_in_progress = True

    # Start retraining in background
    def _run():
        try:
            recommender._retrain_models()
        finally:
            global _retrain_in_progress
            _retrain_in_progress = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "status": "success",
        "message": "Model retraining started in background. This may take a few minutes.",
        "details": "Check server logs for progress and completion."
    }
# --- Run the API ---

@app.post("/auth/signup", response_model=Token, summary="Sign Up")
async def signup(user_data: UserSignUp):
    """
    Register a new user with email and password.
    """
    return await sign_up_user(user_data)

@app.post("/auth/login", response_model=Token, summary="Login")
async def login(user_data: UserLogin):
    """
    Login with email and password.
    Returns an access token upon successful authentication.
    """
    return await login_user(user_data)

@app.get("/auth/me", summary="Get Current User")
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Get the current authenticated user's information.
    Requires a valid JWT token in the Authorization header.
    """
    return current_user

@app.get("/movies", response_model=List[Movie], summary="Get All Movies")
def get_movies(current_user: dict = Depends(require_auth)):
    """
    Returns the list of all movies available for recommendations.
    This is fast because it reads from the pre-loaded DataFrame in memory.
    Requires authentication.
    """
    return recommender.get_all_movies()

@app.post("/recommend", response_model=List[Movie], summary="Get Movie Recommendations")
def recommend_movies(request: MovieRequest, current_user: dict = Depends(require_auth)):
    """
    Generates movie recommendations based on a list of selected movie IDs.
    This is a CPU-bound task, so we use a standard 'def' endpoint.
    Requires authentication.
    """
    try:
        recommendations = recommender.get_recommendations(
            selected_movie_ids=request.movie_ids,
            num_recs=request.num_recommendations
        )
        return recommendations
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/search/{title}", response_model=Movie, summary="Search for a Movie")
def search_movie(title: str, current_user: dict = Depends(require_auth)):
    """
    Searches for a movie in the local cache. If not found, fetches from OMDb,
    adds it to the database, and returns the movie data.
    Note: The live models are NOT updated with this new movie. may cause hallucinations
    Requires authentication.
    """
    try:
        return recommender.find_or_add_movie(title)
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error during search for '{title}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
