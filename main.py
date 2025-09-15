# main.py

#streamlined and optimzed for quick start up

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import the logic functions from your new recommender.py file
import recommender

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

@app.get("/movies", response_model=List[Movie], summary="Get All Movies")
def get_movies():
    """
    Returns the list of all movies available for recommendations.
    This is fast because it reads from the pre-loaded DataFrame in memory.
    """
    return recommender.get_all_movies()

@app.post("/recommend", response_model=List[Movie], summary="Get Movie Recommendations")
def recommend_movies(request: MovieRequest):
    """
    Generates movie recommendations based on a list of selected movie IDs.
    This is a CPU-bound task, so we use a standard 'def' endpoint.
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
def search_movie(title: str):
    """
    Searches for a movie in the local cache. If not found, fetches from OMDb,
    adds it to the database, and returns the movie data.
    Note: The live models are NOT updated with this new movie. may cause hallucinations 
    """
    try:
        return recommender.find_or_add_movie(title)
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error during search for '{title}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
