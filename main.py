import os
import sqlite3
import pickle
import requests
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
if not OMDB_API_KEY:
    raise ValueError("OMDB_API_KEY not found in environment variables.")

# Constants
DB_PATH = "movies.db"
SIM_MATRIX_PATH = "similarity_matrix.pkl"
OMDB_API_URL = "http://www.omdbapi.com/"

# Initialize FastAPI app
app = FastAPI()

# CORS setup for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust accordingly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_db():
    """Initialize SQLite database and create tables if they don't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                id TEXT PRIMARY KEY,
                title TEXT,
                overview TEXT,
                genres TEXT,
                cast TEXT,
                poster_path TEXT,
                vote_average REAL,
                release_date TEXT,
                combined_features TEXT
            )
        """)
        conn.commit()

class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.similarity_matrix: Optional[np.ndarray] = None
        self.vectorizer = CountVectorizer(stop_words="english")
        init_db()
        self.load_movies()
        self.load_similarity_matrix()

    def fetch_movie_by_title(self, title: str) -> Optional[dict]:
        """Fetch movie info from OMDb API by title."""
        params = {"apikey": OMDB_API_KEY, "t": title, "plot": "full"}
        try:
            resp = requests.get(OMDB_API_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("Response") == "True":
                imdb_rating = data.get("imdbRating", "0")
                vote_average = float(imdb_rating) if imdb_rating not in ["N/A", None, ""] else 0.0
                combined_features = f"{data.get('Genre', '')} {data.get('Actors', '')} {data.get('Plot', '')}"
                return {
                    "id": data.get("imdbID", f"unknown_{title}"),
                    "title": data.get("Title", ""),
                    "overview": data.get("Plot", ""),
                    "genres": data.get("Genre", ""),
                    "cast": data.get("Actors", ""),
                    "poster_path": data.get("Poster", ""),
                    "vote_average": vote_average,
                    "release_date": data.get("Year", ""),
                    "combined_features": combined_features
                }
            return None
        except Exception as e:
            print(f"Error fetching movie '{title}': {e}")
            return None

    def load_movies(self):
        """Load movies from DB; if empty, fetch default titles from OMDb API."""
        with sqlite3.connect(DB_PATH) as conn:
            self.movies_df = pd.read_sql_query("SELECT * FROM movies", conn)

        if self.movies_df.empty:
            print("Movie DB empty. Fetching default movies from OMDb...")
            default_titles = [
                "Inception", "The Dark Knight", "Interstellar", "The Matrix",
                "Pulp Fiction", "Forrest Gump", "The Shawshank Redemption",
                "Gladiator", "Titanic", "Avatar"
            ]
            movies = []
            for title in default_titles:
                movie = self.fetch_movie_by_title(title)
                if movie:
                    movies.append(movie)
            if movies:
                self.movies_df = pd.DataFrame(movies)
                self.save_movies()
            else:
                print("Fallback: adding minimal dataset.")
                fallback = [{
                    "id": "tt0372784",
                    "title": "Batman Begins",
                    "overview": "Bruce Wayne becomes Batman.",
                    "genres": "Action, Adventure, Crime",
                    "cast": "Christian Bale, Michael Caine",
                    "poster_path": "",
                    "vote_average": 8.2,
                    "release_date": "2005",
                    "combined_features": "Action Adventure Crime Christian Bale Michael Caine Bruce Wayne becomes Batman."
                }]
                self.movies_df = pd.DataFrame(fallback)
                self.save_movies()

    def save_movies(self):
        """Save current movie dataframe to SQLite."""
        with sqlite3.connect(DB_PATH) as conn:
            self.movies_df.to_sql("movies", conn, if_exists="replace", index=False)

    def build_similarity_matrix(self):
        """Compute and persist the similarity matrix based on combined features."""
        if self.movies_df.empty:
            print("No movies available to build similarity matrix.")
            self.similarity_matrix = None
            return

        corpus = self.movies_df["combined_features"].fillna("").astype(str).tolist()
        if not any(corpus):
            print("Empty combined features; cannot build similarity.")
            self.similarity_matrix = None
            return

        max_features = min(5000, len(set(" ".join(corpus).split())))
        self.vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
        try:
            features_matrix = self.vectorizer.fit_transform(corpus)
            self.similarity_matrix = cosine_similarity(features_matrix)
            with open(SIM_MATRIX_PATH, "wb") as f:
                pickle.dump(self.similarity_matrix, f)
            print(f"Similarity matrix built and saved: shape={self.similarity_matrix.shape}")
        except Exception as e:
            print(f"Error building similarity matrix: {e}")
            self.similarity_matrix = None

    def load_similarity_matrix(self):
        """Load the similarity matrix from disk or rebuild if missing/invalid."""
        try:
            with open(SIM_MATRIX_PATH, "rb") as f:
                sim_matrix = pickle.load(f)
            if sim_matrix.shape[0] != len(self.movies_df):
                print("Similarity matrix size mismatch; rebuilding...")
                self.build_similarity_matrix()
            else:
                self.similarity_matrix = sim_matrix
                print("Similarity matrix loaded from file.")
        except FileNotFoundError:
            print("No similarity matrix file; building new one...")
            self.build_similarity_matrix()

    def get_recommendations(self, selected_movie_ids: List[str], num_recommendations: int = 5) -> List[dict]:
        """Return recommended movies given selected IMDb IDs."""
        if self.similarity_matrix is None or self.movies_df.empty:
            return []

        indices = self.movies_df[self.movies_df["id"].isin(selected_movie_ids)].index.tolist()
        if not indices:
            return []

        avg_scores = np.mean(self.similarity_matrix[indices, :], axis=0)
        sorted_indices = np.argsort(avg_scores)[::-1]

        recommendations = []
        exclusion_set = set(selected_movie_ids)

        for idx in sorted_indices:
            if idx >= len(self.movies_df):
                continue
            movie = self.movies_df.iloc[idx]
            if movie["id"] not in exclusion_set:
                recommendations.append(movie.to_dict())
                exclusion_set.add(movie["id"])
                if len(recommendations) >= num_recommendations:
                    break

        return recommendations

    def add_movie(self, title: str) -> Optional[dict]:
        """Add a new movie by title, update DB and similarity matrix if new."""
        movie = self.fetch_movie_by_title(title)
        if movie and movie["id"] not in self.movies_df["id"].values:
            self.movies_df = pd.concat([self.movies_df, pd.DataFrame([movie])], ignore_index=True)
            self.save_movies()
            self.build_similarity_matrix()
            return movie
        return None

# Instantiate recommender system
rec_sys = MovieRecommendationSystem()

# Pydantic Schemas
class MovieRequest(BaseModel):
    movie_ids: List[str]
    num_recommendations: Optional[int] = 5

class Movie(BaseModel):
    id: str
    title: str
    overview: str
    genres: str
    cast: str
    poster_path: str
    vote_average: float
    release_date: str

@app.get("/movies", response_model=List[Movie])
async def get_movies():
    if rec_sys.movies_df.empty:
        raise HTTPException(status_code=500, detail="No movie data available")
    # Explicitly ensure all records are returned
    movies = rec_sys.movies_df.to_dict(orient="records")
    return movies

@app.post("/recommend", response_model=List[Movie])
async def recommend_movies(request: MovieRequest):
    recommendations = rec_sys.get_recommendations(request.movie_ids, request.num_recommendations or 5)
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found for given movies")
    return recommendations

@app.get("/search/{title}", response_model=List[Movie])
async def search_movie(title: str, background_tasks: BackgroundTasks):
    movie = rec_sys.add_movie(title)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return [movie]
