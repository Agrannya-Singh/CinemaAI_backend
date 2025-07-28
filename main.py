import os
import sqlite3
import pickle
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
if not OMDB_API_KEY:
    raise ValueError("OMDB_API_KEY not found in environment variables.")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLite database setup
DB_PATH = "movies.db"
SIM_MATRIX_PATH = "similarity_matrix.pkl"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
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
    ''')
    conn.commit()
    conn.close()

class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.similarity_matrix = None
        self.vectorizer = CountVectorizer(stop_words='english')
        self.API_KEY = OMDB_API_KEY
        self.BASE_URL = "http://www.omdbapi.com/"
        init_db()
        self.load_data()

    def fetch_movie_by_title(self, title):
        """Fetch a single movie by title from OMDb API."""
        params = {"apikey": self.API_KEY, "t": title, "plot": "full"}
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("Response") == "True":
                return {
                    "id": data.get("imdbID", f"unknown_{title}"),
                    "title": data.get("Title", ""),
                    "overview": data.get("Plot", ""),
                    "genres": data.get("Genre", ""),
                    "cast": data.get("Actors", ""),
                    "poster_path": data.get("Poster", ""),
                    "vote_average": float(data.get("imdbRating", 0)) if data.get("imdbRating", "0") not in ["N/A", None, ""] else 0.0,
                    "release_date": data.get("Year", ""),
                    "combined_features": f"{data.get('Genre', '')} {data.get('Actors', '')} {data.get('Plot', '')}"
                }
            return None
        except Exception as e:
            print(f"Error fetching movie '{title}': {e}")
            return None

    def load_data(self):
        """Load movie data from SQLite or fetch from OMDb API."""
        conn = sqlite3.connect(DB_PATH)
        self.movies_df = pd.read_sql_query("SELECT * FROM movies", conn)
        conn.close()

        if self.movies_df.empty:
            print("No data in database. Fetching from OMDb API.")
            titles = [
                "Inception", "The Dark Knight", "Interstellar", "The Matrix", "Pulp Fiction",
                "Forrest Gump", "The Shawshank Redemption", "Gladiator", "Titanic", "Avatar"
            ]  # Limited default list for brevity
            movies_data = [self.fetch_movie_by_title(title) for title in titles if self.fetch_movie_by_title(title)]
            if movies_data:
                self.movies_df = pd.DataFrame(movies_data)
                self.save_data()
            else:
                print("Using fallback dataset.")
                self.movies_df = pd.DataFrame([
                    {"id": "tt0372784", "title": "Batman Begins", "overview": "Bruce Wayne becomes Batman.", "genres": "Action, Adventure, Crime", "cast": "Christian Bale, Michael Caine", "poster_path": "", "vote_average": 8.2, "release_date": "2005", "combined_features": "Action Adventure Crime Christian Bale Michael Caine Bruce Wayne becomes Batman."}
                ])

        self.load_similarity_matrix()

    def save_data(self):
        """Save movie data to SQLite."""
        conn = sqlite3.connect(DB_PATH)
        self.movies_df.to_sql("movies", conn, if_exists="replace", index=False)
        conn.close()

    def build_similarity_matrix(self):
        """Build and save similarity matrix."""
        if self.movies_df is not None and not self.movies_df.empty:
            corpus = self.movies_df["combined_features"].fillna("").astype(str).tolist()
            if not any(corpus):
                print("Corpus is empty.")
                self.similarity_matrix = None
                return
            max_features = min(5000, len(set(" ".join(corpus).split())) or 1)
            self.vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
            try:
                vectorized_features = self.vectorizer.fit_transform(corpus)
                self.similarity_matrix = cosine_similarity(vectorized_features)
                with open(SIM_MATRIX_PATH, "wb") as f:
                    pickle.dump(self.similarity_matrix, f)
                print(f"Similarity matrix built and saved: {self.similarity_matrix.shape}")
            except Exception as e:
                print(f"Error building similarity matrix: {e}")
                self.similarity_matrix = None
        else:
            print("No movie data to build similarity matrix.")
            self.similarity_matrix = None

    def load_similarity_matrix(self):
        """Load similarity matrix from disk or rebuild."""
        try:
            with open(SIM_MATRIX_PATH, "rb") as f:
                self.similarity_matrix = pickle.load(f)
                if self.similarity_matrix.shape[0] != len(self.movies_df):
                    print("Similarity matrix size mismatch. Rebuilding.")
                    self.build_similarity_matrix()
                else:
                    print("Similarity matrix loaded from disk.")
        except FileNotFoundError:
            print("No similarity matrix found. Building new one.")
            self.build_similarity_matrix()

    def get_recommendations(self, selected_movie_ids: List[str], num_recommendations: int = 5) -> List[dict]:
        """Get movie recommendations based on selected movie IDs."""
        if self.similarity_matrix is None or self.movies_df is None or self.movies_df.empty:
            return []
        valid_indices = self.movies_df[self.movies_df["id"].isin(selected_movie_ids)].index.tolist()
        if not valid_indices:
            return []
        avg_similarity_scores = np.mean(self.similarity_matrix[valid_indices, :], axis=0)
        sorted_indices = np.argsort(avg_similarity_scores)[::-1]
        recommendations = []
        seen = set(selected_movie_ids)
        for idx in sorted_indices:
            if idx >= len(self.movies_df):
                continue
            movie = self.movies_df.iloc[idx]
            if movie["id"] not in seen:
                recommendations.append(movie.to_dict())
                seen.add(movie["id"])
                if len(recommendations) >= num_recommendations:
                    break
        return recommendations

    def add_movie(self, title: str):
        """Add a new movie to the dataset and update similarity matrix."""
        movie_data = self.fetch_movie_by_title(title)
        if movie_data and movie_data["id"] not in self.movies_df["id"].values:
            self.movies_df = pd.concat([self.movies_df, pd.DataFrame([movie_data])], ignore_index=True)
            self.save_data()
            self.build_similarity_matrix()
            return movie_data
        return None

# Initialize recommender
rec_sys = MovieRecommendationSystem()

# Pydantic models
class MovieRequest(BaseModel):
    movie_ids: List[str]
    num_recommendations: int = 5

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
    """Return all available movies."""
    if rec_sys.movies_df is None or rec_sys.movies_df.empty:
        raise HTTPException(status_code=500, detail="Movie data not available")
    return rec_sys.movies_df.to_dict(orient="records")

@app.post("/recommend", response_model=List[Movie])
async def get_recommendations(request: MovieRequest):
    """Get movie recommendations."""
    recommendations = rec_sys.get_recommendations(request.movie_ids, request.num_recommendations)
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found")
    return recommendations

@app.get("/search/{title}", response_model=List[Movie])
async def search_movie(title: str, background_tasks: BackgroundTasks):
    """Search for a movie by title."""
    movie_data = rec_sys.add_movie(title)
    if not movie_data:
        raise HTTPException(status_code=404, detail="Movie not found")
    return [movie_data]
