import os
import sqlite3
import requests
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix

# Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
if not OMDB_API_KEY:
    raise ValueError("OMDB_API_KEY not found in environment variables.")

# Constants
DB_PATH = "movies.db"
OMDB_API_URL = "http://www.omdbapi.com/"

# --- Recommendation Engine Class (with Search/Add functionality) ---

class MovieRecommender:
    def __init__(self, db_path="movies.db"):
        self.db_path = db_path
        self.movies_df = pd.DataFrame()
        self.ratings_df = pd.DataFrame()
        self.content_matrix = None
        self.collab_model = None
        self.user_movie_matrix = None
        self.scaler = MinMaxScaler()
        self.load_data()
        self.build_models()

    def load_data(self):
        """Load movies and ratings from the database."""
        with sqlite3.connect(self.db_path) as conn:
            self.movies_df = pd.read_sql_query("SELECT * FROM movies", conn, index_col='id')
            self.ratings_df = pd.read_sql_query("SELECT * FROM ratings", conn)
            # Ensure database tables exist if they don't
            conn.execute("""
                CREATE TABLE IF NOT EXISTS movies (
                    id TEXT PRIMARY KEY, title TEXT, overview TEXT, genres TEXT, director TEXT,
                    cast TEXT, poster_path TEXT, vote_average REAL, release_date TEXT, combined_features TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ratings (user_id INTEGER, movie_id TEXT, rating REAL)
            """)

    def build_models(self):
        """Build (or rebuild) both recommendation models."""
        self._build_content_model()
        self._build_collaborative_model()

    def _build_content_model(self):
        """Build the content-based similarity matrix using TF-IDF."""
        print("Building content-based model...")
        if self.movies_df.empty:
            print("No movies in DataFrame to build content model.")
            return
        corpus = self.movies_df["combined_features"].fillna("").astype(str)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        self.content_matrix = cosine_similarity(tfidf_matrix)
        print("Content-based model built.")

    def _build_collaborative_model(self):
        """Build the item-based collaborative model using NearestNeighbors."""
        if self.ratings_df.empty or len(self.ratings_df['movie_id'].unique()) < 2:
            print("Not enough data to build collaborative model.")
            return
        print("Building collaborative model...")
        user_movie_matrix = self.ratings_df.pivot(
            index='movie_id', columns='user_id', values='rating'
        ).fillna(0)
        movie_features_sparse = csr_matrix(user_movie_matrix.values)
        self.collab_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.collab_model.fit(movie_features_sparse)
        self.user_movie_matrix = user_movie_matrix
        print("Collaborative model built.")
        
    def _fetch_movie_from_api(self, title: str) -> Optional[dict]:
        """Helper to fetch movie data from OMDb API."""
        params = {"apikey": OMDB_API_KEY, "t": title, "plot": "full"}
        try:
            resp = requests.get(OMDB_API_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("Response") == "True":
                director_clean = ''.join(data.get('Director', '').split())
                genres_clean = ''.join(data.get('Genre', '').split(','))
                actors_clean = ''.join(data.get('Actors', '').split(',')[:3])
                combined_features = f"{director_clean} {genres_clean} {actors_clean} {data.get('Plot', '')}"
                return {
                    "id": data.get("imdbID"), "title": data.get("Title"), "overview": data.get("Plot"),
                    "genres": data.get("Genre"), "director": data.get("Director"), "cast": data.get("Actors"),
                    "poster_path": data.get("Poster"), "vote_average": float(data.get("imdbRating", 0.0) if data.get("imdbRating") != "N/A" else 0.0),
                    "release_date": data.get("Year"), "combined_features": combined_features
                }
            return None
        except Exception as e:
            print(f"Error fetching movie '{title}': {e}")
            return None

    def add_movie(self, title: str) -> Optional[dict]:
        """Search for a movie, add it to the DB, and rebuild models."""
        movie_data = self._fetch_movie_from_api(title)
        if movie_data:
            # Check if movie already exists
            if movie_data["id"] not in self.movies_df.index:
                print(f"Adding new movie: {movie_data['title']}")
                new_movie_df = pd.DataFrame([movie_data]).set_index('id')
                self.movies_df = pd.concat([self.movies_df, new_movie_df])
                # Save to database
                with sqlite3.connect(self.db_path) as conn:
                    new_movie_df.reset_index().to_sql("movies", conn, if_exists="append", index=False)
                # Rebuild models to include the new data
                self.build_models()
            return movie_data
        return None

    def get_recommendations(self, selected_movie_ids: list, num_recs: int = 10):
        """Generate hybrid recommendations."""
        if not all(mid in self.movies_df.index for mid in selected_movie_ids):
             raise HTTPException(status_code=404, detail="One or more selected movies not found in the database.")
        
        indices = [self.movies_df.index.get_loc(mid) for mid in selected_movie_ids]
        content_scores = np.mean(self.content_matrix[indices, :], axis=0)
        collab_scores = np.zeros(len(self.movies_df))
        if self.collab_model and self.user_movie_matrix is not None:
            for movie_id in selected_movie_ids:
                if movie_id in self.user_movie_matrix.index:
                    movie_idx = self.user_movie_matrix.index.get_loc(movie_id)
                    distances, indices = self.collab_model.kneighbors(
                        self.user_movie_matrix.iloc[movie_idx, :].values.reshape(1, -1), n_neighbors=20
                    )
                    sim_scores = 1 - distances.flatten()
                    for i, neighbor_idx in enumerate(indices.flatten()):
                        neighbor_movie_id = self.user_movie_matrix.index[neighbor_idx]
                        if neighbor_movie_id in self.movies_df.index:
                            main_df_idx = self.movies_df.index.get_loc(neighbor_movie_id)
                            collab_scores[main_df_idx] += sim_scores[i]

        content_scores_norm = self.scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
        collab_scores_norm = self.scaler.fit_transform(collab_scores.reshape(-1, 1)).flatten()
        hybrid_scores = (0.5 * content_scores_norm) + (0.5 * collab_scores_norm)
        recs_df = pd.DataFrame({'score': hybrid_scores, 'id': self.movies_df.index})
        recs_df = recs_df[~recs_df['id'].isin(selected_movie_ids)]
        recs_df = recs_df.sort_values('score', ascending=False).head(num_recs)
        results = self.movies_df.loc[recs_df['id'].tolist()].reset_index().to_dict(orient="records")
        return results

# --- FastAPI Application ---

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <--- THIS IS THE CHANGE
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = MovieRecommender(db_path=DB_PATH)

# --- Pydantic Schemas ---
class MovieRequest(BaseModel):
    movie_ids: List[str]
    num_recommendations: Optional[int] = 10

class Movie(BaseModel):
    id: str
    title: str
    overview: Optional[str] = ""
    genres: Optional[str] = ""
    director: Optional[str] = ""
    cast: Optional[str] = ""
    poster_path: Optional[str] = ""
    vote_average: Optional[float] = 0.0
    release_date: Optional[str] = ""

# --- API Endpoints ---
@app.get("/movies", response_model=List[Movie])
async def get_movies():
    if recommender.movies_df.empty:
        raise HTTPException(status_code=500, detail="No movie data available")
    return recommender.movies_df.reset_index().to_dict(orient="records")

@app.post("/recommend", response_model=List[Movie])
async def recommend_movies(request: MovieRequest):
    recommendations = recommender.get_recommendations(
        request.movie_ids, request.num_recommendations or 10
    )
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found.")
    return recommendations

@app.get("/search/{title}", response_model=Movie)
async def search_movie(title: str):
    """
    Search for a movie by title. If found via API, add it to our database
    and rebuild the recommendation models before returning it.
    """
    movie = recommender.add_movie(title)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found.")
    return movie
