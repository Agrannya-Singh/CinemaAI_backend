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

# New imports for the hybrid model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

# Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
if not OMDB_API_KEY:
    raise ValueError("OMDB_API_KEY not found in environment variables.")

# Constants
DB_PATH = "movies.db"
CONTENT_MATRIX_PATH = "content_similarity_matrix.pkl"
COLLAB_MODEL_PATH = "collaborative_model.pkl"
OMDB_API_URL = "http://www.omdbapi.com/"

# Initialize FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
                director TEXT,
                cast TEXT,
                poster_path TEXT,
                vote_average REAL,
                release_date TEXT,
                combined_features TEXT
            )
        """)
        # New tables for collaborative filtering
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                user_id INTEGER,
                movie_id TEXT,
                rating REAL,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (movie_id) REFERENCES movies (id),
                PRIMARY KEY (user_id, movie_id)
            )
        """)
        conn.commit()

class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.content_similarity_matrix: Optional[np.ndarray] = None
        self.collab_model = None
        self.scaler = MinMaxScaler()

        init_db()
        self.load_movies()
        # This function will create dummy data if the DB is fresh
        self._populate_dummy_ratings_if_needed() 
        self.load_or_build_models()

    def fetch_movie_by_title(self, title: str) -> Optional[dict]:
        """Fetch movie info from OMDb API by title."""
        params = {"apikey": OMDB_API_KEY, "t": title, "plot": "full"}
        try:
            resp = requests.get(OMDB_API_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("Response") == "True":
                # --- ENHANCEMENT: Feature Weighting ---
                director_clean = ''.join(data.get('Director', '').split())
                genres_clean = ''.join(data.get('Genre', '').split(','))
                actors_clean = ''.join(data.get('Actors', '').split(',')[:3]) # Top 3 actors

                # Give director 3x weight, genres 2x weight
                combined_features = (
                    f"{director_clean} {director_clean} {director_clean} "
                    f"{genres_clean} {genres_clean} "
                    f"{actors_clean} {data.get('Plot', '')}"
                )
                
                return {
                    "id": data.get("imdbID"),
                    "title": data.get("Title"),
                    "overview": data.get("Plot"),
                    "genres": data.get("Genre"),
                    "director": data.get("Director"),
                    "cast": data.get("Actors"),
                    "poster_path": data.get("Poster"),
                    "vote_average": float(data.get("imdbRating", 0.0) if data.get("imdbRating") != "N/A" else 0.0),
                    "release_date": data.get("Year"),
                    "combined_features": combined_features
                }
            return None
        except Exception as e:
            print(f"Error fetching movie '{title}': {e}")
            return None

    def load_movies(self):
        """Load movies from DB; if empty, fetch default titles."""
        with sqlite3.connect(DB_PATH) as conn:
            self.movies_df = pd.read_sql_query("SELECT * FROM movies", conn, index_col='id')

        if self.movies_df.empty:
            print("Movie DB empty. Fetching default movies...")
            default_titles = [
                "Inception", "The Dark Knight", "Interstellar", "The Matrix",
                "Pulp Fiction", "Forrest Gump", "The Shawshank Redemption",
                "Gladiator", "The Godfather", "Fight Club", "Goodfellas", "Seven"
            ]
            movies = [self.fetch_movie_by_title(t) for t in default_titles if t]
            if movies:
                self.movies_df = pd.DataFrame(movies).set_index('id')
                self.save_movies()

    def save_movies(self):
        with sqlite3.connect(DB_PATH) as conn:
            self.movies_df.reset_index().to_sql("movies", conn, if_exists="replace", index=False)

    def _build_content_matrix(self):
        """Compute and persist the content-based similarity matrix using TF-IDF."""
        print("Building content-based similarity matrix...")
        corpus = self.movies_df["combined_features"].fillna("").astype(str)
        # --- ENHANCEMENT: Using TfidfVectorizer ---
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        features_matrix = vectorizer.fit_transform(corpus)
        self.content_similarity_matrix = cosine_similarity(features_matrix)
        with open(CONTENT_MATRIX_PATH, "wb") as f:
            pickle.dump(self.content_similarity_matrix, f)
        print("Content matrix built and saved.")

    def _build_collaborative_model(self):
        """Train and persist the item-based collaborative filtering model."""
        print("Building collaborative filtering model...")
        with sqlite3.connect(DB_PATH) as conn:
            ratings_df = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
        
        if len(ratings_df) < 10: # Need some data to train
            print("Not enough rating data to build collaborative model.")
            return

        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        # Item-based collaborative filtering
        sim_options = {'name': 'cosine', 'user_based': False}
        self.collab_model = KNNBasic(sim_options=sim_options)
        self.collab_model.fit(trainset)

        with open(COLLAB_MODEL_PATH, "wb") as f:
            pickle.dump(self.collab_model, f)
        print("Collaborative model built and saved.")

    def load_or_build_models(self):
        """Load all models from disk or rebuild them if necessary."""
        # Content Model
        try:
            with open(CONTENT_MATRIX_PATH, "rb") as f:
                self.content_similarity_matrix = pickle.load(f)
            if self.content_similarity_matrix.shape[0] != len(self.movies_df):
                raise ValueError("Matrix size mismatch.")
            print("Content matrix loaded from file.")
        except (FileNotFoundError, ValueError):
            print("Rebuilding content matrix...")
            self._build_content_matrix()

        # Collaborative Model
        try:
            with open(COLLAB_MODEL_PATH, "rb") as f:
                self.collab_model = pickle.load(f)
            # A simple check to see if the model is valid
            if not hasattr(self.collab_model, 'sim'):
                 raise ValueError("Invalid collaborative model file.")
            print("Collaborative model loaded from file.")
        except (FileNotFoundError, ValueError):
            print("Rebuilding collaborative model...")
            self._build_collaborative_model()

    def get_recommendations(self, selected_movie_ids: List[str], num_recommendations: int = 10) -> List[dict]:
        """Generate hybrid recommendations."""
        if self.content_similarity_matrix is None and self.collab_model is None:
            return []
        if not all(mid in self.movies_df.index for mid in selected_movie_ids):
            return [] # Some selected movies are not in our DB

        # --- HYBRID LOGIC STEP 1: Get Content-Based Scores ---
        indices = [self.movies_df.index.get_loc(mid) for mid in selected_movie_ids]
        content_scores = np.mean(self.content_similarity_matrix[indices, :], axis=0)
        
        # --- HYBRID LOGIC STEP 2: Get Collaborative Scores ---
        collab_scores = np.zeros(len(self.movies_df))
        if self.collab_model:
            # For each movie the user likes, find similar movies from the collab model and add their scores
            for movie_id in selected_movie_ids:
                try:
                    inner_id = self.collab_model.trainset.to_inner_iid(movie_id)
                    neighbors = self.collab_model.get_neighbors(inner_id, k=20)
                    for neighbor_inner_id in neighbors:
                        neighbor_movie_id = self.collab_model.trainset.to_raw_iid(neighbor_inner_id)
                        if neighbor_movie_id in self.movies_df.index:
                            neighbor_idx = self.movies_df.index.get_loc(neighbor_movie_id)
                            # Add the similarity score to the corresponding movie's score
                            collab_scores[neighbor_idx] += self.collab_model.sim[inner_id][neighbor_inner_id]
                except ValueError:
                    # Movie not in training set, skip it
                    continue

        # --- HYBRID LOGIC STEP 3: Combine and Rank ---
        # Normalize scores to be on the same scale (0 to 1)
        content_scores_norm = self.scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
        collab_scores_norm = self.scaler.fit_transform(collab_scores.reshape(-1, 1)).flatten()

        # Combine with weights (e.g., 50% content, 50% collaborative)
        hybrid_scores = (0.5 * content_scores_norm) + (0.5 * collab_scores_norm)

        # Create a DataFrame for easy sorting
        recs_df = pd.DataFrame({
            'score': hybrid_scores,
            'id': self.movies_df.index
        })
        
        # Filter out already selected movies and sort
        recs_df = recs_df[~recs_df['id'].isin(selected_movie_ids)]
        recs_df = recs_df.sort_values('score', ascending=False).head(num_recommendations)

        # Fetch full movie data for the recommendations
        top_movie_ids = recs_df['id'].tolist()
        results = self.movies_df.loc[top_movie_ids].reset_index().to_dict(orient="records")
        return results

    def add_movie(self, title: str) -> Optional[dict]:
        """Add a new movie by title and rebuild models."""
        # Avoid adding duplicates
        existing_movie = self.movies_df[self.movies_df['title'].str.lower() == title.lower()]
        if not existing_movie.empty:
            return existing_movie.reset_index().to_dict(orient='records')[0]

        movie = self.fetch_movie_by_title(title)
        if movie and movie["id"] not in self.movies_df.index:
            new_movie_df = pd.DataFrame([movie]).set_index('id')
            self.movies_df = pd.concat([self.movies_df, new_movie_df])
            self.save_movies()
            # Rebuild models to include the new movie
            self._build_content_matrix()
            # Note: Collaborative model won't include this movie until it gets ratings
            return movie
        return None

    def _populate_dummy_ratings_if_needed(self):
        """Adds fake user ratings to the DB if it's empty."""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            ratings_count = cursor.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
            if ratings_count > 0:
                return # Data already exists

            print("No ratings found. Populating with dummy data for demonstration.")
            # Create 50 dummy users
            users = [(i, f"user_{i}") for i in range(1, 51)]
            cursor.executemany("INSERT OR IGNORE INTO users (id, username) VALUES (?, ?)", users)

            # Create dummy ratings
            movie_ids = self.movies_df.index.tolist()
            if not movie_ids:
                return
            
            ratings = []
            for user_id in range(1, 51):
                num_ratings = np.random.randint(5, 15)
                rated_movies = np.random.choice(movie_ids, num_ratings, replace=False)
                for movie_id in rated_movies:
                    rating = np.random.randint(6, 11) # Users tend to rate movies they like
                    ratings.append((user_id, movie_id, rating))
            
            cursor.executemany("INSERT INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?)", ratings)
            conn.commit()
            print(f"Added {len(ratings)} dummy ratings.")


# Instantiate recommender system
rec_sys = MovieRecommendationSystem()

# Pydantic Schemas
class MovieRequest(BaseModel):
    movie_ids: List[str]
    num_recommendations: Optional[int] = 10

class Movie(BaseModel):
    id: str
    title: str
    overview: str
    genres: str
    director: Optional[str] = None
    cast: str
    poster_path: str
    vote_average: float
    release_date: str

# API Endpoints (UNCHANGED)
@app.get("/movies", response_model=List[Movie])
async def get_movies():
    if rec_sys.movies_df.empty:
        raise HTTPException(status_code=500, detail="No movie data available")
    return rec_sys.movies_df.reset_index().to_dict(orient="records")

@app.post("/recommend", response_model=List[Movie])
async def recommend_movies(request: MovieRequest):
    recommendations = rec_sys.get_recommendations(
        request.movie_ids, request.num_recommendations or 10
    )
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found for given movies.")
    return recommendations

@app.get("/search/{title}", response_model=Movie)
async def search_movie(title: str, background_tasks: BackgroundTasks):
    # Check if movie is already in our DB
    found_movie = rec_sys.movies_df[rec_sys.movies_df['title'].str.contains(title, case=False)]
    if not found_movie.empty:
        return found_movie.iloc[0].name, found_movie.iloc[0].to_dict()

    # If not, search for it and add it
    movie_data = rec_sys.add_movie(title)
    if not movie_data:
        raise HTTPException(status_code=404, detail="Movie not found via OMDb API.")
    return movie_data
