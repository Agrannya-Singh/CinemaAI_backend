import os
import sqlite3
import pickle
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
SIM_MATRIX_PATH = "content_similarity_matrix.pkl"
USER_MATRIX_PATH = "user_similarity_matrix.pkl"

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
            director TEXT,
            poster_path TEXT,
            vote_average REAL,
            release_date TEXT,
            combined_features TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_ratings (
            user_id TEXT,
            movie_id TEXT,
            rating REAL,
            PRIMARY KEY (user_id, movie_id)
        )
    ''')
    conn.commit()
    conn.close()

class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.user_ratings_df = None
        self.content_similarity_matrix = None
        self.user_similarity_matrix = None
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.API_KEY = OMDB_API_KEY
        self.BASE_URL = "http://www.omdbapi.com/"
        self.session = self._create_session()
        init_db()
        self.load_data()

    def _create_session(self):
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retries))
        return session

    def fetch_movie_by_title(self, title):
        """Fetch a single movie by title from OMDb API with retry logic."""
        params = {"apikey": self.API_KEY, "t": title, "plot": "full"}
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("Response") == "True":
                return {
                    "id": data.get("imdbID", f"unknown_{title}"),
                    "title": data.get("Title", ""),
                    "overview": data.get("Plot", ""),
                    "genres": data.get("Genre", ""),
                    "cast": data.get("Actors", ""),
                    "director": data.get("Director", ""),
                    "poster_path": data.get("Poster", ""),
                    "vote_average": float(data.get("imdbRating", 0)) if data.get("imdbRating", "0") not in ["N/A", None, ""] else 0.0,
                    "release_date": data.get("Year", ""),
                    "combined_features": f"{data.get('Genre', '')} {data.get('Actors', '')} {data.get('Director', '')} {data.get('Plot', '')}"
                }
            return None
        except Exception as e:
            print(f"Error fetching movie '{title}': {e}")
            return None

    def load_data(self):
        """Load movie and user rating data from SQLite or initialize with a larger dataset."""
        conn = sqlite3.connect(DB_PATH)
        self.movies_df = pd.read_sql_query("SELECT * FROM movies", conn)
        self.user_ratings_df = pd.read_sql_query("SELECT * FROM user_ratings", conn)
        conn.close()

        if self.movies_df.empty:
            print("No data in database. Initializing with a larger dataset.")
            # Simulate a larger dataset (replace with TMDB/MovieLens in production)
            titles = [
                "Inception", "The Dark Knight", "Interstellar", "The Matrix", "Pulp Fiction",
                "Forrest Gump", "The Shawshank Redemption", "Gladiator", "Titanic", "Avatar",
                "The Godfather", "Fight Club", "The Lord of the Rings: The Fellowship of the Ring",
                "Star Wars: Episode IV – A New Hope", "Jurassic Park", "The Avengers", "Parasite",
                "La La Land", "Mad Max: Fury Road", "The Wolf of Wall Street",  # Expanded list
                "Blade Runner 2049", "Get Out", "Dunkirk", "The Grand Budapest Hotel", 
                "Whiplash", "Spirited Away", "The Social Network", "Moonlight", "Coco",
                "The Empire Strikes Back", "Back to the Future", "Toy Story", "The Lion King",
                "Schindler's List", "Good Will Hunting", "Memento", "Django Unchained",
                "The Prestige", "Se7en", "The Departed", "Inglourious Basterds"
            ]
            movies_data = [self.fetch_movie_by_title(title) for title in titles if self.fetch_movie_by_title(title)]
            if movies_data:
                self.movies_df = pd.DataFrame(movies_data)
                self.save_data()
            else:
                print("Using fallback dataset.")
                self.movies_df = pd.DataFrame([
                    {
                        "id": "tt0372784",
                        "title": "Batman Begins",
                        "overview": "Bruce Wayne becomes Batman.",
                        "genres": "Action, Adventure, Crime",
                        "cast": "Christian Bale, Michael Caine",
                        "director": "Christopher Nolan",
                        "poster_path": "",
                        "vote_average": 8.2,
                        "release_date": "2005",
                        "combined_features": "Action Adventure Crime Christian Bale Michael Caine Christopher Nolan Bruce Wayne becomes Batman."
                    }
                ])
            # Simulate user ratings for collaborative filtering
            self.user_ratings_df = pd.DataFrame([
                {"user_id": "user1", "movie_id": "tt1375666", "rating": 4.5},  # Inception
                {"user_id": "user1", "movie_id": "tt0468569", "rating": 5.0},  # The Dark Knight
                {"user_id": "user2", "movie_id": "tt0816692", "rating": 4.0},  # Interstellar
                {"user_id": "user2", "movie_id": "tt0137523", "rating": 4.8},  # Fight Club
            ])
            self.save_data()
        self.load_similarity_matrices()

    def save_data(self):
        """Save movie and user rating data to SQLite."""
        conn = sqlite3.connect(DB_PATH)
        self.movies_df.to_sql("movies", conn, if_exists="replace", index=False)
        self.user_ratings_df.to_sql("user_ratings", conn, if_exists="replace", index=False)
        conn.close()

    def build_content_similarity_matrix(self):
        """Build and save content-based similarity matrix using TF-IDF."""
        if self.movies_df is not None and not self.movies_df.empty:
            corpus = self.movies_df["combined_features"].fillna("").astype(str).tolist()
            if not any(corpus):
                print("Corpus is empty.")
                self.content_similarity_matrix = None
                return
            try:
                vectorized_features = self.vectorizer.fit_transform(corpus)
                self.content_similarity_matrix = cosine_similarity(vectorized_features, vectorized_features)
                self.content_similarity_matrix = csr_matrix(self.content_similarity_matrix)  # Use sparse matrix
                with open(SIM_MATRIX_PATH, "wb") as f:
                    pickle.dump(self.content_similarity_matrix, f)
                print(f"Content similarity matrix built and saved: {self.content_similarity_matrix.shape}")
            except Exception as e:
                print(f"Error building content similarity matrix: {e}")
                self.content_similarity_matrix = None
        else:
            print("No movie data to build content similarity matrix.")
            self.content_similarity_matrix = None

    def build_user_similarity_matrix(self):
        """Build user similarity matrix based on ratings."""
        if self.user_ratings_df is not None and not self.user_ratings_df.empty:
            user_movie_matrix = self.user_ratings_df.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
            try:
                self.user_similarity_matrix = cosine_similarity(user_movie_matrix)
                self.user_similarity_matrix = csr_matrix(self.user_similarity_matrix)
                with open(USER_MATRIX_PATH, "wb") as f:
                    pickle.dump(self.user_similarity_matrix, f)
                print(f"User similarity matrix built and saved: {self.user_similarity_matrix.shape}")
            except Exception as e:
                print(f"Error building user similarity matrix: {e}")
                self.user_similarity_matrix = None
        else:
            print("No user ratings to build user similarity matrix.")
            self.user_similarity_matrix = None

    def load_similarity_matrices(self):
        """Load content and user similarity matrices from disk or rebuild."""
        try:
            with open(SIM_MATRIX_PATH, "rb") as f:
                self.content_similarity_matrix = pickle.load(f)
                if self.content_similarity_matrix.shape[0] != len(self.movies_df):
                    print("Content similarity matrix size mismatch. Rebuilding.")
                    self.build_content_similarity_matrix()
                else:
                    print("Content similarity matrix loaded from disk.")
        except FileNotFoundError:
            print("No content similarity matrix found. Building new one.")
            self.build_content_similarity_matrix()

        try:
            with open(USER_MATRIX_PATH, "rb") as f:
                self.user_similarity_matrix = pickle.load(f)
                if self.user_similarity_matrix.shape[0] != len(self.user_ratings_df["user_id"].unique()):
                    print("User similarity matrix size mismatch. Rebuilding.")
                    self.build_user_similarity_matrix()
                else:
                    print("User similarity matrix loaded from disk.")
        except FileNotFoundError:
            print("No user similarity matrix found. Building new one.")
            self.build_user_similarity_matrix()

    def update_content_similarity_matrix(self, new_movie_index):
        """Incrementally update content similarity matrix for a new movie."""
        if self.content_similarity_matrix is None or self.movies_df is None:
            self.build_content_similarity_matrix()
            return
        new_features = self.movies_df.iloc[new_movie_index]["combined_features"]
        vectorized_new = self.vectorizer.transform([new_features])
        new_similarities = cosine_similarity(vectorized_new, self.vectorizer.transform(self.movies_df["combined_features"]))
        new_similarities = csr_matrix(new_similarities)
        self.content_similarity_matrix = csr_matrix(
            np.vstack([self.content_similarity_matrix.toarray(), new_similarities.toarray()])
        )
        self.content_similarity_matrix = csr_matrix(
            np.hstack([self.content_similarity_matrix.toarray(), new_similarities.T.toarray()])
        )
        with open(SIM_MATRIX_PATH, "wb") as f:
            pickle.dump(self.content_similarity_matrix, f)
        print(f"Content similarity matrix updated for new movie.")

    def get_hybrid_recommendations(self, user_id: str, movie_ids: List[str], preferred_genres: Optional[List[str]] = None, num_recommendations: int = 5) -> List[dict]:
        """Get hybrid recommendations combining content-based and collaborative filtering."""
        if self.content_similarity_matrix is None or self.movies_df is None or self.movies_df.empty:
            return []

        # Content-based recommendations
        content_recs = []
        valid_indices = self.movies_df[self.movies_df["id"].isin(movie_ids)].index.tolist()
        if valid_indices:
            avg_content_scores = np.mean(self.content_similarity_matrix[valid_indices, :].toarray(), axis=0)
        else:
            avg_content_scores = np.zeros(self.content_similarity_matrix.shape[1])

        # Collaborative filtering recommendations
        collab_recs = []
        if self.user_similarity_matrix is not None and not self.user_ratings_df.empty:
            user_idx = self.user_ratings_df["user_id"].unique().tolist().index(user_id) if user_id in self.user_ratings_df["user_id"].values else None
            if user_idx is not None:
                similar_users = np.argsort(self.user_similarity_matrix[user_idx].toarray().flatten())[::-1][1:]  # Exclude self
                user_movie_matrix = self.user_ratings_df.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
                collab_scores = np.zeros(len(self.movies_df))
                for sim_user_idx in similar_users[:10]:  # Top 10 similar users
                    sim_score = self.user_similarity_matrix[user_idx, sim_user_idx]
                    user_ratings = user_movie_matrix.iloc[sim_user_idx]
                    collab_scores += sim_score * user_ratings.reindex(self.movies_df["id"], fill_value=0).values
                collab_recs = collab_scores / (np.sum(self.user_similarity_matrix[user_idx].toarray()) + 1e-10)

        # Combine scores (weight content-based 0.6, collaborative 0.4)
        final_scores = 0.6 * avg_content_scores + 0.4 * (collab_recs if len(collab_recs) > 0 else np.zeros_like(avg_content_scores))

        # Filter by preferred genres if provided
        if preferred_genres:
            genre_mask = self.movies_df["genres"].apply(lambda x: any(g in x.lower() for g in [pg.lower() for pg in preferred_genres]))
            final_scores = final_scores * genre_mask.values

        # Get top recommendations
        sorted_indices = np.argsort(final_scores)[::-1]
        recommendations = []
        seen = set(movie_ids)
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
            self.update_content_similarity_matrix(len(self.movies_df) - 1)
            return movie_data
        return None

    def add_user_rating(self, user_id: str, movie_id: str, rating: float):
        """Add or update a user rating and update user similarity matrix."""
        if movie_id not in self.movies_df["id"].values:
            return False
        new_rating = pd.DataFrame([{"user_id": user_id, "movie_id": movie_id, "rating": rating}])
        self.user_ratings_df = pd.concat([self.user_ratings_df, new_rating], ignore_index=True).drop_duplicates(subset=["user_id", "movie_id"], keep="last")
        self.save_data()
        self.build_user_similarity_matrix()
        return True

# Initialize recommender
rec_sys = MovieRecommendationSystem()

# Pydantic models
class MovieRequest(BaseModel):
    user_id: str
    movie_ids: List[str]
    preferred_genres: Optional[List[str]] = None
    num_recommendations: int = 5

class Movie(BaseModel):
    id: str
    title: str
    overview: str
    genres: str
    cast: str
    director: str
    poster_path: str
    vote_average: float
    release_date: str

class RatingRequest(BaseModel):
    user_id: str
    movie_id: str
    rating: float

@app.get("/movies", response_model=List[Movie])
async def get_movies():
    """Return all available movies."""
    if rec_sys.movies_df is None or rec_sys.movies_df.empty:
        raise HTTPException(status_code=500, detail="Movie data not available")
    return rec_sys.movies_df.to_dict(orient="records")

@app.post("/recommend", response_model=List[Movie])
async def get_recommendations(request: MovieRequest):
    """Get hybrid movie recommendations."""
    recommendations = rec_sys.get_hybrid_recommendations(
        request.user_id, request.movie_ids, request.preferred_genres, request.num_recommendations
    )
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

@app.post("/rate", response_model=bool)
async def rate_movie(request: RatingRequest):
    """Add or update a user rating."""
    success = rec_sys.add_user_rating(request.user_id, request.movie_id, request.rating)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid movie ID or rating")
    return success
