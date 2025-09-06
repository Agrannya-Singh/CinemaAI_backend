import os
import sqlite3
import requests
import logging
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not OMDB_API_KEY:
    raise ValueError("OMDB_API_KEY not found in environment variables.")

# IMPORTANT: This path MUST match the Mount Path of your attached Render Disk.
# Free tier filesystems are temporary. You need a paid Render Disk for data to persist.
DB_PATH = "/data/render/disk/movies.db"
OMDB_API_URL = "http://www.omdbapi.com/"

CONFIG = {
    "TFIDF_MAX_FEATURES": 5000,
    "API_TIMEOUT_SECONDS": 10,
    "CONTENT_WEIGHT": 0.5,
    "COLLAB_WEIGHT": 0.5,
    "SEED_MOVIE_COUNT": 500,
}


# --- Recommendation Engine Class ---

class MovieRecommender:
    """Manages movie data, recommendation models, and recommendation logic."""

    def __init__(self, db_path: str = DB_PATH):
        """Initializes the recommender, loads data, and builds models."""
        self.db_path = db_path
        self.movies_df = pd.DataFrame()
        self.ratings_df = pd.DataFrame()
        self.content_matrix = None
        self.collab_model = None
        self.user_movie_matrix = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.scaler = MinMaxScaler()
        self.load_data()
        self.build_models()

    def _seed_initial_movies(self) -> None:
        """Fetches popular movies to seed the database if it's empty."""
        logging.info("Database is empty. Seeding with popular movies...")
        if not TMDB_API_KEY:
            logging.warning("TMDB_API_KEY not found. Cannot seed database.")
            return

        popular_movies_titles = set()
        num_pages = (CONFIG["SEED_MOVIE_COUNT"] // 20) + 1  # TMDB has 20 movies per page
        try:
            for page in range(1, num_pages + 1):
                url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&page={page}"
                response = requests.get(url, timeout=CONFIG["API_TIMEOUT_SECONDS"])
                response.raise_for_status()
                for movie in response.json().get("results", []):
                    popular_movies_titles.add(movie["title"])
        except requests.RequestException as e:
            logging.error(f"Failed to fetch popular movies from TMDB: {e}")
            return

        logging.info(f"Fetched {len(popular_movies_titles)} unique titles. Fetching details...")
        new_movies = [
            self._process_api_data(raw_data)
            for title in popular_movies_titles
            if (raw_data := self._fetch_movie_from_api(title))
        ]

        if not new_movies:
            logging.error("Failed to fetch any movie details for seeding.")
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                pd.DataFrame(new_movies).to_sql("movies", conn, if_exists="append", index=False)
            logging.info(f"Successfully seeded database with {len(new_movies)} movies.")
        except sqlite3.Error as e:
            logging.error(f"Database error during seeding: {e}")

    def load_data(self) -> None:
        """Loads data and seeds the database on the first run."""
        logging.info("Loading data from database...")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS movies (
                        id TEXT PRIMARY KEY, title TEXT, overview TEXT, genres TEXT,
                        director TEXT, cast TEXT, poster_path TEXT, vote_average REAL,
                        release_date TEXT, combined_features TEXT
                    )
                    """
                )
                conn.execute("CREATE TABLE IF NOT EXISTS ratings (user_id INTEGER, movie_id TEXT, rating REAL)")

                if conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0] == 0:
                    self._seed_initial_movies()

                self.movies_df = pd.read_sql_query("SELECT * FROM movies", conn, index_col='id')
                self.ratings_df = pd.read_sql_query("SELECT * FROM ratings", conn)
                logging.info(f"Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings.")
        except sqlite3.Error as e:
            logging.error(f"Database error during data load: {e}")
            raise

    def build_models(self) -> None:
        """Builds (or rebuilds) both recommendation models."""
        self._build_content_model()
        self._build_collaborative_model()

    def _build_content_model(self) -> None:
        """Builds the content-based similarity matrix using TF-IDF."""
        logging.info("Building content-based model...")
        if self.movies_df.empty:
            logging.warning("No movies to build content model.")
            return
        corpus = self.movies_df["combined_features"].fillna("").astype(str)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=CONFIG["TFIDF_MAX_FEATURES"])
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.content_matrix = cosine_similarity(self.tfidf_matrix)
        logging.info("Content-based model built successfully.")

    def _build_collaborative_model(self) -> None:
        """Builds the item-based collaborative model using NearestNeighbors."""
        logging.info("Building collaborative model...")
        if self.ratings_df.empty or self.movies_df.empty or len(self.ratings_df['movie_id'].unique()) < 2:
            logging.warning("Not enough data to build collaborative model.")
            return

        user_movie_matrix = self.ratings_df.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
        self.user_movie_matrix = user_movie_matrix.reindex(self.movies_df.index, fill_value=0)
        movie_features_sparse = csr_matrix(self.user_movie_matrix.values)
        self.collab_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.collab_model.fit(movie_features_sparse)
        logging.info("Collaborative model built successfully.")

    def _fetch_movie_from_api(self, title: str) -> Optional[dict]:
        """Fetches raw movie data from the OMDb API."""
        params = {"apikey": OMDB_API_KEY, "t": title, "plot": "full"}
        try:
            resp = requests.get(OMDB_API_URL, params=params, timeout=CONFIG["API_TIMEOUT_SECONDS"])
            resp.raise_for_status()
            data = resp.json()
            return data if data.get("Response") == "True" else None
        except requests.Timeout:
            logging.error(f"Timeout fetching '{title}' from OMDb.")
        except requests.RequestException as e:
            logging.error(f"Error fetching '{title}': {e}")
        return None

    def _process_api_data(self, data: dict) -> dict:
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
            "poster_path": data.get("Poster"), "vote_average": float(data.get("imdbRating", 0.0) if data.get("imdbRating") != "N/A" else 0.0),
            "release_date": data.get("Year"), "combined_features": combined_features
        }

    def _update_content_matrix_for_new_movie(self, new_movie_features: str) -> None:
        """Quickly updates the content matrix in memory for a new movie."""
        if self.content_matrix is None or self.vectorizer is None:
            return
        new_movie_vector = self.vectorizer.transform([new_movie_features])
        new_sim_scores = cosine_similarity(new_movie_vector, self.tfidf_matrix).flatten()
        self.content_matrix = np.hstack([self.content_matrix, new_sim_scores.reshape(-1, 1)])
        new_row = np.append(new_sim_scores, 1.0)
        self.content_matrix = np.vstack([self.content_matrix, new_row])
        self.tfidf_matrix = np.vstack([self.tfidf_matrix, new_movie_vector])

    def find_or_add_movie(self, title: str, background_tasks: BackgroundTasks) -> Optional[dict]:
        """
        Finds a movie in the local DB first (cache). If not found,
        fetches from the API, adds it, and triggers a background rebuild.
        """
        if not self.movies_df.empty:
            local_match = self.movies_df[self.movies_df['title'].str.lower() == title.lower()]
            if not local_match.empty:
                logging.info(f"Found '{title}' in local database cache.")
                return local_match.reset_index().iloc[0].to_dict()

        logging.info(f"'{title}' not in local cache. Fetching from OMDb API.")
        raw_data = self._fetch_movie_from_api(title)
        if raw_data and (movie_data := self._process_api_data(raw_data)):
            if movie_data["id"] not in self.movies_df.index:
                logging.info(f"Adding new movie: {movie_data['title']}")
                new_movie_df = pd.DataFrame([movie_data]).set_index('id')
                
                self._update_content_matrix_for_new_movie(new_movie_df.iloc[0]["combined_features"])
                self.movies_df = pd.concat([self.movies_df, new_movie_df])
                
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        new_movie_df.reset_index().to_sql("movies", conn, if_exists="append", index=False)
                except sqlite3.Error as e:
                    logging.error(f"Database error while adding movie: {e}")
                    return None
                
                background_tasks.add_task(self.build_models)
                
            return movie_data
        return None

    def get_recommendations(self, selected_movie_ids: list, num_recs: int = 10) -> List[dict]:
        """Generates hybrid recommendations."""
        if self.content_matrix is None:
            raise HTTPException(status_code=503, detail="Recommendation models are not ready.")
        if not all(mid in self.movies_df.index for mid in selected_movie_ids):
            raise HTTPException(status_code=404, detail="One or more selected movies not found.")

        indices = [self.movies_df.index.get_loc(mid) for mid in selected_movie_ids]
        content_scores = np.mean(self.content_matrix[indices, :], axis=0)

        collab_scores = np.zeros(len(self.movies_df))
        if self.collab_model and self.user_movie_matrix is not None:
            for movie_id in selected_movie_ids:
                if movie_id in self.user_movie_matrix.index:
                    movie_idx = self.user_movie_matrix.index.get_loc(movie_id)
                    n_neighbors = min(20, len(self.user_movie_matrix.index))
                    distances, neighbor_indices = self.collab_model.kneighbors(
                        self.user_movie_matrix.iloc[movie_idx, :].values.reshape(1, -1), n_neighbors=n_neighbors
                    )
                    sim_scores = 1 - distances.flatten()
                    for i, neighbor_idx in enumerate(neighbor_indices.flatten()):
                        neighbor_movie_id = self.user_movie_matrix.index[neighbor_idx]
                        if neighbor_movie_id in self.movies_df.index:
                            main_df_idx = self.movies_df.index.get_loc(neighbor_movie_id)
                            collab_scores[main_df_idx] += sim_scores[i]

        content_scores_norm = self.scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
        collab_scores_norm = self.scaler.fit_transform(collab_scores.reshape(-1, 1)).flatten()
        hybrid_scores = (CONFIG["CONTENT_WEIGHT"] * content_scores_norm) + (CONFIG["COLLAB_WEIGHT"] * collab_scores_norm)

        recs_df = pd.DataFrame({'score': hybrid_scores, 'id': self.movies_df.index})
        recs_df = recs_df[~recs_df['id'].isin(selected_movie_ids)]
        recs_df = recs_df.sort_values('score', ascending=False).head(num_recs)

        return self.movies_df.loc[recs_df['id'].tolist()].reset_index().to_dict(orient="records")


# --- FastAPI Application ---

app = FastAPI(title="CinemaAI Recommendation API", description="A hybrid movie recommendation engine.")

# WARNING: This is insecure for production. Restrict origins to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = MovieRecommender()


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
    """Returns a list of all movies currently in the database."""
    if recommender.movies_df.empty:
        return []
    return recommender.movies_df.reset_index().to_dict(orient="records")

@app.post("/recommend", response_model=List[Movie])
async def recommend_movies(request: MovieRequest):
    """Generates movie recommendations based on a list of selected movie IDs."""
    if recommender.movies_df.empty:
        raise HTTPException(status_code=404, detail="No movies in DB to make recommendations.")
    recommendations = recommender.get_recommendations(request.movie_ids, request.num_recommendations or 10)
    if not recommendations:
        raise HTTPException(status_code=404, detail="Could not generate recommendations.")
    return recommendations

@app.get("/search/{title}", response_model=Movie)
async def search_movie(title: str, background_tasks: BackgroundTasks):
    """
    Searches for a movie first in the local DB. If not found, fetches
    from the OMDb API, adds it, and triggers a background rebuild.
    """
    movie = recommender.find_or_add_movie(title, background_tasks)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found.")
    return movie
