import os
import sqlite3
import requests
import pandas as pd

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

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
if not OMDB_API_KEY:
    raise ValueError("OMDB_API_KEY not found in environment variables.")

# Constants
DB_PATH = "/var/data/movies.db"
OMDB_API_URL = "http://www.omdbapi.com/"
TFIDF_MAX_FEATURES = 5000
API_TIMEOUT = 10
CONTENT_WEIGHT = 0.5
COLLAB_WEIGHT = 0.5

# --- Recommendation Engine Class (with Search/Add functionality) ---


class MovieRecommender:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self.movies_df = pd.DataFrame()
        self.ratings_df = pd.DataFrame()
        self.content_matrix = None
        self.collab_model = None
        self.user_movie_matrix = None
        self.scaler = MinMaxScaler()
        self.load_data()
        self.build_models()

    def load_data(self) -> None:
        """
        Load movies and ratings from the database.
        
        This method ensures the database tables exist and loads the data
        into DataFrames.
        
        Raises:
            sqlite3.Error: If a database error occurs.
        """
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
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
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ratings (
                        user_id INTEGER,
                        movie_id TEXT,
                        rating REAL
                    )
                """)
                self.movies_df = pd.read_sql_query(
                    "SELECT * FROM movies", conn, index_col="id"
                )
                self.ratings_df = pd.read_sql_query(
                    "SELECT * FROM ratings", conn
                )
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise

    def build_models(self) -> None:
        """
        Build (or rebuild) both recommendation models.
        
        This method calls the private methods to build content-based
        and collaborative filtering models.
        """
        self._build_content_model()
        self._build_collaborative_model()

    def _build_content_model(self) -> None:
        """
        Build the content-based similarity matrix using TF-IDF.
        
        This method creates a TF-IDF vectorizer from the 'combined_features'
        column of the movies DataFrame and computes a cosine similarity matrix.
        """
        logger.info("Building content-based model...")
        if self.movies_df.empty:
            logger.warning("No movies in DataFrame to build content model.")
            return
        corpus = self.movies_df["combined_features"].fillna("").astype(str)
        vectorizer = TfidfVectorizer(
            stop_words="english", max_features=TFIDF_MAX_FEATURES
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        self.content_matrix = cosine_similarity(tfidf_matrix)
        logger.info("Content-based model built.")

    def _build_collaborative_model(self) -> None:
        """
        Build the item-based collaborative model using NearestNeighbors.
        
        This method pivots the ratings DataFrame into a user-movie matrix
        and fits a NearestNeighbors model for collaborative filtering.
        """
        if self.ratings_df.empty or len(self.ratings_df["movie_id"].unique()) < 2:
            logger.warning("Not enough data to build collaborative model.")
            return
        logger.info("Building collaborative model...")
        user_movie_matrix = self.ratings_df.pivot(
            index="movie_id", columns="user_id", values="rating"
        ).fillna(0)
        # Ensure all movies are in the matrix
        all_movie_ids = set(self.movies_df.index)
        missing_movies = all_movie_ids - set(user_movie_matrix.index)
        for movie_id in missing_movies:
            user_movie_matrix.loc[movie_id] = 0
        movie_features_sparse = csr_matrix(user_movie_matrix.values)
        self.collab_model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.collab_model.fit(movie_features_sparse)
        self.user_movie_matrix = user_movie_matrix
        logger.info("Collaborative model built.")

    def _fetch_movie_from_api(self, title: str) -> Optional[dict]:
        """
        Fetch movie data from OMDb API by title.
        
        Args:
            title (str): The movie title to search for.
        
        Returns:
            Optional[dict]: Movie data if found, None otherwise.
        """
        params = {"apikey": OMDB_API_KEY, "t": title, "plot": "full"}
        try:
            resp = requests.get(OMDB_API_URL, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if data.get("Response") != "True":
                logger.info(f"Movie '{title}' not found in OMDb API.")
                return None
            return self._process_movie_data(data)
        except requests.Timeout:
            logger.error(f"Timeout fetching movie '{title}'")
            return None
        except requests.ConnectionError:
            logger.error(f"Network error fetching movie '{title}'")
            return None
        except requests.HTTPError as e:
            logger.error(f"HTTP error fetching movie '{title}': {e}")
            return None

    def _process_movie_data(self, data: dict) -> dict:
        """
        Process raw OMDb API data into a structured movie dictionary.
        
        Args:
            data (dict): Raw API response data.
        
        Returns:
            dict: Processed movie data with combined features.
        """
        director = data.get("Director", "")
        genres = data.get("Genre", "")
        actors = data.get("Actors", "")
        director_clean = "".join(director.split()) if director else ""
        genres_clean = "".join(genres.split(",")) if genres else ""
        actors_clean = "".join(actors.split(",")[:3]) if actors else ""
        combined_features = (
            f"{director_clean} {genres_clean} {actors_clean} {data.get('Plot', '')}"
        )
        vote_average = data.get("imdbRating", "N/A")
        vote_average = float(vote_average) if vote_average != "N/A" else 0.0
        return {
            "id": data.get("imdbID"),
            "title": data.get("Title"),
            "overview": data.get("Plot"),
            "genres": genres,
            "director": director,
            "cast": actors,
            "poster_path": data.get("Poster"),
            "vote_average": vote_average,
            "release_date": data.get("Year"),
            "combined_features": combined_features,
        }

    def add_movie(self, title: str) -> Optional[dict]:
        """
        Search for a movie, add it to the DB, and rebuild models.
        
        Args:
            title (str): The movie title to add.
        
        Returns:
            Optional[dict]: The added movie data if successful, None otherwise.
        """
        movie_data = self._fetch_movie_from_api(title)
        if movie_data:
            if movie_data["id"] not in self.movies_df.index:
                logger.info(f"Adding new movie: {movie_data['title']}")
                new_movie_df = pd.DataFrame([movie_data]).set_index("id")
                self.movies_df = pd.concat([self.movies_df, new_movie_df])
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        new_movie_df.reset_index().to_sql(
                            "movies", conn, if_exists="append", index=False
                        )
                except sqlite3.Error as e:
                    logger.error(f"Database error adding movie: {e}")
                    return None
                self.build_models()
            return movie_data
        return None

    def get_recommendations(
        self, selected_movie_ids: list, num_recs: int = 10
    ) -> list:
        """
        Generate hybrid recommendations.
        
        Args:
            selected_movie_ids (list): List of movie IDs to base recommendations on.
            num_recs (int): Number of recommendations to return (default: 10).
        
        Returns:
            list: List of recommended movie dictionaries.
        
        Raises:
            HTTPException: If models are not ready or movies not found.
        """
        if self.content_matrix is None:
            raise HTTPException(
                status_code=503, detail="Recommendation models are not ready."
            )
        if not all(mid in self.movies_df.index for mid in selected_movie_ids):
            raise HTTPException(
                status_code=404, detail="One or more selected movies not found in the database."
            )

        indices = [self.movies_df.index.get_loc(mid) for mid in selected_movie_ids]
        content_scores = np.mean(self.content_matrix[indices, :], axis=0)
        collab_scores = np.zeros(len(self.movies_df))
        if self.collab_model and self.user_movie_matrix is not None:
            for movie_id in selected_movie_ids:
                if movie_id in self.user_movie_matrix.index:
                    movie_idx = self.user_movie_matrix.index.get_loc(movie_id)
                    distances, indices = self.collab_model.kneighbors(
                        self.user_movie_matrix.iloc[movie_idx, :].values.reshape(1, -1),
                        n_neighbors=min(20, len(self.user_movie_matrix.index)),
                    )
                    sim_scores = 1 - distances.flatten()
                    for i, neighbor_idx in enumerate(indices.flatten()):
                        neighbor_movie_id = self.user_movie_matrix.index[neighbor_idx]
                        if neighbor_movie_id in self.movies_df.index:
                            main_df_idx = self.movies_df.index.get_loc(neighbor_movie_id)
                            collab_scores[main_df_idx] += sim_scores[i]

        content_scores_norm = self.scaler.fit_transform(
            content_scores.reshape(-1, 1)
        ).flatten()
        collab_scores_norm = self.scaler.fit_transform(
            collab_scores.reshape(-1, 1)
        ).flatten()
        hybrid_scores = (CONTENT_WEIGHT * content_scores_norm) + (
            COLLAB_WEIGHT * collab_scores_norm
        )
        recs_df = pd.DataFrame({"score": hybrid_scores, "id": self.movies_df.index})
        recs_df = recs_df[~recs_df["id"].isin(selected_movie_ids)]
        recs_df = recs_df.sort_values("score", ascending=False).head(num_recs)
        results = self.movies_df.loc[recs_df["id"].tolist()].reset_index().to_dict(
            orient="records"
        )
        return results


# --- FastAPI Application ---

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the recommender system once on startup
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
    if recommender.movies_df.empty:
        return []
    return recommender.movies_df.reset_index().to_dict(orient="records")


@app.post("/recommend", response_model=List[Movie])
async def recommend_movies(request: MovieRequest):
    if recommender.movies_df.empty:
        raise HTTPException(status_code=404, detail="No movies in the database to make recommendations.")
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
