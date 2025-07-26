# Import necessary libraries
import os
import threading
import time
import json
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import html
import re

# Load environment variables from .env file
# This allows sensitive data like API keys to be stored securely
load_dotenv()

# --- MovieRecommendationSystem Class ---
# This class handles fetching movie data from OMDb API, preparing data, and generating recommendations
class MovieRecommendationSystem:
    def __init__(self):
        # Initialize instance variables
        self.movies_df = None  # DataFrame to store movie data
        self.similarity_matrix = None  # Matrix for storing cosine similarity scores
        self.vectorizer = CountVectorizer(stop_words='english')  # For text vectorization
        self.API_KEY = os.getenv("OMDB_API_KEY")  # Load OMDb API key from environment
        self.BASE_URL = "http://www.omdbapi.com/"  # OMDb API base URL
        self.HEADERS = {}  # Optional headers for API requests
        # Check if API key is present
        if not self.API_KEY:
            print("🚨 WARNING: OMDB_API_KEY not found in environment variables.")

    def fetch_movie_by_title(self, title: str) -> Optional[Dict]:
        """Fetch a single movie's data from OMDb API by title."""
        if not self.API_KEY:
            print("🚨 OMDb API key is missing. Cannot fetch movie.")
            return None
        params = {"apikey": self.API_KEY, "t": title, "plot": "full"}
        try:
            # Make API request with a timeout of 10 seconds
            response = requests.get(self.BASE_URL, headers=self.HEADERS, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("Response") == "True":
                    return data
            print(f"Error fetching movie '{title}': {response.status_code} or movie not found")
            return None
        except requests.exceptions.Timeout:
            print(f"Timeout fetching movie '{title}'.")
            return None
        except Exception as e:
            print(f"Error fetching movie '{title}': {e}")
            return None

    def fetch_movies(self, titles: Optional[List[str]] = None, limit: int = 50) -> List[Dict]:
        """Fetch a list of movies from OMDb API or use a default list if none provided."""
        if titles is None:
            # Default list of popular movie titles (limited to avoid excessive API calls)
            titles = [
                "Inception", "The Dark Knight", "Interstellar", "The Matrix", "Fight Club",
                "Pulp Fiction", "Forrest Gump", "The Shawshank Redemption", "Gladiator", "Titanic",
                "Avatar", "The Avengers", "Jurassic Park", "Star Wars", "The Lord of the Rings"
            ][:limit]
        movies = []
        for title in titles[:limit]:
            movie_data = self.fetch_movie_by_title(title)
            if movie_data:
                movies.append(movie_data)
        return movies

    def prepare_movie_data(self) -> pd.DataFrame:
        """Prepare movie data by fetching from OMDb or using fallback data."""
        movies = self.fetch_movies()
        if not movies:
            print("🚨 API returned no movies. Using fallback dataset.")
            # Fallback dataset in case API fails
            fallback_movies = [
                {
                    'id': 'tt0372784', 'title': 'Batman Begins',
                    'overview': 'A young Bruce Wayne becomes Batman to fight crime in Gotham.',
                    'genres': 'Action, Adventure, Crime', 'cast': 'Christian Bale, Michael Caine',
                    'poster_path': 'https://m.media-amazon.com/images/M/MV5BMjE3NDcyNDExNF5BMl5BanBnXkFtZTcwMDYwNDk0OA@@._V1_SX300.jpg',
                    'vote_average': 8.2, 'release_date': '2005',
                    'combined_features': 'Action Adventure Crime Christian Bale Michael Caine A young Bruce Wayne becomes Batman to fight crime in Gotham.'
                },
                # Add more fallback movies as needed
            ]
            self.movies_df = pd.DataFrame(fallback_movies)
        else:
            print(f"✅ Fetched {len(movies)} movies from OMDb API.")
            movie_data = []
            for movie in movies:
                # Structure movie data for DataFrame
                movie_info = {
                    'id': movie.get('imdbID', movie.get('Title', 'unknown')),
                    'title': movie.get('Title', ''),
                    'overview': movie.get('Plot', ''),
                    'genres': movie.get('Genre', ''),
                    'cast': movie.get('Actors', ''),
                    'poster_path': movie.get('Poster', ''),
                    'vote_average': float(movie.get('imdbRating', '0')) if movie.get('imdbRating') not in ['N/A', None] else 0,
                    'release_date': movie.get('Year', ''),
                    'combined_features': f"{movie.get('Genre', '')} {movie.get('Actors', '')} {movie.get('Plot', '')}"
                }
                movie_data.append(movie_info)
            self.movies_df = pd.DataFrame(movie_data)
        # Build similarity matrix for recommendations
        self.build_similarity_matrix()
        return self.movies_df

    def build_similarity_matrix(self) -> None:
        """Build a cosine similarity matrix based on movie features."""
        if self.movies_df is not None and not self.movies_df.empty:
            # Use CountVectorizer to convert text features into numerical vectors
            self.vectorizer = CountVectorizer(stop_words='english', max_features=5000)
            corpus = self.movies_df['combined_features'].fillna('').tolist()
            vectorized_features = self.vectorizer.fit_transform(corpus)
            self.similarity_matrix = cosine_similarity(vectorized_features)
            print(f"✅ Similarity matrix built with shape: {self.similarity_matrix.shape}")
        else:
            print("🚨 Cannot build similarity matrix: movies_df is empty.")

    def get_recommendations(self, selected_movie_ids: List[str], num_recommendations: int = 5) -> List[Dict]:
        """Generate movie recommendations based on selected movie IDs."""
        if self.similarity_matrix is None or self.movies_df.empty:
            print("🚨 Similarity matrix or movies_df is empty.")
            return []
        # Find indices of selected movies
        selected_indices = self.movies_df[self.movies_df['id'].isin(selected_movie_ids)].index.tolist()
        if not selected_indices:
            print("🚨 No selected movies found in DataFrame.")
            return []
        # Calculate average similarity scores
        avg_similarity_scores = np.mean(self.similarity_matrix[selected_indices], axis=0)
        # Sort movies by similarity
        movie_indices = np.argsort(avg_similarity_scores)[::-1]
        recommendations = []
        for idx in movie_indices:
            movie = self.movies_df.iloc[idx]
            if movie['id'] not in selected_movie_ids:  # Exclude selected movies
                recommendations.append(movie.to_dict())
                if len(recommendations) >= num_recommendations:
                    break
        return recommendations

# Initialize the recommender
recommender = MovieRecommendationSystem()

# --- Flask Application ---
# Create a Flask app for the RESTful API
app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests from different origins

@app.route('/')
def index():
    """Root endpoint for health check and API documentation."""
    return jsonify({
        "message": "CinemaAI API is running!",
        "status": "success",
        "endpoints": {
            "/api/movies": "GET - Fetch all movies",
            "/api/recommend": "POST - Get movie recommendations",
            "/api/health": "GET - Check API health"
        }
    })

@app.route('/api/movies', methods=['GET'])
def get_movies():
    """Fetch all movies from the recommender system."""
    try:
        if recommender.movies_df is None or recommender.movies_df.empty:
            print("Preparing movie data...")
            recommender.prepare_movie_data()
        movies = recommender.movies_df.to_dict('records')
        return jsonify(movies)
    except Exception as e:
        print(f"Error in get_movies: {e}")
        return jsonify({'error': 'Failed to fetch movies'}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_movies():
    """Generate recommendations based on selected movie IDs."""
    try:
        data = request.json
        selected_movie_ids = data.get('selected_movies', [])
        if len(selected_movie_ids) < 5:
            return jsonify({'error': 'Please select at least 5 movies'}), 400
        recommendations = recommender.get_recommendations(selected_movie_ids)
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in recommend_movies: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check the health of the API and recommender system."""
    return jsonify({
        "status": "healthy",
        "movies_loaded": len(recommender.movies_df) if recommender.movies_df is not None else 0,
        "similarity_matrix_built": recommender.similarity_matrix is not None
    })

# --- Main Execution ---
if __name__ == "__main__":
    # Start Flask server
    print("🚀 Starting Flask backend server...")
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
