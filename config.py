# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- OMDb API Configuration ---
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
if not OMDB_API_KEY:
    raise ValueError("OMDB_API_KEY not found in environment variables.")

# --- Supabase Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables.")

# --- JWT Configuration ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# --- File Paths ---
DB_PATH = "movies.db"  # Legacy - keeping for reference, now using Supabase
ASSETS_PATH = "assets"

# --- API Settings ---
OMDB_API_URL = "http://www.omdbapi.com/"
API_TIMEOUT_SECONDS = 10

# --- Supabase Table Names ---
MOVIES_TABLE = "movies"
USERS_TABLE = "users"
USER_RATINGS_TABLE = "user_ratings"
MODELS_TABLE = "movies"  # Using movies table for model training

# --- Supabase Storage Configuration ---
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "trained_models")  # Your Supabase Storage bucket name

# --- Model Training Configuration ---
MODEL_UPDATE_DELAY = 300  # 5 minutes in seconds (time to wait after last movie addition before retraining)
MIN_MOVIES_FOR_RETRAIN = 5  # Minimum new movies before considering retraining
MAX_RETRAIN_INTERVAL = 86400  # 24 hours in seconds (force retrain after this time)

# --- Model Weights ---
CONTENT_WEIGHT = 0.8
COLLAB_WEIGHT = 0.2 
#collab model is how other users liked movies #before the current user. as userbase #increase we increase collab weight
