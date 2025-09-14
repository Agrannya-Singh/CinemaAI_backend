# config.py

import os
from dotenv import load_dotenv

load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")
if not OMDB_API_KEY:
    raise ValueError("OMDB_API_KEY not found in environment variables.")

# --- File Paths ---
DB_PATH = "movies.db"
ASSETS_PATH = "assets"

# --- API Settings ---
OMDB_API_URL = "http://www.omdbapi.com/"
API_TIMEOUT_SECONDS = 10

# --- Model Weights ---
CONTENT_WEIGHT = 0.5
COLLAB_WEIGHT = 0.5
