# db_population.py
import sqlite3
import requests
import time
import os
import logging

# --- Configuration ---
DB_FILE = "movies_updated.db"
TITLES_FILE = "movie_titles.txt"
PROGRESS_FILE = "progress.txt"
BATCH_SIZE = 10  # How many movies to process before saving progress
OMDB_API_KEY = os.getenv("OMDB_API_KEY") # IMPORTANT: Set this as an environment variable

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- State Management Functions ---
def save_progress(index):
    """Saves the index of the last successfully processed movie."""
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(index))
    logging.info(f"Progress saved. Processed up to movie number {index}.")

def load_progress():
    """Loads the index to resume from. Returns 0 if no progress file exists."""
    if not os.path.exists(PROGRESS_FILE):
        return 0
    try:
        with open(PROGRESS_FILE, "r") as f:
            content = f.read().strip()
            if content:
                return int(content)
            return 0
    except (IOError, ValueError):
        return 0

# --- Database and API Functions ---
def setup_database():
    """Creates the database and the movies table if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
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
    """)
    conn.commit()
    conn.close()
    logging.info(f"Database '{DB_FILE}' is set up and ready.")

def fetch_movie_data(title):
    """Fetches movie data from OMDb by title."""
    url = f"http://www.omdbapi.com/?t={title.replace(' ', '+')}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("Response") == "True":
            # Create a combined features string for the recommendation model
            combined_features = (
                f"{data.get('Genre', '')} "
                f"{data.get('Actors', '')} "
                f"{data.get('Director', '')} "
                f"{data.get('Plot', '')}"
            )
            
            # Map API response to our database schema
            return {
                "id": data.get("imdbID"),
                "title": data.get("Title"),
                "overview": data.get("Plot"),
                "genres": data.get("Genre"),
                "cast": data.get("Actors"),
                "director": data.get("Director"),
                "poster_path": data.get("Poster"),
                "vote_average": float(data.get("imdbRating", 0)) if data.get("imdbRating") != "N/A" else 0,
                "release_date": data.get("Year"),
                "combined_features": combined_features.strip().replace("  ", " ")
            }
        else:
            logging.warning(f"Could not find movie '{title}': {data.get('Error')}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for '{title}': {e}")
        return None

# --- Main Logic ---
def populate_database():
    """Main function to orchestrate the database population."""
    if not OMDB_API_KEY:
        logging.error("OMDB_API_KEY environment variable not set. Exiting.")
        return

    setup_database()

    try:
        with open(TITLES_FILE, "r") as f:
            movie_titles = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"'{TITLES_FILE}' not found. Please create it and add movie titles.")
        return

    start_index = load_progress()
    logging.info(f"Starting process. Resuming from movie number {start_index}.")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for i, title in enumerate(movie_titles):
        if i < start_index:
            continue  # Skip already processed movies

        logging.info(f"Processing movie {i + 1}/{len(movie_titles)}: '{title}'")
        movie_data = fetch_movie_data(title)

        if movie_data and movie_data.get("id"):
            # Prepare data for insertion
            columns = ', '.join(movie_data.keys())
            placeholders = ', '.join('?' * len(movie_data))
            sql = f"INSERT OR IGNORE INTO movies ({columns}) VALUES ({placeholders})"
            values = tuple(movie_data.values())
            
            cursor.execute(sql, values)
            conn.commit() # Commit after each successful insert to save progress
        
        # Save progress every BATCH_SIZE movies
        if (i + 1) % BATCH_SIZE == 0:
            save_progress(i + 1)

        time.sleep(0.2) # Be polite to the API

    conn.close()
    save_progress(len(movie_titles)) # Final save
    logging.info("Database population complete!")


if __name__ == "__main__":
    populate_database()
