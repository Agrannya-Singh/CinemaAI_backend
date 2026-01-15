
# CinemaAI Backend

check repo discription for the hosted links.$$
Movie recommendation system backend implementation using FastAPI framework with machine learning-based recommendation engine.

## Overview

This application provides REST API endpoints for movie data management and intelligent recommendations. The system uses cosine similarity algorithms to analyze movie metadata including genres, cast, and plot summaries to generate personalized recommendations.

## System Requirements

- Python 3.8 or higher
- OMDb API key (registration required at omdbapi.com)
- 512MB RAM minimum
- 100MB disk space

## Dependencies

Core dependencies as specified in requirements.txt:
- fastapi: Web framework for API implementation
- uvicorn: ASGI server for FastAPI applications
- sqlite3: Database engine (included in Python standard library)
- pandas: Data manipulation and analysis
- scikit-learn: Machine learning library for cosine similarity calculations
- numpy: Numerical computing support
- requests: HTTP library for external API calls
- python-dotenv: Environment variable management

## Installation

1. Clone repository:
```bash
git clone https://github.com/Agrannya-Singh/CinemaAI_backend.git
cd CinemaAI_backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
echo "OMDB_API_KEY=your_api_key_here" > .env
```

4. Start application:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## File Structure

```
CinemaAI_backend/
├── main.py                 # Primary application module
├── requirements.txt        # Python package dependencies
├── .env                   # Environment configuration (user-created)
├── movies.db              # SQLite database file (auto-generated)
└── similarity_matrix.pkl  # Cached similarity computations (auto-generated)
```

## Database Schema

SQLite database with single table structure:

```sql
CREATE TABLE IF NOT EXISTS movies (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    overview TEXT,
    genres TEXT,
    cast TEXT,
    poster_path TEXT,
    vote_average REAL,
    release_date TEXT
);
```

## API Contract

Base URL: https://cinemaai-backend.onrender.com

### GET /movies

Returns all movies in database.

**Request:**
- Method: GET
- Headers: None required
- Body: None

**Response:**
- Status: 200 OK
- Content-Type: application/json
- Body: Array of movie objects

**Movie Object Schema:**
```json
{
  "id": "string",
  "title": "string", 
  "overview": "string",
  "genres": "string",
  "cast": "string",
  "poster_path": "string",
  "vote_average": "number",
  "release_date": "string"
}
```

**Error Responses:**
- 500: Internal server error, no movie data available

### POST /recommend

Generate movie recommendations based on input movie IDs.

**Request:**
- Method: POST
- Headers: Content-Type: application/json
- Body:
```json
{
  "movie_ids": ["string", "string"],
  "num_recommendations": "integer"
}
```

**Request Parameters:**
- movie_ids: Array of valid movie ID strings (required)
- num_recommendations: Integer between 1-50, default 10 (optional)

**Response:**
- Status: 200 OK
- Content-Type: application/json
- Body: Array of recommended movie objects (same schema as GET /movies)

**Error Responses:**
- 400: Bad request, invalid input format
- 404: Movie IDs not found in database
- 500: Internal server error, recommendation engine failure

### GET /search/{title}

Search for movie by title, add to database if not exists.

**Request:**
- Method: GET
- Path Parameter: title (string, URL-encoded)
- Headers: None required
- Body: None

**Response:**
- Status: 200 OK
- Content-Type: application/json
- Body: Array containing single movie object

**Error Responses:**
- 404: Movie not found in OMDb API
- 500: Internal server error, API key invalid or external service unavailable

## Configuration

### Environment Variables

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| OMDB_API_KEY | string | yes | API key for OMDb service |

### CORS Settings

Default configuration allows requests from:
- http://localhost:3000
- Methods: GET, POST, PUT, DELETE, OPTIONS
- Headers: All headers permitted
- Credentials: Enabled

## Default Data

System initializes with following movie entries:
- Inception (tt1375666)
- The Dark Knight (tt0468569) 
- Interstellar (tt0816692)
- The Matrix (tt0133093)
- Pulp Fiction (tt0110912)
- Forrest Gump (tt0109830)
- The Shawshank Redemption (tt0111161)
- Gladiator (tt0172495)
- Titanic (tt0120338)
- Avatar (tt0499549)

## Recommendation Algorithm

## Recommendation Algorithm

Implementation uses **Sentence Transformers** for robust semantic similarity:

1.  **Semantic Embedding**: Uses the `all-MiniLM-L6-v2` Sentence Transformer model to generate dense vector embeddings for movie metadata (combined features: overview, genres, cast).
2.  **Context Awareness**: Unlike TF-IDF, this model captures deep semantic meaning and context (e.g., understanding that "scary" is semantically similar to "horror").
3.  **Cosine Similarity**: Calculates the cosine similarity matrix between these dense embeddings.
4.  **Nearest Neighbors**: Selects the closest matches based on semantic distance.

*Note: The system also includes a backup collaborative filtering model using Nearest Neighbors on user ratings, if sufficient data exists.*

## Operational Considerations

### Performance
- Similarity matrix cached as pickle file for improved response times
- Background task processing for matrix computation
- Database queries optimized for single-table operations

### Scaling
- SQLite suitable for development and small deployments
-  PostgreSQL for production environments(supabase)
- Implement connection pooling for concurrent requests

### Security
- API key stored in environment variables
- No authentication implemented (add as needed)
- Input validation via Pydantic models
- CORS configured for specific origins

## Error Handling

Standard HTTP status codes:
- 200: Successful operation
- 400: Client error, invalid request format
- 404: Resource not found
- 500: Server error, check logs

## Logging

Application uses Python logging module. Log levels:
- INFO: Application startup, database operations
- WARNING: API request failures, external service issues
- ERROR: Critical failures, exception handling

## Health Monitoring

No built-in health check endpoint. Monitor via:
- HTTP response codes
- Application logs  
- Database file presence
- Process status

## Development

Run in development mode:
```bash
uvicorn main:app --reload --log-level debug
```

## Testing

Manual testing via curl:

```bash
# Get all movies
curl http://localhost:8000/movies

# Get recommendations  
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"movie_ids":["tt1375666"],"num_recommendations":5}'

# Search movie
curl http://localhost:8000/search/Matrix
```

## License

Open source project. fork it ; mold it ; its all yours. [MIT License]

## Support

Issues tracked via GitHub repository issue tracker.
