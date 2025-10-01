# Backend API Endpoints

This document outlines the API endpoints that the CinemaAI frontend application expects from the backend service.

**Base URL:** `https://cinemaai-backend.onrender.com`

---

### 1. Fetch All Movies

- **Endpoint:** `GET /movies`
- **Description:** Retrieves a comprehensive list of all movies available in the database.
- **Success Response:**
  - **Code:** 200 OK
  - **Content:** An array of movie objects.

- **Example Movie Object:**
  ```json
  {
    "id": "tt0111161",
    "title": "The Shawshank Redemption",
    "year": "1994",
    "director": "Frank Darabont",
    "genre": "Drama",
    "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
    "poster": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_SX300.jpg"
  }
  ```

### 2. Search for a Movie

- **Endpoint:** `GET /search/{identifier}`
- **Description:** Searches for a movie using either its IMDb ID or title.
- **URL Parameters:**
  - `identifier` (string, required): The IMDb ID or title of the movie to search for.
- **Success Response:**
  - **Code:** 200 OK
  - **Content:** An array containing the movie object that matches the identifier. If no movie is found, it returns an empty array.
- **Error Response:**
  - **Code:** 404 Not Found
  - **Content:** `{"detail":"Movie not found"}`

### 3. Get Movie Recommendations

- **Endpoint:** `POST /recommend`
- **Description:** Generates personalized movie recommendations based on a list of user-selected movie IDs.
- **Request Body:**
  ```json
  {
    "movie_ids": ["tt0111161", "tt0068646"],
    "num_recommendations": 10
  }
  ```
- **Success Response:**
  - **Code:** 200 OK
  - **Content:** An array of recommended movie objects, with the same structure as in the `/movies` endpoint response.
- **Error Response:**
  - **Code:** 422 Unprocessable Entity (if request body is invalid)

### 4. Check Model Loading Status

- **Endpoint:** `GET /models/status`
- **Description:** Returns the loading status of the recommendation models. This is useful for deployment health checks to confirm if the application is ready to serve recommendation requests after starting up. The models are loaded in the background to ensure a quick startup time.
- **Success Response:**
  - **Code:** 200 OK
  - **Content:** A JSON object indicating whether the models are loaded.
  - **Example (Models Loaded):**
    ```json
    {
      "models_loaded": true
    }
    ```
  - **Example (Models Still Loading):**
    ```json
    {
      "models_loaded": false
    }
    ```
