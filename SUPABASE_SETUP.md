# Supabase Setup Guide

This guide will help you set up your Supabase database for the CinemaAI backend.

## Prerequisites

- Active Supabase account
- Supabase project created

## Step 1: Database Schema Setup

Run the following SQL commands in your Supabase SQL Editor to create the necessary tables:

### 1. Movies Table

```sql
-- Create movies table
CREATE TABLE IF NOT EXISTS movies (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    overview TEXT,
    genres TEXT,
    director TEXT,
    cast TEXT,
    poster_path TEXT,
    vote_average FLOAT,
    release_date TEXT,
    combined_features TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster searches
CREATE INDEX IF NOT EXISTS idx_movies_title ON movies(title);
CREATE INDEX IF NOT EXISTS idx_movies_created_at ON movies(created_at);
```

### 2. User Ratings Table (Optional - for future collaborative filtering)

```sql
-- Create user_ratings table for collaborative filtering
CREATE TABLE IF NOT EXISTS user_ratings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    movie_id TEXT REFERENCES movies(id) ON DELETE CASCADE,
    rating FLOAT CHECK (rating >= 0 AND rating <= 10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, movie_id)
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_user_ratings_user_id ON user_ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_user_ratings_movie_id ON user_ratings(movie_id);
```

### 3. Enable Row Level Security (RLS)

```sql
-- Enable RLS on movies table
ALTER TABLE movies ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to read movies
CREATE POLICY "Allow authenticated users to read movies"
ON movies FOR SELECT
TO authenticated
USING (true);

-- Allow authenticated users to insert movies (for search feature)
CREATE POLICY "Allow authenticated users to insert movies"
ON movies FOR INSERT
TO authenticated
WITH CHECK (true);

-- Enable RLS on user_ratings table
ALTER TABLE user_ratings ENABLE ROW LEVEL SECURITY;

-- Allow users to read all ratings
CREATE POLICY "Allow authenticated users to read ratings"
ON user_ratings FOR SELECT
TO authenticated
USING (true);

-- Allow users to insert their own ratings
CREATE POLICY "Allow users to insert their own ratings"
ON user_ratings FOR INSERT
TO authenticated
WITH CHECK (auth.uid() = user_id);

-- Allow users to update their own ratings
CREATE POLICY "Allow users to update their own ratings"
ON user_ratings FOR UPDATE
TO authenticated
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

-- Allow users to delete their own ratings
CREATE POLICY "Allow users to delete their own ratings"
ON user_ratings FOR DELETE
TO authenticated
USING (auth.uid() = user_id);
```

## Step 2: Migrate Existing SQLite Data (Optional)

If you have existing data in your `movies.db` SQLite database, you can migrate it to Supabase:

### Option A: Using Python Script

Create a migration script `migrate_to_supabase.py`:

```python
import sqlite3
import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Use service role key for admin operations
)

# Connect to SQLite database
conn = sqlite3.connect('movies.db')

# Read movies from SQLite
movies_df = pd.read_sql_query("SELECT * FROM movies", conn)
conn.close()

# Convert DataFrame to list of dictionaries
movies_data = movies_df.to_dict('records')

# Insert into Supabase in batches
batch_size = 100
for i in range(0, len(movies_data), batch_size):
    batch = movies_data[i:i+batch_size]
    try:
        response = supabase.table('movies').upsert(batch, on_conflict='id').execute()
        print(f"Inserted batch {i//batch_size + 1}: {len(batch)} movies")
    except Exception as e:
        print(f"Error inserting batch {i//batch_size + 1}: {e}")

print("Migration completed!")
```

### Option B: Export to CSV and Import

1. Export from SQLite:
```bash
sqlite3 movies.db
.headers on
.mode csv
.output movies.csv
SELECT * FROM movies;
.quit
```

2. Import to Supabase using the Supabase Dashboard:
   - Go to Table Editor
   - Select the `movies` table
   - Click "Insert" → "Import data from CSV"
   - Upload your `movies.csv` file

## Step 3: Configure Environment Variables

1. Copy `.env.sample` to `.env`:
```bash
cp .env.sample .env
```

2. Fill in your Supabase credentials in `.env`:
   - **SUPABASE_URL**: Found in Project Settings → API → Project URL
   - **SUPABASE_ANON_KEY**: Found in Project Settings → API → Project API keys → anon/public
   - **SUPABASE_SERVICE_ROLE_KEY**: Found in Project Settings → API → Project API keys → service_role (keep this secret!)

3. Generate a secure JWT secret:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
   - Use this value for **JWT_SECRET_KEY**

## Step 4: Enable Supabase Auth

Your Supabase project should have Auth enabled by default. Configure it:

1. Go to Authentication → Settings
2. Configure email settings:
   - Enable "Enable email confirmations" if you want email verification
   - Configure SMTP settings or use Supabase's default email service

3. (Optional) Configure OAuth providers:
   - Google, GitHub, etc. can be enabled in Authentication → Providers

## Step 5: Test the Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
uvicorn main:app --reload
```

3. Test authentication endpoints:

**Sign Up:**
```bash
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "securepassword123", "full_name": "Test User"}'
```

**Login:**
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "securepassword123"}'
```

**Get Current User:**
```bash
curl -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Get Movies (Protected):**
```bash
curl -X GET "http://localhost:8000/movies" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## API Documentation

Once your server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Security Best Practices

1. **Never commit `.env` file** - It contains sensitive credentials
2. **Use SUPABASE_SERVICE_ROLE_KEY carefully** - Only use it for admin operations, never expose it to clients
3. **Restrict CORS origins** - Update `allow_origins` in `main.py` to your frontend URL
4. **Use HTTPS in production** - Never send tokens over HTTP
5. **Rotate JWT secrets regularly** - Update JWT_SECRET_KEY periodically
6. **Enable email confirmation** - Verify user emails before allowing access
7. **Set up rate limiting** - Protect your API from abuse

## Troubleshooting

### Issue: "Could not validate credentials"
- Check if JWT_SECRET_KEY matches between signup and login
- Verify token is being sent in Authorization header as "Bearer TOKEN"

### Issue: "Supabase client is not initialized"
- Verify SUPABASE_URL and SUPABASE_ANON_KEY are set in .env
- Check if .env file is in the project root directory

### Issue: "Row Level Security" errors
- Make sure RLS policies are created correctly
- Verify user is authenticated before accessing protected resources

### Issue: Migration fails
- Check if SUPABASE_SERVICE_ROLE_KEY is set correctly
- Verify table schema matches between SQLite and Supabase
- Try smaller batch sizes if getting timeout errors

## Next Steps

1. **Implement user ratings** - Allow users to rate movies for better recommendations
2. **Add user preferences** - Store user preferences for personalized recommendations
3. **Implement caching** - Use Redis or similar for frequently accessed data
4. **Add analytics** - Track user behavior for improving recommendations
5. **Set up monitoring** - Use Supabase logs and monitoring tools

## Support

For more information:
- [Supabase Documentation](https://supabase.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Supabase Python Client](https://github.com/supabase-community/supabase-py)
