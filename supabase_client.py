# supabase_client.py

import logging
from supabase import create_client, Client
import config

# Initialize Supabase client
try:
    supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_ANON_KEY)
    logging.info("Supabase client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Supabase client: {e}")
    supabase = None

def get_supabase_client() -> Client:
    """Returns the Supabase client instance."""
    if supabase is None:
        raise Exception("Supabase client is not initialized.")
    return supabase
