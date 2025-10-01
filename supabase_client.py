# supabase_client.py

import logging
import os
from typing import Optional, Tuple
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

def upload_to_storage(bucket_name: str, file_path: str, file_data: bytes) -> bool:
    """Upload a file to Supabase Storage."""
    try:
        supabase = get_supabase_client()
        # Create bucket if it doesn't exist
        try:
            supabase.storage.create_bucket(bucket_name, public=True)
        except Exception as e:
            if 'Bucket already exists' not in str(e):
                raise
        
        # Upload file
        supabase.storage.from_(bucket_name).upload(file_path, file_data)
        logging.info(f"Uploaded {file_path} to {bucket_name} bucket")
        return True
    except Exception as e:
        logging.error(f"Error uploading to Supabase Storage: {e}")
        return False

def download_from_storage(bucket_name: str, file_path: str) -> Optional[bytes]:
    """Download a file from Supabase Storage."""
    try:
        supabase = get_supabase_client()
        response = supabase.storage.from_(bucket_name).download(file_path)
        logging.info(f"Downloaded {file_path} from {bucket_name} bucket")
        return response
    except Exception as e:
        if 'The resource was not found' not in str(e):
            logging.error(f"Error downloading from Supabase Storage: {e}")
        return None

def ensure_local_dir(path: str) -> None:
    """Ensure local directory exists."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
def save_model_with_storage(local_path: str, bucket_name: str = "models") -> bool:
    """Save a model both locally and to Supabase Storage."""
    try:
        # Load local file
        with open(local_path, 'rb') as f:
            file_data = f.read()
        
        # Upload to Supabase
        return upload_to_storage(bucket_name, os.path.basename(local_path), file_data)
    except Exception as e:
        logging.error(f"Error saving model to storage: {e}")
        return False

def load_model_with_storage(local_path: str, bucket_name: str = "models") -> Optional[bytes]:
    """Load a model from Supabase Storage if not available locally."""
    # Try to load from local storage first
    if os.path.exists(local_path):
        try:
            with open(local_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logging.warning(f"Error reading local model {local_path}: {e}")
    
    # If not found locally, try to download from Supabase
    file_data = download_from_storage(bucket_name, os.path.basename(local_path))
    if file_data:
        try:
            ensure_local_dir(local_path)
            with open(local_path, 'wb') as f:
                f.write(file_data)
            return file_data
        except Exception as e:
            logging.error(f"Error saving downloaded model locally: {e}")
    
    return None
