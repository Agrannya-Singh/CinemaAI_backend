import os
import logging
import requests
from typing import Optional
from supabase import create_client
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseStorage:
    def __init__(self):
        """Initialize Supabase storage client."""
        self.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_ANON_KEY)
        self.bucket = config.STORAGE_BUCKET
        self.public_url = f"{config.SUPABASE_URL}/storage/v1/object/public/{self.bucket}"

    def upload_file(self, file_path: str, object_name: str = None) -> bool:
        """
        Upload a file to the public bucket.
        
        Args:
            file_path: Path to the local file
            object_name: Name to give the file in storage (defaults to filename)
            
        Returns:
            bool: True if upload was successful
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            if object_name is None:
                object_name = os.path.basename(file_path)

            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Upload with overwrite if exists
            response = self.supabase.storage. \
                from_(self.bucket). \
                upload(object_name, file_data, {'x-upsert': 'true'})
            
            if response.get('error'):
                raise Exception(response['error']['message'])
            
            logger.info(f"Uploaded {object_name} to {self.bucket}")
            return True
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False

    def download_file(self, object_name: str, local_path: str) -> bool:
        """
        Download a file from the public bucket.
        
        Args:
            object_name: Name of the file in storage
            local_path: Local path to save the file
            
        Returns:
            bool: True if download was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Direct download from public URL
            url = f"{self.public_url}/{object_name}"
            response = requests.get(url)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded {object_name} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False

    def get_public_url(self, object_name: str) -> str:
        """
        Get the public URL for a file in the bucket.
        
        Args:
            object_name: Name of the file in storage
            
        Returns:
            str: Public URL of the file
        """
        return f"{self.public_url}/{object_name}"

# Singleton instance for easy importing
storage = SupabaseStorage()