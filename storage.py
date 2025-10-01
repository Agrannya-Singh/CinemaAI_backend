import os
import logging
from supabase import create_client
from config import SUPABASE_URL, SUPABASE_ANON_KEY, STORAGE_BUCKET

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseStorage:
    def __init__(self):
        """Initialize Supabase client for storage operations."""
        self.supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        self.bucket = STORAGE_BUCKET
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Ensure the storage bucket exists."""
        try:
            self.supabase.storage.get_bucket(self.bucket)
        except Exception as e:
            if "Bucket not found" in str(e):
                logger.info(f"Creating bucket: {self.bucket}")
                self.supabase.storage.create_bucket(
                    self.bucket,
                    public=True,  # Set to False for private buckets
                    file_size_limit=100  # MB
                )
            else:
                logger.error(f"Error checking bucket: {e}")
                raise

    def upload_file(self, file_path: str, object_name: str = None) -> bool:
        """
        Upload a file to Supabase Storage.
        
        Args:
            file_path: Path to the local file
            object_name: Name to give the file in storage (defaults to filename)
            
        Returns:
            bool: True if upload was successful
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
                
            if object_name is None:
                object_name = os.path.basename(file_path)
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            self.supabase.storage.\
                from_(self.bucket).\
                upload(object_name, file_data)
            
            logger.info(f"Uploaded {file_path} to {self.bucket}/{object_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading {file_path}: {e}")
            return False

    def download_file(self, object_name: str, file_path: str) -> bool:
        """
        Download a file from Supabase Storage.
        
        Args:
            object_name: Name of the file in storage
            file_path: Local path to save the file
            
        Returns:
            bool: True if download was successful
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            response = self.supabase.storage.\
                from_(self.bucket).\
                download(object_name)
            
            with open(file_path, 'wb') as f:
                f.write(response)
            
            logger.info(f"Downloaded {object_name} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {object_name}: {e}")
            return False

    def get_public_url(self, object_name: str) -> str:
        """
        Get a public URL for a file in storage.
        
        Args:
            object_name: Name of the file in storage
            
        Returns:
            str: Public URL or empty string if error
        """
        try:
            return self.supabase.storage.\
                from_(self.bucket).\
                get_public_url(object_name)
        except Exception as e:
            logger.error(f"Error getting URL for {object_name}: {e}")
            return ""

# Singleton instance
storage = SupabaseStorage()
