import os
import boto3
import logging
from botocore.exceptions import ClientError
from typing import Optional, BinaryIO
from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    S3_BUCKET_NAME
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Storage:
    def __init__(self):
        """Initialize S3 client with credentials from environment variables."""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        self.bucket_name = S3_BUCKET_NAME

    def upload_file(self, file_path: str, object_name: str = None) -> bool:
        """
        Upload a file to the S3 bucket.
        
        Args:
            file_path: Path to the file to upload
            object_name: S3 object name. If not specified, file_path is used
            
        Returns:
            bool: True if file was uploaded, else False
        """
        if object_name is None:
            object_name = os.path.basename(file_path)
            
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            logger.info(f"Successfully uploaded {file_path} to {self.bucket_name}/{object_name}")
            return True
        except ClientError as e:
            logger.error(f"Error uploading {file_path} to S3: {e}")
            return False

    def download_file(self, object_name: str, file_path: str) -> bool:
        """
        Download a file from the S3 bucket.
        
        Args:
            object_name: S3 object name
            file_path: Path to save the downloaded file
            
        Returns:
            bool: True if file was downloaded, else False
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, object_name, file_path)
            logger.info(f"Successfully downloaded {object_name} to {file_path}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.warning(f"The object {object_name} does not exist in {self.bucket_name}.")
            else:
                logger.error(f"Error downloading {object_name} from S3: {e}")
            return False

    def get_file_url(self, object_name: str, expires_in: int = 3600) -> Optional[str]:
        """
        Generate a pre-signed URL to share an S3 object.
        
        Args:
            object_name: S3 object name
            expires_in: Time in seconds for the URL to remain valid
            
        Returns:
            str: Pre-signed URL as a string, or None if error
        """
        try:
            response = self.s3_client.generate_presigned_url('get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_name},
                ExpiresIn=expires_in
            )
            logger.info(f"Generated pre-signed URL for {object_name}")
            return response
        except ClientError as e:
            logger.error(f"Error generating pre-signed URL: {e}")
            return None

# Singleton instance
s3_storage = S3Storage()
