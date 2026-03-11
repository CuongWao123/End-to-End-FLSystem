"""MinIO client for model storage."""

from minio import Minio
from minio.error import S3Error
import io
import torch
import os

from  core.config import settings


class MinIOClient:
    """Client for interacting with MinIO storage."""

    def __init__(
        self,
        bucket_name: str = "fl-models",
    ):
        """Initialize MinIO client."""
        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
        )
        self.bucket_name = bucket_name
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"Created bucket: {self.bucket_name}")
        except S3Error as e:
            print(f"Error creating bucket: {e}")

    def save_model(self, state_dict: dict, object_name: str):
        """Save PyTorch model to MinIO."""
        try:
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            buffer.seek(0)
            
            self.client.put_object(
                self.bucket_name,
                object_name,
                buffer,
                length=buffer.getbuffer().nbytes,
                content_type="application/octet-stream",
            )
            print(f"✓ Model saved to MinIO: {self.bucket_name}/{object_name}")
            
        except S3Error as e:
            print(f"Error saving model to MinIO: {e}")

    def load_model(self, object_name: str):
        """Load PyTorch model from MinIO."""
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            buffer = io.BytesIO(response.read())
            state_dict = torch.load(buffer)
            print(f"✓ Model loaded from MinIO: {self.bucket_name}/{object_name}")
            return state_dict
        except S3Error as e:
            print(f"Error loading model from MinIO: {e}")
            return None