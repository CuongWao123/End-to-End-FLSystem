from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_DIR: str = "./quickstart-pytorch"
    FLWR_VERSION: str = "1.26.1"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin123"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()