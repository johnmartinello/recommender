import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import torch


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3

class LocalModelSettings(LLMSettings):
    """Settings for local embedding model configurations."""
    
    path_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Path to local model or HuggingFace model identifier"
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run model on (cuda/cpu)"
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    type_model: str = Field(
        default="sentence-transformer",
        description="Type of model (sentence-transformer, bert, etc)"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory to cache downloaded models"
    )


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings"
    embedding_dimensions: int = 768
    time_partition_interval: timedelta = timedelta(days=7)
    

class Settings(BaseModel):
    local: LocalModelSettings = Field(default_factory=LocalModelSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)

@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    load_dotenv(dotenv_path="./.env")
    settings = Settings()
    setup_logging()
    return settings
