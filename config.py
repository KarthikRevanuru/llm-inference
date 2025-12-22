"""Configuration management for Orpheus TTS deployment."""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Model Configuration
    model_name: str = "canopylabs/orpheus-3b-0.1-ft"
    max_model_len: int = 2048
    tensor_parallel_size: int = 1  # Set to number of GPUs for multi-GPU
    gpu_memory_utilization: float = 0.90  # Use 90% of GPU memory
    
    # vLLM Batching Configuration
    max_num_seqs: int = 8  # Maximum concurrent sequences
    max_batch_size: int = 8  # Maximum batch size for continuous batching
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1  # Keep at 1 for GPU models
    log_level: str = "info"
    
    # Audio Configuration
    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2  # 16-bit audio
    
    # Available voices for the finetuned model
    available_voices: List[str] = [
        "tara", "leah", "jess", "leo", 
        "dan", "mia", "zac", "zoe"
    ]
    default_voice: str = "tara"
    
    # Generation parameters
    default_temperature: float = 1.0
    default_top_p: float = 0.95
    default_repetition_penalty: float = 1.1  # Required for stable generation
    
    # Request limits
    max_prompt_length: int = 1000  # Characters
    request_timeout: int = 60  # Seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
