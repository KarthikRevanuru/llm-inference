"""Configuration management for Orpheus TTS deployment."""
import configparser
import os
from pathlib import Path
from typing import List


class Settings:
    """Application settings loaded from config.ini."""
    
    def __init__(self, config_path: str = None):
        """Load settings from config.ini file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.ini"
        
        self._config = configparser.ConfigParser()
        
        # Load config file if it exists
        if os.path.exists(config_path):
            self._config.read(config_path)
        
        # Model Configuration
        self.model_name: str = self._get("model", "name", "canopylabs/orpheus-3b-0.1-ft")
        self.max_model_len: int = self._getint("model", "max_model_len", 2048)
        self.tensor_parallel_size: int = self._getint("model", "tensor_parallel_size", 1)
        self.gpu_memory_utilization: float = self._getfloat("model", "gpu_memory_utilization", 0.80)
        
        # vLLM Batching Configuration
        self.max_num_seqs: int = self._getint("batching", "max_num_seqs", 4)
        self.max_batch_size: int = self._getint("batching", "max_batch_size", 4)
        
        # Server Configuration
        self.host: str = self._get("server", "host", "0.0.0.0")
        self.port: int = self._getint("server", "port", 8000)
        self.workers: int = self._getint("server", "workers", 1)
        self.log_level: str = self._get("server", "log_level", "info")
        
        # Audio Configuration
        self.sample_rate: int = self._getint("audio", "sample_rate", 24000)
        self.channels: int = self._getint("audio", "channels", 1)
        self.sample_width: int = self._getint("audio", "sample_width", 2)
        self.default_voice: str = self._get("audio", "default_voice", "tara")
        
        # Available voices for the finetuned model
        self.available_voices: List[str] = [
            "tara", "leah", "jess", "leo", 
            "dan", "mia", "zac", "zoe"
        ]
        
        # Generation parameters
        self.default_temperature: float = self._getfloat("generation", "default_temperature", 1.0)
        self.default_top_p: float = self._getfloat("generation", "default_top_p", 0.95)
        self.default_repetition_penalty: float = self._getfloat("generation", "default_repetition_penalty", 1.1)
        
        # Request limits
        self.max_prompt_length: int = self._getint("limits", "max_prompt_length", 1000)
        self.request_timeout: int = self._getint("limits", "request_timeout", 60)
    
    def _get(self, section: str, key: str, default: str) -> str:
        """Get string value with fallback to default."""
        try:
            return self._config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def _getint(self, section: str, key: str, default: int) -> int:
        """Get integer value with fallback to default."""
        try:
            return self._config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def _getfloat(self, section: str, key: str, default: float) -> float:
        """Get float value with fallback to default."""
        try:
            return self._config.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default


# Global settings instance
settings = Settings()
