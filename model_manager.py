"""Model manager for Orpheus TTS with vLLM backend."""
import logging
import threading
from typing import Iterator, Optional
from orpheus_tts import OrpheusModel
from config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton model manager for efficient GPU utilization."""
    
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the model (only once)."""
        if self._initialized:
            return
            
        logger.info(f"Initializing Orpheus model: {settings.model_name}")
        
        try:
            # Initialize OrpheusModel
            # Note: vLLM-specific parameters (max_model_len, tensor_parallel_size, etc.)
            # are not directly supported by OrpheusModel - they may be configured
            # via environment variables or the library's internal configuration
            self.model = OrpheusModel(
                model_name=settings.model_name,
            )
            
            logger.info("Model initialized successfully")
            logger.info(f"Max concurrent sequences: {settings.max_num_seqs}")
            logger.info(f"GPU memory utilization: {settings.gpu_memory_utilization}")
            
            # Lock for serializing generation requests (orpheus_tts uses hardcoded request_id)
            self._generation_lock = threading.Lock()
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def generate_speech_stream(
        self,
        prompt: str,
        voice: str = "tara",
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Iterator[bytes]:
        """
        Generate speech audio in streaming fashion.
        
        Args:
            prompt: Text to convert to speech
            voice: Voice name (tara, leah, jess, leo, dan, mia, zac, zoe)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty (must be >= 1.1)
            
        Yields:
            Audio chunks as bytes (16-bit PCM at 24kHz)
        """
        try:
            # Ensure repetition penalty is valid
            if repetition_penalty < 1.1:
                logger.warning(f"Repetition penalty {repetition_penalty} < 1.1, setting to 1.1")
                repetition_penalty = 1.1
            
            logger.info(f"Generating speech for prompt (length: {len(prompt)})")
            logger.debug(f"Voice: {voice}, temp: {temperature}, top_p: {top_p}, rep_penalty: {repetition_penalty}")
            
            # Acquire lock to prevent concurrent requests (orpheus_tts uses hardcoded request_id)
            # This serializes requests but ensures stability
            with self._generation_lock:
                # Generate speech tokens (streaming)
                syn_tokens = self.model.generate_speech(
                    prompt=prompt,
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                
                # Collect all chunks while holding the lock
                # We must complete generation before releasing the lock
                audio_chunks = list(syn_tokens)
            
            # Stream audio chunks (lock released, safe to yield)
            chunk_count = 0
            for audio_chunk in audio_chunks:
                chunk_count += 1
                yield audio_chunk
            
            logger.info(f"Generated {chunk_count} audio chunks")
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if model is initialized and ready."""
        return self._initialized and self.model is not None


# Global model manager instance (lazy initialization to avoid multiprocessing issues)
_model_manager: Optional[ModelManager] = None
_init_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """Get or create the global model manager instance (lazy initialization)."""
    global _model_manager
    if _model_manager is None:
        with _init_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    return _model_manager


# For backwards compatibility - but only when accessed in main process
class _LazyModelManager:
    """Lazy wrapper that defers initialization until first attribute access."""
    
    def __getattr__(self, name):
        return getattr(get_model_manager(), name)


model_manager = _LazyModelManager()
