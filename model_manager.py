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
            # Initialize OrpheusModel with vLLM backend
            self.model = OrpheusModel(
                model_name=settings.model_name,
                max_model_len=settings.max_model_len,
                tensor_parallel_size=settings.tensor_parallel_size,
                gpu_memory_utilization=settings.gpu_memory_utilization,
                max_num_seqs=settings.max_num_seqs,
            )
            
            logger.info("Model initialized successfully")
            logger.info(f"Max concurrent sequences: {settings.max_num_seqs}")
            logger.info(f"GPU memory utilization: {settings.gpu_memory_utilization}")
            
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
            
            # Generate speech tokens (streaming)
            syn_tokens = self.model.generate_speech(
                prompt=prompt,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            
            # Stream audio chunks
            chunk_count = 0
            for audio_chunk in syn_tokens:
                chunk_count += 1
                yield audio_chunk
            
            logger.info(f"Generated {chunk_count} audio chunks")
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if model is initialized and ready."""
        return self._initialized and self.model is not None


# Global model manager instance
model_manager = ModelManager()
