"""
vLLM-based model manager for Orpheus TTS with true concurrent request support.

This implementation bypasses the orpheus_tts library to use vLLM's AsyncLLMEngine
directly, enabling continuous batching for efficient concurrent request handling.
"""
import asyncio
import logging
import torch
import numpy as np
from typing import AsyncIterator, List, Optional
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import snac

from config import settings

logger = logging.getLogger(__name__)

# Voice tokens for Orpheus TTS
VOICE_TOKENS = {
    "tara": "<|tara|>",
    "leah": "<|leah|>",
    "jess": "<|jess|>",
    "leo": "<|leo|>",
    "dan": "<|dan|>",
    "mia": "<|mia|>",
    "zac": "<|zac|>",
    "zoe": "<|zoe|>",
}

# Special tokens
START_TOKEN = "<|audio|>"
END_TOKEN = "<|eoa|>"

# Audio settings
SAMPLE_RATE = 24000


class VLLMModelManager:
    """
    Model manager using vLLM directly for true concurrent TTS generation.
    
    Unlike the orpheus_tts wrapper, this implementation uses vLLM's AsyncLLMEngine
    which supports continuous batching for efficient handling of concurrent requests.
    """
    
    _instance: Optional['VLLMModelManager'] = None
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the vLLM engine and SNAC decoder."""
        if self._initialized:
            return
        
        logger.info(f"Initializing vLLM engine for: {settings.model_name}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        
        # Initialize vLLM AsyncLLMEngine
        engine_args = AsyncEngineArgs(
            model=settings.model_name,
            max_model_len=settings.max_model_len,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            max_num_seqs=settings.max_num_seqs,
            tensor_parallel_size=settings.tensor_parallel_size,
            enforce_eager=True,  # For compatibility
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Initialize SNAC decoder
        logger.info("Loading SNAC audio decoder...")
        self.snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        
        if torch.cuda.is_available():
            self.snac_model = self.snac_model.cuda()
        
        self._request_counter = 0
        self._initialized = True
        
        logger.info("vLLM model manager initialized successfully")
        logger.info(f"Max concurrent sequences: {settings.max_num_seqs}")
    
    def _format_prompt(self, text: str, voice: str) -> str:
        """Format text with voice token and special markers."""
        voice_token = VOICE_TOKENS.get(voice, VOICE_TOKENS["tara"])
        return f"{voice_token}{text}{START_TOKEN}"
    
    def _get_sampling_params(
        self,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> SamplingParams:
        """Create vLLM sampling parameters."""
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=4096,  # Max audio tokens to generate
            stop_token_ids=[self.tokenizer.eos_token_id],
        )
    
    def _tokens_to_audio(self, token_ids: List[int]) -> Optional[bytes]:
        """
        Convert Orpheus audio tokens to PCM audio using SNAC decoder.
        
        Orpheus generates 7 tokens per audio frame that need to be reorganized
        into 3 layers for SNAC decoding.
        """
        if len(token_ids) < 7:
            return None
        
        # Filter to only audio tokens (IDs 128266 to 128266+4096)
        # Audio token offset in Orpheus vocabulary
        audio_token_start = 128266
        audio_token_end = audio_token_start + 4096
        
        audio_tokens = []
        for tid in token_ids:
            if audio_token_start <= tid < audio_token_end:
                audio_tokens.append(tid - audio_token_start)
        
        if len(audio_tokens) < 7:
            return None
        
        # Reorganize tokens into 3 layers for SNAC
        # Orpheus outputs: [L1, L2_1, L2_2, L3_1, L3_2, L3_3, L3_4] per frame
        num_frames = len(audio_tokens) // 7
        if num_frames == 0:
            return None
        
        layer1 = []
        layer2 = []
        layer3 = []
        
        for i in range(num_frames):
            base = i * 7
            layer1.append(audio_tokens[base])
            layer2.extend([audio_tokens[base + 1], audio_tokens[base + 2]])
            layer3.extend([
                audio_tokens[base + 3],
                audio_tokens[base + 4],
                audio_tokens[base + 5],
                audio_tokens[base + 6]
            ])
        
        try:
            # Create tensors for SNAC decoder
            device = next(self.snac_model.parameters()).device
            codes = [
                torch.tensor([layer1], dtype=torch.long, device=device),
                torch.tensor([layer2], dtype=torch.long, device=device),
                torch.tensor([layer3], dtype=torch.long, device=device),
            ]
            
            # Decode to audio
            with torch.no_grad():
                audio = self.snac_model.decode(codes)
            
            # Convert to 16-bit PCM
            audio_np = audio.squeeze().cpu().numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            return audio_int16.tobytes()
            
        except Exception as e:
            logger.warning(f"SNAC decode error: {e}")
            return None
    
    async def generate_speech_stream(
        self,
        prompt: str,
        voice: str = "tara",
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> AsyncIterator[bytes]:
        """
        Generate speech audio in true streaming fashion with concurrent support.
        
        Args:
            prompt: Text to convert to speech
            voice: Voice name (tara, leah, jess, leo, dan, mia, zac, zoe)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty (must be >= 1.1)
            
        Yields:
            Audio chunks as bytes (16-bit PCM at 24kHz)
        """
        # Ensure valid repetition penalty
        if repetition_penalty < 1.1:
            logger.warning(f"Repetition penalty {repetition_penalty} < 1.1, setting to 1.1")
            repetition_penalty = 1.1
        
        # Format prompt
        formatted_prompt = self._format_prompt(prompt, voice)
        
        # Create sampling params
        sampling_params = self._get_sampling_params(temperature, top_p, repetition_penalty)
        
        # Generate unique request ID
        self._request_counter += 1
        request_id = f"tts-{self._request_counter}"
        
        logger.info(f"[{request_id}] Starting generation for prompt (length: {len(prompt)})")
        
        # Collect tokens for audio decoding
        collected_tokens: List[int] = []
        last_decoded_frame = 0
        chunk_count = 0
        
        try:
            # Stream tokens from vLLM
            async for output in self.engine.generate(
                formatted_prompt,
                sampling_params,
                request_id=request_id,
            ):
                if output.outputs:
                    # Get newly generated token IDs
                    new_token_ids = output.outputs[0].token_ids
                    
                    if len(new_token_ids) > len(collected_tokens):
                        # Add new tokens
                        collected_tokens = list(new_token_ids)
                        
                        # Check if we have enough for a new frame (7 tokens per frame)
                        available_frames = len(collected_tokens) // 7
                        
                        if available_frames > last_decoded_frame:
                            # Decode new frames
                            tokens_to_decode = collected_tokens[:available_frames * 7]
                            audio_bytes = self._tokens_to_audio(tokens_to_decode)
                            
                            if audio_bytes:
                                # Calculate new audio portion
                                # Each frame is ~21.33ms of audio (1024 samples at 24kHz)
                                samples_per_frame = 1024
                                prev_samples = last_decoded_frame * samples_per_frame * 2  # 2 bytes per sample
                                new_audio = audio_bytes[prev_samples:]
                                
                                if new_audio:
                                    chunk_count += 1
                                    yield new_audio
                            
                            last_decoded_frame = available_frames
            
            # Final decode for any remaining tokens
            if len(collected_tokens) > last_decoded_frame * 7:
                final_audio = self._tokens_to_audio(collected_tokens)
                if final_audio:
                    prev_samples = last_decoded_frame * 1024 * 2
                    remaining = final_audio[prev_samples:]
                    if remaining:
                        chunk_count += 1
                        yield remaining
            
            logger.info(f"[{request_id}] Generated {chunk_count} audio chunks")
            
        except Exception as e:
            logger.error(f"[{request_id}] Error generating speech: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if model is initialized and ready."""
        return self._initialized


# Lazy initialization wrapper
_vllm_model_manager: Optional[VLLMModelManager] = None


def get_vllm_model_manager() -> VLLMModelManager:
    """Get or create the global vLLM model manager instance."""
    global _vllm_model_manager
    if _vllm_model_manager is None:
        _vllm_model_manager = VLLMModelManager()
    return _vllm_model_manager


class _LazyVLLMModelManager:
    """Lazy wrapper that defers initialization until first attribute access."""
    
    def __getattr__(self, name):
        return getattr(get_vllm_model_manager(), name)


vllm_model_manager = _LazyVLLMModelManager()
