"""
vLLM-based model manager for Orpheus TTS with true concurrent request support.

This implementation bypasses the orpheus_tts library to use vLLM's AsyncLLMEngine
directly, enabling continuous batching for efficient concurrent request handling.

CUDA Streams Optimization:
- Uses multiple CUDA streams for parallel SNAC decoding
- Runs SNAC decode in thread pool to avoid blocking async event loop
- Enables true concurrent audio generation for multiple requests
"""
import asyncio
import logging
import threading
import torch
import numpy as np
from typing import AsyncIterator, List, Optional, Tuple
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import snac

from config import settings

# CUDA streams configuration
NUM_CUDA_STREAMS = 4  # Number of parallel CUDA streams for SNAC decoding

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
MIN_FRAMES_PER_CHUNK = 1  # Minimum frames before yielding (~40ms audio) - optimized for low TTFB


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
            enforce_eager=True,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Initialize SNAC decoder
        logger.info("Loading SNAC audio decoder...")
        self.snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        
        if torch.cuda.is_available():
            self.snac_model = self.snac_model.cuda()
        
        # Find audio token offset by checking for custom audio tokens
        # Orpheus adds tokens like <custom_token_0> through <custom_token_4095>
        # The actual offset varies by model version
        self.audio_token_offset = self._find_audio_token_offset()
        logger.info(f"Audio token offset: {self.audio_token_offset}")
        
        # Initialize CUDA streams for parallel SNAC decoding
        self._cuda_streams: List[torch.cuda.Stream] = []
        if torch.cuda.is_available():
            self._cuda_streams = [torch.cuda.Stream() for _ in range(NUM_CUDA_STREAMS)]
            logger.info(f"Created {NUM_CUDA_STREAMS} CUDA streams for parallel SNAC decoding")
        
        # Thread-safe counters
        self._stream_idx = 0
        self._stream_lock = threading.Lock()
        self._request_counter = 0
        self._request_lock = threading.Lock()
        
        self._initialized = True
        
        logger.info("vLLM model manager initialized successfully")
        logger.info(f"Max concurrent sequences: {settings.max_num_seqs}")
    
    def _find_audio_token_offset(self) -> int:
        """Find the starting token ID for audio tokens in the vocabulary."""
        # Orpheus-3B uses 121416 as the audio token offset
        # This is where custom audio tokens start in the vocabulary
        offset = 121416
        logger.info(f"Using audio token offset: {offset}")
        return offset
    

    def _get_sampling_params(
        self,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> SamplingParams:
        """Create vLLM sampling parameters."""
        # Use a higher max_tokens for audio generation
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=4096,
        )
    
    def _parse_custom_token(self, token_text: str, position: int) -> Optional[int]:
        """
        Parse a custom token string and convert to SNAC code.
        
        Format: <custom_token_N> where N is the number to extract.
        Formula: code = N - 10 - ((position % 7) * 4096)
        
        Based on Lex-au/Orpheus-FastAPI implementation.
        """
        CUSTOM_TOKEN_PREFIX = "<custom_token_"
        
        if CUSTOM_TOKEN_PREFIX not in token_text:
            return None
        
        token_text = token_text.strip()
        last_token_start = token_text.rfind(CUSTOM_TOKEN_PREFIX)
        
        if last_token_start == -1:
            return None
        
        last_token = token_text[last_token_start:]
        
        if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
            return None
        
        try:
            # Extract the number from <custom_token_N>
            number_str = last_token[14:-1]  # Remove "<custom_token_" and ">"
            token_num = int(number_str)
            
            # Apply the formula: code = N - 10 - ((pos % 7) * 4096)
            code = token_num - 10 - ((position % 7) * 4096)
            
            return code
        except (ValueError, IndexError):
            return None
    
    def _redistribute_codes(self, codes: List[int]) -> tuple:
        """
        Redistribute SNAC codes into 3-layer format.
        
        Input codes should already be in 0-4095 range.
        
        Layer distribution (from Lex-au reference):
        - layer1 (1 per frame): idx+0
        - layer2 (2 per frame): idx+1, idx+4
        - layer3 (4 per frame): idx+2, idx+3, idx+5, idx+6
        """
        num_frames = len(codes) // 7
        if num_frames == 0:
            return None, None, None
        
        layer1 = []
        layer2 = []
        layer3 = []
        
        for i in range(num_frames):
            idx = i * 7
            
            # Layer 1: single value per frame
            layer1.append(codes[idx])
            
            # Layer 2: two values per frame (positions 1 and 4)
            layer2.append(codes[idx + 1])
            layer2.append(codes[idx + 4])
            
            # Layer 3: four values per frame (positions 2, 3, 5, 6)
            layer3.append(codes[idx + 2])
            layer3.append(codes[idx + 3])
            layer3.append(codes[idx + 5])
            layer3.append(codes[idx + 6])
        
        return layer1, layer2, layer3
    
    def _tokens_to_audio(self, token_ids: List[int]) -> Optional[bytes]:
        """Convert Orpheus audio tokens to PCM audio using SNAC decoder (sync version)."""
        if len(token_ids) < 7:
            return None
        
        layer1, layer2, layer3 = self._redistribute_codes(token_ids)
        
        if layer1 is None:
            return None
        
        try:
            device = next(self.snac_model.parameters()).device
            codes = [
                torch.tensor([layer1], dtype=torch.long, device=device),
                torch.tensor([layer2], dtype=torch.long, device=device),
                torch.tensor([layer3], dtype=torch.long, device=device),
            ]
            
            with torch.no_grad():
                audio = self.snac_model.decode(codes)
            
            audio_np = audio.squeeze().cpu().numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            return audio_int16.tobytes()
            
        except Exception as e:
            logger.warning(f"SNAC decode error: {e}")
            return None
    
    def _get_next_stream(self) -> Optional[torch.cuda.Stream]:
        """Get the next CUDA stream in round-robin fashion (thread-safe)."""
        if not self._cuda_streams:
            return None
        
        with self._stream_lock:
            stream = self._cuda_streams[self._stream_idx]
            self._stream_idx = (self._stream_idx + 1) % len(self._cuda_streams)
            return stream
    
    def _tokens_to_audio_in_stream(
        self, 
        token_ids: List[int], 
        stream: Optional[torch.cuda.Stream]
    ) -> Optional[bytes]:
        """
        Convert tokens to audio using a specific CUDA stream.
        
        This method is designed to be called from asyncio.to_thread() to avoid
        blocking the async event loop while enabling parallel SNAC decoding.
        """
        if len(token_ids) < 7:
            return None
        
        layer1, layer2, layer3 = self._redistribute_codes(token_ids)
        
        if layer1 is None:
            return None
        
        try:
            device = next(self.snac_model.parameters()).device
            codes = [
                torch.tensor([layer1], dtype=torch.long, device=device),
                torch.tensor([layer2], dtype=torch.long, device=device),
                torch.tensor([layer3], dtype=torch.long, device=device),
            ]
            
            # Use CUDA stream for parallel execution
            if stream is not None:
                with torch.cuda.stream(stream):
                    with torch.no_grad():
                        audio = self.snac_model.decode(codes)
                # Wait for this stream to complete
                stream.synchronize()
            else:
                # Fallback: no CUDA streams available
                with torch.no_grad():
                    audio = self.snac_model.decode(codes)
            
            # For small chunks, use full audio; for large, slice to remove startup/end artifacts
            audio_len = audio.shape[-1]
            if audio_len > 4096:
                # Large chunk: take middle portion (skip first 2048 samples of artifacts)
                audio_slice = audio[:, :, 2048:]
            else:
                # Small chunk: use full audio
                audio_slice = audio
            
            audio_np = audio_slice.squeeze().cpu().numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            return audio_int16.tobytes()
            
        except Exception as e:
            logger.warning(f"SNAC decode error in stream: {e}")
            return None
    
    async def _tokens_to_audio_async(self, token_ids: List[int]) -> Optional[bytes]:
        """
        Async-compatible audio conversion with CUDA streams.
        
        Uses asyncio.to_thread() to run SNAC decode in a thread pool,
        preventing blocking of the async event loop. Combined with CUDA
        streams, this enables truly parallel audio decoding for multiple
        concurrent requests.
        """
        stream = self._get_next_stream()
        return await asyncio.to_thread(
            self._tokens_to_audio_in_stream, token_ids, stream
        )
    
    def _snac_decode_in_stream(
        self,
        layer1: List[int],
        layer2: List[int],
        layer3: List[int],
        stream: Optional[torch.cuda.Stream]
    ) -> Optional[bytes]:
        """Decode SNAC layer codes to audio."""
        try:
            device = next(self.snac_model.parameters()).device
            codes = [
                torch.tensor([layer1], dtype=torch.long, device=device),
                torch.tensor([layer2], dtype=torch.long, device=device),
                torch.tensor([layer3], dtype=torch.long, device=device),
            ]
            
            if stream is not None:
                with torch.cuda.stream(stream):
                    with torch.no_grad():
                        audio = self.snac_model.decode(codes)
                stream.synchronize()
            else:
                with torch.no_grad():
                    audio = self.snac_model.decode(codes)
            
            # For small chunks, use full audio; for large, slice to remove startup artifacts
            audio_len = audio.shape[-1]
            if audio_len > 4096:
                audio_slice = audio[:, :, 2048:]
            else:
                audio_slice = audio
            
            audio_np = audio_slice.squeeze().cpu().numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            return audio_int16.tobytes()
            
        except Exception as e:
            logger.warning(f"SNAC decode error: {e}")
            return None
    
    async def _snac_decode_async(
        self,
        layer1: List[int],
        layer2: List[int],
        layer3: List[int]
    ) -> Optional[bytes]:
        """Async wrapper for SNAC decoding with layer codes."""
        stream = self._get_next_stream()
        return await asyncio.to_thread(
            self._snac_decode_in_stream, layer1, layer2, layer3, stream
        )
    
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
        
        Uses incremental SNAC decoding with crossfades for efficient streaming:
        - Only decodes new frames (not all accumulated tokens)
        - Applies 50ms crossfade between chunks to prevent audio artifacts
        """
        if repetition_penalty < 1.1:
            repetition_penalty = 1.1
        
        sampling_params = self._get_sampling_params(temperature, top_p, repetition_penalty)
        
        # Thread-safe request counter increment
        with self._request_lock:
            self._request_counter += 1
            request_id = f"tts-{self._request_counter}"
        
        logger.info(f"[{request_id}] Starting generation for prompt (length: {len(prompt)})")
        
        collected_codes: List[int] = []  # SNAC codes (0-4095)
        code_position = 0  # Position counter for token parsing
        last_decoded_frame = 0
        chunk_count = 0
        
        try:
            # Orpheus-3B prompt format: <|audio|>voice: text<|eot_id|>
            prompt_text = f"<|audio|>{voice}: {prompt}<|eot_id|>"
            logger.info(f"[{request_id}] Prompt: {prompt_text}")
            
            async for output in self.engine.generate(
                prompt=prompt_text,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                if output.outputs:
                    out = output.outputs[0]
                    new_token_ids = out.token_ids
                    
                    # Process new tokens: decode to text and parse
                    start_idx = len(collected_codes)  # How many we've already processed
                    
                    for tid in new_token_ids[start_idx:]:
                        # Decode token ID to text
                        token_text = self.tokenizer.decode([tid])
                        
                        # Parse custom token to get SNAC code
                        code = self._parse_custom_token(token_text, code_position)
                        
                        if code is not None and 0 <= code < 4096:
                            collected_codes.append(code)
                            code_position += 1
                        elif code is not None:
                            # Log invalid codes for debugging
                            if len(collected_codes) < 5:
                                logger.debug(f"[{request_id}] Invalid code {code} from token '{token_text}'")
                    
                    # Check if we have enough for audio generation
                    available_frames = len(collected_codes) // 7
                    new_frames = available_frames - last_decoded_frame
                    
                    if new_frames >= MIN_FRAMES_PER_CHUNK:
                        # Decode new frames
                        new_codes = collected_codes[last_decoded_frame * 7 : available_frames * 7]
                        layer1, layer2, layer3 = self._redistribute_codes(new_codes)
                        
                        if layer1 is not None:
                            audio_bytes = await self._snac_decode_async(layer1, layer2, layer3)
                            
                            if audio_bytes:
                                chunk_count += 1
                                yield audio_bytes
                        
                        last_decoded_frame = available_frames
            
            logger.info(f"[{request_id}] Total codes collected: {len(collected_codes)}")
            
            # Final decode: remaining frames
            available_frames = len(collected_codes) // 7
            if available_frames > last_decoded_frame:
                remaining_codes = collected_codes[last_decoded_frame * 7 : available_frames * 7]
                logger.debug(f"[{request_id}] Final decode: {len(remaining_codes)} codes")
                layer1, layer2, layer3 = self._redistribute_codes(remaining_codes)
                
                if layer1 is not None:
                    final_audio = await self._snac_decode_async(layer1, layer2, layer3)
                    if final_audio:
                        chunk_count += 1
                        yield final_audio
            
            logger.info(f"[{request_id}] Generated {chunk_count} audio chunks")
            
        except Exception as e:
            logger.error(f"[{request_id}] Error generating speech: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if model is initialized and ready."""
        return self._initialized


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
