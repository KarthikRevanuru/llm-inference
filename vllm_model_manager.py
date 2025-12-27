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
import time
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
CROSSFADE_SAMPLES = 1200  # 50ms crossfade at 24kHz (0.05 * 24000)
MIN_FRAMES_PER_CHUNK = 7  # Minimum frames before yielding (~300ms of audio)


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
            max_model_len=1024,  # TTS prompts are short, 1024 is plenty
            gpu_memory_utilization=0.85,  # Balanced for A40 + SNAC
            max_num_seqs=settings.max_num_seqs,
            tensor_parallel_size=settings.tensor_parallel_size,
            enforce_eager=False,  # Enable CUDA graphs for better throughput
            dtype="bfloat16",  # A40 supports bf16 natively
            enable_prefix_caching=True,  # Speed up TTFT for repeated patterns
            block_size=16,  # Smaller blocks for better latency
            num_scheduler_steps=10,  # Multi-step scheduling to reduce CPU overhead
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Build token ID mapping at startup (high-performance lookup)
        logger.info("Building token ID mapping from tokenizer vocabulary...")
        self._token_id_to_num = {}
        vocab = self.tokenizer.get_vocab()
        CUSTOM_TOKEN_PREFIX = "<custom_token_"
        for token_text, tid in vocab.items():
            if token_text.startswith(CUSTOM_TOKEN_PREFIX) and token_text.endswith(">"):
                try:
                    num = int(token_text[14:-1])
                    self._token_id_to_num[tid] = num
                except ValueError:
                    continue
        logger.info(f"Mapped {len(self._token_id_to_num)} custom tokens")
        
        # Initialize SNAC decoder
        logger.info("Loading SNAC audio decoder...")
        self.snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        
        if torch.cuda.is_available():
            self.snac_model = self.snac_model.cuda()
            snac_device = next(self.snac_model.parameters()).device
            logger.info(f"SNAC decoder device: {snac_device}")
        else:
            logger.warning("CUDA not available - SNAC running on CPU (will be slow!)")
        
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
        
        # Log GPU status summary
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            logger.info(f"vLLM engine: GPU (tensor_parallel={settings.tensor_parallel_size})")
        else:
            logger.warning("No GPU detected - running on CPU")
        
        logger.info(f"Max concurrent sequences: {settings.max_num_seqs}")
    
    def _find_audio_token_offset(self) -> int:
        """Find the starting token ID for audio tokens in the vocabulary."""
        # Orpheus-3B uses 121416 as the audio token offset
        # This is where custom audio tokens start in the vocabulary
        offset = 121416
        logger.info(f"Using audio token offset: {offset}")
    async def warmup(self):
        """Warm up the engine and decoder to reduce TTFA on first real request."""
        logger.info("Warming up TTS engine...")
        try:
            # Short dummy generation
            async for _ in self.generate_speech_stream("Warm up", voice="tara"):
                pass
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def _get_sampling_params(
        self,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> SamplingParams:
        """Create vLLM sampling parameters."""
        # Audio control tokens (from Maya-1-Voice reference)
        CODE_END_TOKEN_ID = 128258  # End of Speech - stop generation here
        
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=4096,
            stop_token_ids=[CODE_END_TOKEN_ID],  # Stop when audio ends
        )
    
    def _token_id_to_code(self, token_id: int, position: int) -> Optional[int]:
        """
        Convert token ID directly to SNAC code without tokenizer.decode().
        
        Based on reference implementation:
        CODE_TOKEN_OFFSET = 128266  # Start of SNAC codes
        code = (token_id - CODE_TOKEN_OFFSET) % 4096
        
        This avoids CPU-bound tokenizer.decode() calls in the hot path.
        """
        # Constants from reference
        CODE_TOKEN_OFFSET = 128266  # Start of SNAC codes
        SNAC_MAX_ID = 156937        # 128266 + (7 * 4096) - 1
        
        # Check if this is a SNAC token
        if token_id < CODE_TOKEN_OFFSET or token_id > SNAC_MAX_ID:
            return None
        
        # Simple formula: mod 4096 handles layer cycling automatically
        code = (token_id - CODE_TOKEN_OFFSET) % 4096
        
        return code
    
    def _parse_custom_token(self, token_text: str, position: int) -> Optional[int]:
        """
        Parse a custom token string and convert to SNAC code.
        
        Format: <custom_token_N> where N is the number to extract.
        Formula: code = N - 10 - ((position % 7) * 4096)
        
        Based on Lex-au/Orpheus-FastAPI implementation.
        DEPRECATED: Use _token_id_to_code() for better performance.
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
    
    def _get_next_stream(self) -> Optional[torch.cuda.Stream]:
        """Get the next CUDA stream in round-robin fashion (thread-safe)."""
        if not self._cuda_streams:
            return None
        
        with self._stream_lock:
            stream = self._cuda_streams[self._stream_idx]
            self._stream_idx = (self._stream_idx + 1) % len(self._cuda_streams)
            return stream
    
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
            
            audio_np = audio.squeeze().cpu().numpy()
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
    
    def _apply_crossfade(
        self, 
        prev_audio: Optional[np.ndarray], 
        new_audio: np.ndarray,
        is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply crossfade between previous and new audio chunks.
        
        Args:
            prev_audio: Previous chunk's overlap region
            new_audio: New audio to process
            is_final: If True, don't hold back overlap (last chunk)
        
        Returns:
            Tuple of (audio_to_yield, overlap_for_next_chunk)
        """
        if prev_audio is None or len(prev_audio) == 0:
            # First chunk: keep overlap at end for next crossfade (unless final)
            if not is_final and len(new_audio) > CROSSFADE_SAMPLES:
                audio_to_yield = new_audio[:-CROSSFADE_SAMPLES]
                overlap = new_audio[-CROSSFADE_SAMPLES:]
                return audio_to_yield, overlap
            else:
                return new_audio, np.array([], dtype=np.int16)
        
        # Apply crossfade: fade out prev_audio, fade in new_audio
        fade_out = np.linspace(1.0, 0.0, CROSSFADE_SAMPLES, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, CROSSFADE_SAMPLES, dtype=np.float32)
        
        # Blend the overlap region
        prev_float = prev_audio.astype(np.float32)
        new_start_float = new_audio[:CROSSFADE_SAMPLES].astype(np.float32)
        
        blended = (prev_float * fade_out + new_start_float * fade_in).astype(np.int16)
        
        # Combine: blended region + rest of new audio (minus overlap for next if not final)
        if is_final:
            # Final chunk: yield everything, no overlap needed
            audio_to_yield = np.concatenate([blended, new_audio[CROSSFADE_SAMPLES:]])
            return audio_to_yield, np.array([], dtype=np.int16)
        elif len(new_audio) > CROSSFADE_SAMPLES * 2:
            audio_to_yield = np.concatenate([blended, new_audio[CROSSFADE_SAMPLES:-CROSSFADE_SAMPLES]])
            overlap = new_audio[-CROSSFADE_SAMPLES:]
        elif len(new_audio) > CROSSFADE_SAMPLES:
            audio_to_yield = blended
            overlap = new_audio[CROSSFADE_SAMPLES:]
        else:
            audio_to_yield = blended
            overlap = np.array([], dtype=np.int16)
        
        return audio_to_yield, overlap
    
    async def generate_speech_stream(
        self,
        prompt: str,
        voice: str = "tara",
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> AsyncIterator[bytes]:
        """
        Generate speech audio in true streaming fashion with concurrent support.
        
        Uses incremental SNAC decoding with crossfades for efficient streaming:
        - Only decodes new frames (not all accumulated tokens)
        - Applies 50ms crossfade between chunks to prevent audio artifacts
        
        Performance metrics are logged at the end of each request.
        """
        if repetition_penalty < 1.1:
            repetition_penalty = 1.1
        
        sampling_params = self._get_sampling_params(temperature, top_p, repetition_penalty)
        
        # Thread-safe request counter increment
        with self._request_lock:
            self._request_counter += 1
            request_id = f"tts-{self._request_counter}"
        
        logger.info(f"[{request_id}] Starting generation for prompt (length: {len(prompt)})")
        
        # Performance tracking
        request_start_time = time.perf_counter()
        first_audio_time: Optional[float] = None
        total_tokens = 0
        snac_decode_times: list[float] = []
        
        collected_codes: List[int] = []  # SNAC codes (0-4095)
        code_position = 0  # Position counter for token parsing
        last_decoded_frame = 0
        chunk_count = 0
        crossfade_overlap: Optional[np.ndarray] = None
        
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
                    total_tokens = len(new_token_ids)
                    
                    # Process new tokens: look up in pre-built map (fastest)
                    start_idx = len(collected_codes)
                    
                    for tid in new_token_ids[start_idx:]:
                        # Check pre-built token ID to num mapping
                        token_num = self._token_id_to_num.get(tid)
                        
                        if token_num is not None:
                            # Apply formula: code = N - 10 - ((pos % 7) * 4096)
                            code = token_num - 10 - ((code_position % 7) * 4096)
                            
                            if 0 <= code < 4096:
                                collected_codes.append(code)
                                code_position += 1
                    
                    # Check if we have enough for audio generation
                    available_frames = len(collected_codes) // 7
                    new_frames = available_frames - last_decoded_frame
                    
                    if new_frames >= MIN_FRAMES_PER_CHUNK:
                        # Decode new frames
                        new_codes = collected_codes[last_decoded_frame * 7 : available_frames * 7]
                        layer1, layer2, layer3 = self._redistribute_codes(new_codes)
                        
                        if layer1 is not None:
                            # Time SNAC decode
                            snac_start = time.perf_counter()
                            audio_bytes = await self._snac_decode_async(layer1, layer2, layer3)
                            snac_elapsed = time.perf_counter() - snac_start
                            snac_decode_times.append(snac_elapsed)
                            
                            if audio_bytes:
                                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                                audio_to_yield, crossfade_overlap = self._apply_crossfade(
                                    crossfade_overlap, audio_array
                                )
                                
                                if len(audio_to_yield) > 0:
                                    # Record time to first audio
                                    if first_audio_time is None:
                                        first_audio_time = time.perf_counter()
                                        ttfa = first_audio_time - request_start_time
                                        logger.info(f"[{request_id}] TTFA: {ttfa*1000:.1f}ms")
                                    
                                    chunk_count += 1
                                    yield audio_to_yield.tobytes()
                        
                        last_decoded_frame = available_frames
            
            logger.info(f"[{request_id}] Total codes collected: {len(collected_codes)}")
            
            # Final decode: remaining frames
            available_frames = len(collected_codes) // 7
            if available_frames > last_decoded_frame:
                remaining_codes = collected_codes[last_decoded_frame * 7 : available_frames * 7]
                logger.debug(f"[{request_id}] Final decode: {len(remaining_codes)} codes")
                layer1, layer2, layer3 = self._redistribute_codes(remaining_codes)
                
                if layer1 is not None:
                    snac_start = time.perf_counter()
                    final_audio = await self._snac_decode_async(layer1, layer2, layer3)
                    snac_elapsed = time.perf_counter() - snac_start
                    snac_decode_times.append(snac_elapsed)
                    
                    if final_audio:
                        audio_array = np.frombuffer(final_audio, dtype=np.int16)
                        # Use is_final=True to not hold back samples
                        audio_to_yield, crossfade_overlap = self._apply_crossfade(
                            crossfade_overlap, audio_array, is_final=True
                        )
                        if len(audio_to_yield) > 0:
                            if first_audio_time is None:
                                first_audio_time = time.perf_counter()
                            chunk_count += 1
                            yield audio_to_yield.tobytes()
            
            # Yield any remaining crossfade overlap (in case final decode didn't happen)
            if crossfade_overlap is not None and len(crossfade_overlap) > 0:
                yield crossfade_overlap.tobytes()
            
            # Performance summary
            total_time = time.perf_counter() - request_start_time
            tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
            ttfa_ms = (first_audio_time - request_start_time) * 1000 if first_audio_time else 0
            avg_snac_ms = (sum(snac_decode_times) / len(snac_decode_times) * 1000) if snac_decode_times else 0
            
            logger.info(
                f"[{request_id}] PERF: {total_tokens} tokens in {total_time:.2f}s "
                f"({tokens_per_sec:.0f} tok/s), TTFA: {ttfa_ms:.0f}ms, "
                f"SNAC: {len(snac_decode_times)} calls avg {avg_snac_ms:.1f}ms"
            )
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
