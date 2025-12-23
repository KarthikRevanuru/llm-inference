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
        
        self._request_counter = 0
        self._initialized = True
        
        logger.info("vLLM model manager initialized successfully")
        logger.info(f"Max concurrent sequences: {settings.max_num_seqs}")
    
    def _find_audio_token_offset(self) -> int:
        """Find the starting token ID for audio tokens in the vocabulary."""
        # Pattern 1: <custom_token_0> style (standard for Orpheus)
        test_token = "<custom_token_0>"
        token_id = self.tokenizer.convert_tokens_to_ids(test_token)
        if token_id is not None and token_id != self.tokenizer.unk_token_id:
            return token_id
        
        # Pattern 2: Check for <|audio|> which is also a common marker
        audio_token = "<|audio|>"
        audio_id = self.tokenizer.convert_tokens_to_ids(audio_token)
        
        # Many versions of Orpheus start audio tokens immediately after <|audio|>
        # or at specific hardcoded offsets like 121416
        known_offsets = [121416, 128266, 128256, 156939]
        
        if audio_id is not None and audio_id != self.tokenizer.unk_token_id:
            # Check if tokens immediately after <|audio|> look like audio tokens
            # (they won't be in the standard tokenizer vocabulary or will be special)
            if audio_id + 1 < len(self.tokenizer):
                return audio_id + 1

        for offset in known_offsets:
            if offset < len(self.tokenizer):
                return offset
        
        # Fallback
        logger.warning(f"Could not detect audio token offset, using fallback 121416")
        return 121416
    
    def _format_prompt(self, text: str, voice: str) -> List[int]:
        """
        Format text into token IDs for Orpheus-3B.
        
        Standard format: <|voice_token|>text<|audio|>
        """
        # Get voice token (e.g., <|tara|>)
        voice_token = VOICE_TOKENS.get(voice.lower(), f"<|{voice}|>")
        
        # Check if voice token exists in tokenizer
        voice_id = self.tokenizer.convert_tokens_to_ids(voice_token)
        if voice_id == self.tokenizer.unk_token_id:
            logger.warning(f"Voice token {voice_token} not found, using literal '{voice}:'")
            prompt_text = f"{voice}: {text}<|audio|>"
            return self.tokenizer.encode(prompt_text, add_special_tokens=True)
        
        # Build prompt: [VOICE_ID] + [TEXT_IDS] + [AUDIO_ID]
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        audio_id = self.tokenizer.convert_tokens_to_ids("<|audio|>")
        
        if audio_id == self.tokenizer.unk_token_id:
            # Fallback to observed ID for <|audio|> if not in tokenizer by name
            audio_id = 128257 
            
        full_ids = [voice_id] + text_ids + [audio_id]
        
        logger.debug(f"Formatted prompt IDs: {full_ids}")
        return full_ids
    
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
    
    def _redistribute_codes(self, token_ids: List[int]) -> tuple:
        """
        Redistribute Orpheus audio tokens into SNAC's 3-layer format.
        
        Formula: code = tid - offset - (pos_per_frame * 4096)
        """
        offset = self.audio_token_offset
        
        # Audio tokens must come in frames of 7
        num_frames = len(token_ids) // 7
        if num_frames == 0:
            return None, None, None
            
        audio_codes = []
        for i in range(num_frames):
            frame = token_ids[i*7 : (i+1)*7]
            for pos, tid in enumerate(frame):
                # Calculate the 0-4095 code for this position
                code = tid - offset - (pos * 4096)
                
                # Validation: if code is wildly out of range, then we aren't in audio mode
                if not (0 <= code < 4096):
                    logger.debug(f"Non-audio token detected: {tid} at pos {pos} (calc code: {code})")
                    # Break generation if we encounter non-audio tokens in the middle
                    return None, None, None
                
                audio_codes.append(code)
        
        layer1 = []
        layer2 = []
        layer3 = []
        
        for i in range(num_frames):
            base = i * 7
            layer1.append(audio_codes[base])
            layer2.append(audio_codes[base + 1])
            layer2.append(audio_codes[base + 2])
            layer3.append(audio_codes[base + 3])
            layer3.append(audio_codes[base + 4])
            layer3.append(audio_codes[base + 5])
            layer3.append(audio_codes[base + 6])
        
        return layer1, layer2, layer3
    
    def _tokens_to_audio(self, token_ids: List[int]) -> Optional[bytes]:
        """Convert Orpheus audio tokens to PCM audio using SNAC decoder."""
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
    
    async def generate_speech_stream(
        self,
        prompt: str,
        voice: str = "tara",
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> AsyncIterator[bytes]:
        """Generate speech audio in true streaming fashion with concurrent support."""
        if repetition_penalty < 1.1:
            repetition_penalty = 1.1
        
        # Use token IDs directly to avoid re-tokenization issues in vLLM
        prompt_token_ids = self._format_prompt(prompt, voice)
        sampling_params = self._get_sampling_params(temperature, top_p, repetition_penalty)
        
        self._request_counter += 1
        request_id = f"tts-{self._request_counter}"
        
        logger.info(f"[{request_id}] Starting generation for prompt (length: {len(prompt)})")
        
        collected_tokens: List[int] = []
        last_decoded_frame = 0
        chunk_count = 0
        
        try:
            async for output in self.engine.generate(
                prompt=None,
                sampling_params=sampling_params,
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
            ):
                if output.outputs:
                    out = output.outputs[0]
                    new_token_ids = out.token_ids
                    finish_reason = out.finish_reason
                    
                    # Debug: Log first output to understand what's happening
                    if len(collected_tokens) == 0 and len(new_token_ids) > 0:
                        logger.info(f"[{request_id}] First output - {len(new_token_ids)} tokens, finish_reason: {finish_reason}")
                        logger.info(f"[{request_id}] Token IDs: {list(new_token_ids[:min(20, len(new_token_ids))])}")
                    
                    if len(new_token_ids) > len(collected_tokens):
                        collected_tokens = list(new_token_ids)
                        
                        available_frames = len(collected_tokens) // 7
                        
                        if available_frames > last_decoded_frame:
                            tokens_to_decode = collected_tokens[:available_frames * 7]
                            audio_bytes = self._tokens_to_audio(tokens_to_decode)
                            
                            if audio_bytes:
                                samples_per_frame = 1024
                                prev_samples = last_decoded_frame * samples_per_frame * 2
                                new_audio = audio_bytes[prev_samples:]
                                
                                if new_audio:
                                    chunk_count += 1
                                    yield new_audio
                            
                            last_decoded_frame = available_frames
            
            logger.info(f"[{request_id}] Total tokens collected: {len(collected_tokens)}")
            
            # Final decode
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
