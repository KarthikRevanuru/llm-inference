"""
FastAPI server for Orpheus TTS using vLLM directly for true concurrent request handling.

This server uses vllm_model_manager instead of model_manager for better concurrency.
Run this instead of server.py when you need high-throughput concurrent TTS generation.

Usage: python server_vllm.py
"""
import io
import time
import wave
import logging
import asyncio
import base64
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from sse_starlette.sse import EventSourceResponse

from config import settings
from vllm_model_manager import get_vllm_model_manager
from metrics import (
    request_counter, request_duration, ttfb_histogram,
    audio_duration_generated, generation_speed, active_requests,
    error_counter, get_metrics, get_metrics_content_type
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting Orpheus TTS server (vLLM mode)...")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Max concurrent requests: {settings.max_num_seqs}")
    
    # Initialize model
    model_manager = get_vllm_model_manager()
    if not model_manager.is_ready():
        raise RuntimeError("Model failed to initialize")
    
    logger.info("Server ready to accept requests (vLLM direct mode - true concurrency enabled)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")


app = FastAPI(
    title="Orpheus TTS API (vLLM)",
    description="High-performance streaming TTS API using Orpheus-3B with vLLM for true concurrency",
    version="1.1.0",
    lifespan=lifespan
)


class TTSRequest(BaseModel):
    """Request model for TTS generation."""
    
    prompt: str = Field(..., description="Text to convert to speech")
    voice: str = Field(
        default=settings.default_voice,
        description=f"Voice name. Options: {', '.join(settings.available_voices)}"
    )
    temperature: float = Field(
        default=settings.default_temperature,
        ge=0.1, le=2.0,
        description="Sampling temperature"
    )
    top_p: float = Field(
        default=settings.default_top_p,
        ge=0.0, le=1.0,
        description="Top-p sampling parameter"
    )
    repetition_penalty: float = Field(
        default=settings.default_repetition_penalty,
        ge=1.0, le=2.0,
        description="Repetition penalty (must be >= 1.1 for stable generation)"
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Optional path to save the generated WAV file."
    )
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate prompt length."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        if len(v) > settings.max_prompt_length:
            raise ValueError(f"Prompt too long (max {settings.max_prompt_length} characters)")
        return v.strip()
    
    @validator('voice')
    def validate_voice(cls, v):
        """Validate voice name."""
        if v not in settings.available_voices:
            raise ValueError(f"Invalid voice. Options: {', '.join(settings.available_voices)}")
        return v


class TTSResponse(BaseModel):
    """Response model for non-streaming TTS."""
    
    audio_duration_seconds: float
    generation_time_seconds: float
    realtime_factor: float
    sample_rate: int = settings.sample_rate
    channels: int = settings.channels
    output_path: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_manager = get_vllm_model_manager()
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    
    return {
        "status": "healthy",
        "model": settings.model_name,
        "max_concurrent_requests": settings.max_num_seqs,
        "mode": "vllm_direct"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type()
    )


@app.post("/v1/tts/stream")
async def stream_tts(request: TTSRequest):
    """
    Stream TTS audio using Server-Sent Events.
    
    Uses vLLM's AsyncLLMEngine for true concurrent request handling.
    """
    start_time = time.time()
    ttfb_recorded = False
    model_manager = get_vllm_model_manager()
    
    active_requests.inc()
    
    async def generate():
        nonlocal ttfb_recorded
        
        try:
            total_bytes = 0
            chunk_count = 0
            
            # Stream directly from async generator
            async for audio_chunk in model_manager.generate_speech_stream(
                prompt=request.prompt,
                voice=request.voice,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
            ):
                if not ttfb_recorded:
                    ttfb = time.time() - start_time
                    ttfb_histogram.observe(ttfb)
                    ttfb_recorded = True
                    logger.info(f"TTFB: {ttfb:.3f}s")
                
                chunk_count += 1
                total_bytes += len(audio_chunk)
                
                # Encode as base64 for SSE transmission
                encoded = base64.b64encode(audio_chunk).decode('utf-8')
                
                yield {
                    "event": "audio_chunk",
                    "data": encoded
                }
            
            # Send completion event
            duration = total_bytes / (settings.sample_rate * settings.sample_width)
            generation_time = time.time() - start_time
            rtf = duration / generation_time if generation_time > 0 else 0
            
            yield {
                "event": "complete",
                "data": {
                    "audio_duration_seconds": duration,
                    "generation_time_seconds": generation_time,
                    "realtime_factor": rtf,
                    "total_chunks": chunk_count
                }
            }
            
            # Record metrics
            request_counter.labels(endpoint="stream", status="success").inc()
            request_duration.labels(endpoint="stream").observe(generation_time)
            audio_duration_generated.inc(duration)
            generation_speed.observe(rtf)
            
            logger.info(f"Stream complete: {duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.2f}x)")
            
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            error_counter.labels(error_type=type(e).__name__).inc()
            request_counter.labels(endpoint="stream", status="error").inc()
            
            yield {
                "event": "error",
                "data": str(e)
            }
        
        finally:
            active_requests.dec()
    
    return EventSourceResponse(generate())


@app.post("/v1/tts/generate", response_model=TTSResponse)
async def generate_tts(request: TTSRequest):
    """
    Generate complete TTS audio (non-streaming).
    
    Uses vLLM's AsyncLLMEngine for true concurrent request handling.
    """
    start_time = time.time()
    model_manager = get_vllm_model_manager()
    active_requests.inc()
    
    try:
        # Collect all audio chunks
        audio_chunks = []
        async for chunk in model_manager.generate_speech_stream(
            prompt=request.prompt,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        ):
            audio_chunks.append(chunk)
        
        # Combine chunks into WAV file
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(settings.channels)
            wf.setsampwidth(settings.sample_width)
            wf.setframerate(settings.sample_rate)
            
            total_frames = 0
            for chunk in audio_chunks:
                wf.writeframes(chunk)
                frame_count = len(chunk) // (wf.getsampwidth() * wf.getnchannels())
                total_frames += frame_count
            
            duration = total_frames / wf.getframerate()
        
        generation_time = time.time() - start_time
        rtf = duration / generation_time if generation_time > 0 else 0
        
        # Record metrics
        request_counter.labels(endpoint="generate", status="success").inc()
        request_duration.labels(endpoint="generate").observe(generation_time)
        audio_duration_generated.inc(duration)
        generation_speed.observe(rtf)
        
        logger.info(f"Generated {duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.2f}x)")
        
        # Save to file if output_path provided
        if request.output_path:
            import os
            output_dir = os.path.dirname(request.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            wav_buffer.seek(0)
            with open(request.output_path, 'wb') as f:
                f.write(wav_buffer.read())
            
            logger.info(f"Saved audio to: {request.output_path}")
            
            return JSONResponse(content={
                "audio_duration_seconds": duration,
                "generation_time_seconds": generation_time,
                "realtime_factor": rtf,
                "sample_rate": settings.sample_rate,
                "channels": settings.channels,
                "output_path": request.output_path
            })
        
        # Return WAV file in response
        wav_buffer.seek(0)
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav",
                "X-Audio-Duration": str(duration),
                "X-Generation-Time": str(generation_time),
                "X-Realtime-Factor": str(rtf)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in generation: {e}")
        error_counter.labels(error_type=type(e).__name__).inc()
        request_counter.labels(endpoint="generate", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        active_requests.dec()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server_vllm:app",
        host=settings.host,
        port=settings.port,
        workers=1,  # Must be 1 for async vLLM engine
        log_level=settings.log_level
    )
