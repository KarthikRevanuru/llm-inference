"""FastAPI server for Orpheus TTS with streaming support."""
import io
import time
import wave
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from sse_starlette.sse import EventSourceResponse

from config import settings
from model_manager import model_manager
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
    logger.info("Starting Orpheus TTS server...")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Max concurrent requests: {settings.max_num_seqs}")
    
    # Initialize model (happens in ModelManager singleton)
    if not model_manager.is_ready():
        raise RuntimeError("Model failed to initialize")
    
    logger.info("Server ready to accept requests")
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")


app = FastAPI(
    title="Orpheus TTS API",
    description="High-performance streaming TTS API using Orpheus-3B",
    version="1.0.0",
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    
    return {
        "status": "healthy",
        "model": settings.model_name,
        "max_concurrent_requests": settings.max_num_seqs
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
    
    Returns audio chunks as base64-encoded data in SSE format.
    This minimizes TTFB by streaming chunks as they're generated.
    """
    start_time = time.time()
    ttfb_recorded = False
    
    active_requests.inc()
    
    async def generate():
        nonlocal ttfb_recorded
        
        try:
            total_bytes = 0
            chunk_count = 0
            
            # Run blocking generation in thread pool
            loop = asyncio.get_event_loop()
            
            def sync_generate():
                return model_manager.generate_speech_stream(
                    prompt=request.prompt,
                    voice=request.voice,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                )
            
            # Get the generator
            audio_generator = await loop.run_in_executor(None, sync_generate)
            
            # Stream chunks
            for audio_chunk in audio_generator:
                if not ttfb_recorded:
                    ttfb = time.time() - start_time
                    ttfb_histogram.observe(ttfb)
                    ttfb_recorded = True
                    logger.info(f"TTFB: {ttfb:.3f}s")
                
                chunk_count += 1
                total_bytes += len(audio_chunk)
                
                # Encode as base64 for SSE transmission
                import base64
                encoded = base64.b64encode(audio_chunk).decode('utf-8')
                
                yield {
                    "event": "audio_chunk",
                    "data": encoded
                }
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.001)
            
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
    
    Returns a WAV file with the complete audio.
    """
    start_time = time.time()
    active_requests.inc()
    
    try:
        # Generate audio chunks
        loop = asyncio.get_event_loop()
        
        def sync_generate():
            return list(model_manager.generate_speech_stream(
                prompt=request.prompt,
                voice=request.voice,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
            ))
        
        audio_chunks = await loop.run_in_executor(None, sync_generate)
        
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
        
        # Return WAV file
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
        "server:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level
    )
