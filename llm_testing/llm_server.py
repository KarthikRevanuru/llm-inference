"""
vLLM-based LLM Server with Batch, Streaming, and Concurrent Request Support

Features:
- Batch inference: Continuous batching via vLLM (highly efficient)
- Streaming responses: Async token streaming via AsyncLLMEngine
- Concurrent requests: Native vLLM async handling
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, AsyncGenerator
from contextlib import asynccontextmanager

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import json

from config import (
    SMALL_LLMS,
    DEFAULT_MODEL,
    DEFAULT_SAMPLING_PARAMS,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ServerConfig:
    """Server configuration"""
    model_name: str = DEFAULT_MODEL
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 2048
    max_concurrent_requests: int = 256  # vLLM handles this natively
    host: str = "0.0.0.0"
    port: int = 8000


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateRequest(BaseModel):
    """Single generation request"""
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = False


class BatchRequest(BaseModel):
    """Batch generation request"""
    prompts: List[str]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    """Generation response"""
    request_id: str
    generated_text: str
    input_tokens: int
    output_tokens: int
    generation_time: float
    tokens_per_second: float


class BatchResponse(BaseModel):
    """Batch generation response"""
    request_id: str
    results: List[GenerateResponse]
    total_time: float
    total_tokens: int
    avg_tokens_per_second: float


class ServerStats(BaseModel):
    """Server statistics"""
    model: str
    backend: str = "vLLM"
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    current_active_requests: int
    total_requests_processed: int
    total_tokens_generated: int
    uptime_seconds: float


# ============================================================================
# vLLM Server
# ============================================================================

class VLLMServer:
    """
    vLLM-based server with native async support for streaming and concurrency.
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model_id = SMALL_LLMS.get(config.model_name, config.model_name)
        
        self.engine: Optional[AsyncLLMEngine] = None
        
        # Stats
        self.active_requests = 0
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
    
    async def load_model(self) -> None:
        """Load the model using vLLM AsyncEngine."""
        print(f"🚀 Loading model with vLLM: {self.model_id}")
        print(f"   Tensor Parallel Size: {self.config.tensor_parallel_size}")
        print(f"   GPU Memory Utilization: {self.config.gpu_memory_utilization}")
        
        start_time = time.time()
        
        engine_args = AsyncEngineArgs(
            model=self.model_id,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            trust_remote_code=True,
            dtype="auto",
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
    
    def _create_sampling_params(self, request: GenerateRequest | BatchRequest) -> SamplingParams:
        """Create vLLM SamplingParams from request."""
        return SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Handle single generation request."""
        if self.engine is None:
            raise RuntimeError("Engine not loaded")
        
        self.active_requests += 1
        self.total_requests += 1
        request_id = str(uuid.uuid4())[:8]
        
        try:
            sampling_params = self._create_sampling_params(request)
            
            start_time = time.time()
            
            # Collect all outputs
            final_output = None
            async for output in self.engine.generate(
                prompt=request.prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_output = output
            
            generation_time = time.time() - start_time
            
            if final_output is None:
                raise RuntimeError("No output generated")
            
            generated_text = final_output.outputs[0].text
            output_tokens = len(final_output.outputs[0].token_ids)
            input_tokens = len(final_output.prompt_token_ids)
            
            self.total_tokens += output_tokens
            
            return GenerateResponse(
                request_id=request_id,
                generated_text=generated_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=generation_time,
                tokens_per_second=output_tokens / generation_time if generation_time > 0 else 0,
            )
            
        finally:
            self.active_requests -= 1
    
    async def generate_stream(
        self,
        request: GenerateRequest,
    ) -> AsyncGenerator[str, None]:
        """Stream generation token by token."""
        if self.engine is None:
            yield f"data: {json.dumps({'error': 'Engine not loaded'})}\n\n"
            return
        
        self.active_requests += 1
        self.total_requests += 1
        request_id = str(uuid.uuid4())[:8]
        
        try:
            sampling_params = self._create_sampling_params(request)
            
            start_time = time.time()
            first_token_time = None
            prev_text = ""
            output_tokens = 0
            input_tokens = 0
            
            # Send start event
            yield f"data: {json.dumps({'request_id': request_id, 'status': 'started'})}\n\n"
            
            async for output in self.engine.generate(
                prompt=request.prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                current_text = output.outputs[0].text
                new_text = current_text[len(prev_text):]
                
                if new_text:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    
                    yield f"data: {json.dumps({'token': new_text})}\n\n"
                    prev_text = current_text
                
                output_tokens = len(output.outputs[0].token_ids)
                input_tokens = len(output.prompt_token_ids)
            
            generation_time = time.time() - start_time
            self.total_tokens += output_tokens
            
            # Send completion event
            yield f"data: {json.dumps({'status': 'complete', 'input_tokens': input_tokens, 'output_tokens': output_tokens, 'generation_time': generation_time, 'ttft': first_token_time, 'tokens_per_second': output_tokens / generation_time if generation_time > 0 else 0})}\n\n"
            
        finally:
            self.active_requests -= 1
    
    async def generate_batch(self, request: BatchRequest) -> BatchResponse:
        """Handle batch generation with vLLM's continuous batching."""
        if self.engine is None:
            raise RuntimeError("Engine not loaded")
        
        self.active_requests += 1
        self.total_requests += 1
        batch_request_id = str(uuid.uuid4())[:8]
        
        try:
            sampling_params = self._create_sampling_params(request)
            
            start_time = time.time()
            
            # Create tasks for all prompts
            async def process_one(idx: int, prompt: str) -> Dict[str, Any]:
                req_id = f"{batch_request_id}-{idx}"
                final_output = None
                
                async for output in self.engine.generate(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=req_id,
                ):
                    final_output = output
                
                if final_output is None:
                    return {"error": "No output"}
                
                return {
                    "request_id": req_id,
                    "generated_text": final_output.outputs[0].text,
                    "input_tokens": len(final_output.prompt_token_ids),
                    "output_tokens": len(final_output.outputs[0].token_ids),
                }
            
            # Run all in parallel - vLLM handles continuous batching internally
            tasks = [process_one(i, p) for i, p in enumerate(request.prompts)]
            results = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            total_tokens = sum(r.get("output_tokens", 0) for r in results)
            
            self.total_tokens += total_tokens
            
            return BatchResponse(
                request_id=batch_request_id,
                results=[
                    GenerateResponse(
                        request_id=r["request_id"],
                        generated_text=r["generated_text"],
                        input_tokens=r["input_tokens"],
                        output_tokens=r["output_tokens"],
                        generation_time=total_time / len(request.prompts),
                        tokens_per_second=r["output_tokens"] / (total_time / len(request.prompts)) if total_time > 0 else 0,
                    )
                    for r in results
                ],
                total_time=total_time,
                total_tokens=total_tokens,
                avg_tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
            )
            
        finally:
            self.active_requests -= 1
    
    def get_stats(self) -> ServerStats:
        """Get server statistics."""
        return ServerStats(
            model=self.model_id,
            backend="vLLM",
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            current_active_requests=self.active_requests,
            total_requests_processed=self.total_requests,
            total_tokens_generated=self.total_tokens,
            uptime_seconds=time.time() - self.start_time,
        )
    
    async def shutdown(self):
        """Cleanup resources."""
        if self.engine is not None:
            # vLLM doesn't have explicit shutdown, just delete
            del self.engine
            self.engine = None


# ============================================================================
# FastAPI Application
# ============================================================================

# Global server instance
server: Optional[VLLMServer] = None
config = ServerConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global server
    server = VLLMServer(config)
    await server.load_model()
    yield
    await server.shutdown()


app = FastAPI(
    title="vLLM Server",
    description="LLM Server with Batch, Streaming, and Concurrent Request Support (vLLM backend)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "model": config.model_name, "backend": "vLLM"}


@app.get("/stats", response_model=ServerStats)
async def get_stats():
    """Get server statistics."""
    return server.get_stats()


@app.get("/models")
async def list_models():
    """List available models."""
    return {"models": SMALL_LLMS}


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate text from a single prompt.
    
    If stream=True, returns Server-Sent Events instead.
    """
    if request.stream:
        return StreamingResponse(
            server.generate_stream(request),
            media_type="text/event-stream",
        )
    
    return await server.generate(request)


@app.post("/batch", response_model=BatchResponse)
async def generate_batch(request: BatchRequest):
    """
    Generate text for multiple prompts using vLLM's continuous batching.
    
    All prompts are processed concurrently with optimal GPU utilization.
    """
    return await server.generate_batch(request)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Run the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Server")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Model to load")
    parser.add_argument("--tensor-parallel", "-tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory", type=float, default=0.85, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=2048, help="Max model context length")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    # Update global config
    global config
    config = ServerConfig(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        host=args.host,
        port=args.port,
    )
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                      vLLM Server                             ║
╠══════════════════════════════════════════════════════════════╣
║  Model: {args.model:<52} ║
║  Tensor Parallel: {args.tensor_parallel:<43} ║
║  GPU Memory: {args.gpu_memory:<48} ║
║  Max Model Len: {args.max_model_len:<45} ║
║  Host: {args.host}:{args.port:<49} ║
╠══════════════════════════════════════════════════════════════╣
║  Features:                                                   ║
║    • Continuous Batching (native vLLM)                       ║
║    • Async Streaming (SSE)                                   ║
║    • High Concurrency (PagedAttention)                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "llm_server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
