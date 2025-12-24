"""
Client for vLLM Server - Batch, Streaming, and Concurrent Requests

Examples of how to use the vLLM server with:
- Single requests
- Batch requests
- Streaming responses
- Concurrent requests
"""

import asyncio
import aiohttp
import httpx
import time
from typing import List, AsyncGenerator
import json


class VLLMClient:
    """Client for the vLLM Server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    # ==========================================================================
    # Synchronous Methods
    # ==========================================================================
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict:
        """
        Generate text from a single prompt (synchronous).
        """
        import requests
        
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                **kwargs,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    
    def batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict:
        """
        Generate text for multiple prompts using vLLM's continuous batching.
        """
        import requests
        
        response = requests.post(
            f"{self.base_url}/batch",
            json={
                "prompts": prompts,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    
    def stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ):
        """
        Stream generated text token by token (synchronous generator).
        """
        import requests
        
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                **kwargs,
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "token" in data:
                        yield data["token"]
                    elif data.get("status") == "complete":
                        return data
    
    # ==========================================================================
    # Async Methods
    # ==========================================================================
    
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict:
        """Generate text from a single prompt (async)."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                    **kwargs,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def batch_async(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict:
        """Generate text for multiple prompts (async)."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/batch",
                json={
                    "prompts": prompts,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def stream_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream generated text token by token (async generator)."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                    **kwargs,
                },
            ) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "token" in data:
                            yield data["token"]
    
    async def concurrent_generate(
        self,
        prompts: List[str],
        max_concurrent: int = 16,
        **kwargs,
    ) -> List[dict]:
        """
        Send multiple prompts concurrently (async).
        
        vLLM handles high concurrency natively via continuous batching.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_one(prompt: str) -> dict:
            async with semaphore:
                return await self.generate_async(prompt, **kwargs)
        
        tasks = [process_one(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)


# ==============================================================================
# Example Usage
# ==============================================================================

def example_single_request():
    """Example: Single synchronous request."""
    print("\n" + "="*60)
    print("📝 Example: Single Request (vLLM)")
    print("="*60)
    
    client = VLLMClient()
    
    result = client.generate(
        prompt="What is the capital of France?",
        max_tokens=50,
    )
    
    print(f"Request ID: {result['request_id']}")
    print(f"Generated: {result['generated_text']}")
    print(f"Tokens: {result['output_tokens']} in {result['generation_time']:.2f}s")
    print(f"Speed: {result['tokens_per_second']:.1f} tok/s")


def example_batch_request():
    """Example: Batch request - uses vLLM's continuous batching."""
    print("\n" + "="*60)
    print("📦 Example: Batch Request (vLLM Continuous Batching)")
    print("="*60)
    
    client = VLLMClient()
    
    prompts = [
        "What is 2+2?",
        "Name the largest planet.",
        "What color is the sky?",
        "Who wrote Romeo and Juliet?",
    ]
    
    result = client.batch(prompts=prompts, max_tokens=30)
    
    print(f"Request ID: {result['request_id']}")
    print(f"Total time: {result['total_time']:.2f}s")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Avg speed: {result['avg_tokens_per_second']:.1f} tok/s")
    print("\nResults:")
    for i, r in enumerate(result['results']):
        print(f"  {i+1}. {prompts[i]}")
        print(f"     → {r['generated_text'][:80]}...")


def example_streaming():
    """Example: Streaming response."""
    print("\n" + "="*60)
    print("🌊 Example: Streaming Response (vLLM)")
    print("="*60)
    
    client = VLLMClient()
    
    print("Prompt: Explain quantum computing in 3 sentences.")
    print("Response: ", end="", flush=True)
    
    for token in client.stream(
        prompt="Explain quantum computing in 3 sentences.",
        max_tokens=100,
    ):
        print(token, end="", flush=True)
    
    print("\n")


async def example_concurrent_requests():
    """Example: Concurrent async requests - vLLM handles natively."""
    print("\n" + "="*60)
    print("🔄 Example: Concurrent Requests (vLLM Native Batching)")
    print("="*60)
    
    client = VLLMClient()
    
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
        "What is C++?",
        "What is Java?",
    ]
    
    start_time = time.time()
    
    results = await client.concurrent_generate(
        prompts=prompts,
        max_concurrent=16,  # vLLM can handle high concurrency
        max_tokens=50,
    )
    
    total_time = time.time() - start_time
    
    print(f"Processed {len(prompts)} prompts in {total_time:.2f}s")
    print(f"Avg time per request: {total_time/len(prompts):.2f}s")
    print("\nResults:")
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"  {i+1}. {prompt}")
        print(f"     → {result['generated_text'][:60]}...")


def run_all_examples():
    """Run all examples."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                   vLLM Client Examples                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Synchronous examples
    example_single_request()
    example_batch_request()
    example_streaming()
    
    # Async examples
    asyncio.run(example_concurrent_requests())
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    run_all_examples()
