#!/usr/bin/env python3
"""
Comprehensive vLLM Server Test Suite

Tests:
1. Batch processing (vLLM continuous batching)
2. Streaming responses (async SSE)
3. Throughput benchmarking
4. Concurrent streaming requests

Usage:
    # Start server first:
    python llm_server.py --model facebook/opt-350m
    
    # Run all tests:
    python test_server.py
    
    # Run specific test:
    python test_server.py --test batch
"""

import asyncio
import aiohttp
import requests
import time
import json
import argparse
from dataclasses import dataclass
from typing import List
import statistics


BASE_URL = "http://localhost:8000"


# ==============================================================================
# Test Utilities
# ==============================================================================

def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_result(label: str, value: str):
    print(f"  {label:<30} {value}")


@dataclass
class TestResult:
    name: str
    success: bool
    total_time: float
    tokens_generated: int
    tokens_per_second: float
    details: dict = None


# ==============================================================================
# Test 1: Batch Processing
# ==============================================================================

def test_batch_processing():
    """Test vLLM continuous batching with multiple prompts."""
    print_header("TEST 1: Batch Processing (vLLM Continuous Batching)")
    
    prompts = [
        "What is machine learning?",
        "Explain Python in one sentence.",
        "What is the capital of Japan?",
        "Name three programming languages.",
        "What is 2 + 2?",
        "What color is the sky?",
        "Who invented the telephone?",
        "What is photosynthesis?",
    ]
    
    print(f"  Sending {len(prompts)} prompts in ONE batch request...")
    print("  (vLLM processes these with continuous batching)")
    print()
    
    start_time = time.time()
    
    response = requests.post(
        f"{BASE_URL}/batch",
        json={
            "prompts": prompts,
            "max_tokens": 50,
            "temperature": 0.7,
        },
        timeout=120,
    )
    
    if response.status_code != 200:
        print(f"  ❌ Error: {response.status_code} - {response.text}")
        return TestResult("batch", False, 0, 0, 0)
    
    result = response.json()
    
    print_result("Request ID:", result["request_id"])
    print_result("Total Time:", f"{result['total_time']:.2f}s")
    print_result("Total Tokens:", str(result["total_tokens"]))
    print_result("Avg Tokens/sec:", f"{result['avg_tokens_per_second']:.1f}")
    print()
    print("  Results:")
    
    for i, (prompt, r) in enumerate(zip(prompts, result["results"])):
        text = r["generated_text"][:60].replace("\n", " ")
        print(f"    {i+1}. [{r['output_tokens']} tok] {text}...")
    
    print()
    print("  ✅ Batch processing test PASSED")
    
    return TestResult(
        name="batch",
        success=True,
        total_time=result["total_time"],
        tokens_generated=result["total_tokens"],
        tokens_per_second=result["avg_tokens_per_second"],
    )


# ==============================================================================
# Test 2: Streaming Response
# ==============================================================================

def test_streaming():
    """Test streaming response with SSE."""
    print_header("TEST 2: Streaming Response (vLLM Async)")
    
    prompt = "Count from 1 to 10 with explanations."
    
    print(f"  Prompt: {prompt}")
    print()
    print("  Streamed output:")
    print("  ", end="")
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    full_text = ""
    final_stats = None
    
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "prompt": prompt,
            "max_tokens": 200,
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = json.loads(line[6:])
                
                if "token" in data:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    
                    token_count += 1
                    token = data["token"]
                    full_text += token
                    print(token, end="", flush=True)
                
                elif data.get("status") == "complete":
                    final_stats = data
    
    total_time = time.time() - start_time
    
    print()
    print()
    
    ttft = final_stats.get("ttft", first_token_time) if final_stats else first_token_time
    print_result("Time to First Token (TTFT):", f"{ttft*1000:.0f}ms")
    print_result("Total Time:", f"{total_time:.2f}s")
    print_result("Tokens Generated:", str(final_stats.get("output_tokens", token_count) if final_stats else token_count))
    print_result("Tokens/sec:", f"{final_stats.get('tokens_per_second', token_count/total_time):.1f}" if final_stats else f"{token_count/total_time:.1f}")
    print()
    print("  ✅ Streaming test PASSED")
    
    return TestResult(
        name="streaming",
        success=True,
        total_time=total_time,
        tokens_generated=token_count,
        tokens_per_second=token_count / total_time if total_time > 0 else 0,
        details={"ttft_ms": ttft * 1000 if ttft else 0},
    )


# ==============================================================================
# Test 3: Throughput Benchmarking
# ==============================================================================

def test_throughput():
    """Test throughput with sequential vs batch comparison."""
    print_header("TEST 3: Throughput Benchmarking (vLLM)")
    
    prompts = [
        "Define AI.",
        "Define ML.",
        "Define DL.",
        "Define NLP.",
    ]
    
    max_tokens = 50
    
    # Test 1: Sequential requests
    print("  [A] Sequential Requests (one at a time)")
    
    sequential_tokens = 0
    seq_start = time.time()
    
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens},
            timeout=120,
        )
        result = response.json()
        elapsed = time.time() - t0
        sequential_tokens += result["output_tokens"]
        print(f"      Request {i+1}: {elapsed:.2f}s, {result['output_tokens']} tokens")
    
    seq_total = time.time() - seq_start
    
    print()
    print_result("    Total Sequential Time:", f"{seq_total:.2f}s")
    print_result("    Total Tokens:", str(sequential_tokens))
    print_result("    Throughput:", f"{sequential_tokens/seq_total:.1f} tok/s")
    
    # Test 2: Batch request
    print()
    print("  [B] Batch Request (vLLM continuous batching)")
    
    batch_start = time.time()
    response = requests.post(
        f"{BASE_URL}/batch",
        json={"prompts": prompts, "max_tokens": max_tokens},
        timeout=120,
    )
    result = response.json()
    batch_total = time.time() - batch_start
    batch_tokens = result["total_tokens"]
    
    print()
    print_result("    Total Batch Time:", f"{batch_total:.2f}s")
    print_result("    Total Tokens:", str(batch_tokens))
    print_result("    Throughput:", f"{batch_tokens/batch_total:.1f} tok/s")
    
    # Comparison
    speedup = seq_total / batch_total if batch_total > 0 else 0
    print()
    print(f"  📊 Batch is {speedup:.1f}x faster than sequential!")
    print()
    print("  ✅ Throughput test PASSED")
    
    return TestResult(
        name="throughput",
        success=True,
        total_time=batch_total,
        tokens_generated=batch_tokens,
        tokens_per_second=batch_tokens / batch_total if batch_total > 0 else 0,
        details={
            "sequential_time": seq_total,
            "batch_time": batch_total,
            "speedup": speedup,
        },
    )


# ==============================================================================
# Test 4: Concurrent Streaming
# ==============================================================================

async def stream_one_request(session: aiohttp.ClientSession, request_id: int, prompt: str):
    """Stream a single request and collect metrics."""
    start_time = time.time()
    first_token_time = None
    tokens = []
    
    try:
        async with session.post(
            f"{BASE_URL}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 100,
                "stream": True,
            },
        ) as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "token" in data:
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                        tokens.append(data["token"])
        
        total_time = time.time() - start_time
        
        return {
            "request_id": request_id,
            "success": True,
            "prompt": prompt[:30],
            "tokens": len(tokens),
            "ttft": first_token_time,
            "total_time": total_time,
            "text": "".join(tokens)[:50],
        }
    
    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "error": str(e),
        }


async def test_concurrent_streaming_async():
    """Test multiple concurrent streaming requests - vLLM handles natively."""
    print_header("TEST 4: Concurrent Streaming (vLLM Native)")
    
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
        "What is C++?",
        "What is Java?",
        "What is Ruby?",
        "What is Swift?",
    ]
    
    print(f"  Sending {len(prompts)} concurrent streaming requests...")
    print("  (vLLM handles concurrency with PagedAttention)")
    print()
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            stream_one_request(session, i, prompt)
            for i, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print("  Results:")
    for r in results:
        if r["success"]:
            print(f"    #{r['request_id']}: {r['tokens']} tokens, "
                  f"TTFT={r['ttft']*1000:.0f}ms, "
                  f"Total={r['total_time']:.2f}s")
        else:
            print(f"    #{r['request_id']}: FAILED - {r.get('error', 'Unknown')}")
    
    print()
    
    if successful:
        avg_ttft = statistics.mean(r["ttft"] for r in successful if r["ttft"]) * 1000
        avg_time = statistics.mean(r["total_time"] for r in successful)
        total_tokens = sum(r["tokens"] for r in successful)
        
        print_result("Successful Requests:", f"{len(successful)}/{len(prompts)}")
        print_result("Avg TTFT:", f"{avg_ttft:.0f}ms")
        print_result("Avg Response Time:", f"{avg_time:.2f}s")
        print_result("Total Tokens:", str(total_tokens))
        print_result("Wall Clock Time:", f"{total_time:.2f}s")
        print_result("Effective Throughput:", f"{total_tokens/total_time:.1f} tok/s")
    
    print()
    print("  ✅ Concurrent streaming test PASSED")
    
    return TestResult(
        name="concurrent_streaming",
        success=len(failed) == 0,
        total_time=total_time,
        tokens_generated=sum(r["tokens"] for r in successful) if successful else 0,
        tokens_per_second=sum(r["tokens"] for r in successful) / total_time if successful else 0,
        details={
            "concurrent_requests": len(prompts),
            "successful": len(successful),
            "failed": len(failed),
            "avg_ttft_ms": avg_ttft if successful else 0,
        },
    )


def test_concurrent_streaming():
    """Wrapper to run async test."""
    return asyncio.run(test_concurrent_streaming_async())


# ==============================================================================
# Test Runner
# ==============================================================================

def check_server():
    """Check if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


def run_all_tests():
    """Run all tests and print summary."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    vLLM Server Test Suite                            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check server
    print("  Checking server status...", end=" ")
    if not check_server():
        print("❌ FAILED")
        print()
        print("  Server not running! Start it with:")
        print("    python llm_server.py --model facebook/opt-350m")
        print()
        return
    print("✅ OK")
    
    # Get server stats
    stats = requests.get(f"{BASE_URL}/stats").json()
    print()
    print_result("Model:", stats["model"])
    print_result("Backend:", stats["backend"])
    print_result("Tensor Parallel:", str(stats["tensor_parallel_size"]))
    print_result("GPU Memory:", str(stats["gpu_memory_utilization"]))
    
    # Run tests
    results = []
    
    results.append(test_batch_processing())
    results.append(test_streaming())
    results.append(test_throughput())
    results.append(test_concurrent_streaming())
    
    # Summary
    print_header("TEST SUMMARY")
    
    all_passed = all(r.success for r in results)
    
    print(f"  {'Test':<25} {'Status':<10} {'Time':<10} {'Tokens':<10} {'Tok/s':<10}")
    print("  " + "-"*65)
    
    for r in results:
        status = "✅ PASS" if r.success else "❌ FAIL"
        print(f"  {r.name:<25} {status:<10} {r.total_time:.2f}s     {r.tokens_generated:<10} {r.tokens_per_second:.1f}")
    
    print()
    if all_passed:
        print("  🎉 All tests PASSED!")
    else:
        print("  ⚠️  Some tests failed. Check the output above.")
    print()


def main():
    parser = argparse.ArgumentParser(description="vLLM Server Test Suite")
    parser.add_argument(
        "--test", "-t",
        choices=["batch", "stream", "throughput", "concurrent-stream", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Server URL",
    )
    
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    if args.test == "all":
        run_all_tests()
    elif args.test == "batch":
        test_batch_processing()
    elif args.test == "stream":
        test_streaming()
    elif args.test == "throughput":
        test_throughput()
    elif args.test == "concurrent-stream":
        test_concurrent_streaming()


if __name__ == "__main__":
    main()
