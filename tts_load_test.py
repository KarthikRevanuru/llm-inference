#!/usr/bin/env python3
"""
Orpheus TTS Load Testing Suite

Tests:
1. Batching Efficiency - Proves CUDA streams + async are working
2. Load Testing - Throughput under increasing concurrent load  
3. Staggered Requests - Requests with delays (simulates real traffic)
4. Batch Testing - Bulk audio generation (high throughput mode)

Usage:
    python tts_load_test.py --test efficiency
    python tts_load_test.py --test load --max-concurrent 16
    python tts_load_test.py --test staggered --delay 0.5 --count 10
    python tts_load_test.py --test batch --batch-size 20
    python tts_load_test.py --test all
"""

import asyncio
import aiohttp
import requests
import time
import json
import argparse
import statistics
import base64
from dataclasses import dataclass
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


BASE_URL = "http://localhost:8000"
console = Console()


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class TTSResult:
    """Result from a single TTS request."""
    request_id: int
    success: bool
    prompt: str
    audio_duration: float = 0.0  # seconds of audio generated
    audio_bytes: int = 0  # total bytes received
    ttfb: float = 0.0  # Time to first byte (first audio chunk)
    total_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    chunks: int = 0  # number of audio chunks received
    error: Optional[str] = None


@dataclass
class LoadTestResult:
    """Aggregated results from a load test."""
    concurrent_level: int
    total_requests: int
    successful: int
    failed: int
    total_audio_seconds: float
    wall_clock_time: float
    avg_ttfb: float
    p50_ttfb: float
    p95_ttfb: float
    realtime_factor: float  # audio_seconds / wall_clock_time
    requests_per_second: float


# ==============================================================================
# Core Request Function
# ==============================================================================

async def send_tts_streaming_request(
    session: aiohttp.ClientSession,
    request_id: int,
    prompt: str,
    voice: str = "tara",
) -> TTSResult:
    """Send a single streaming TTS request and collect metrics."""
    start_time = time.time()
    first_chunk_time = None
    total_bytes = 0
    chunk_count = 0
    
    try:
        async with session.post(
            f"{BASE_URL}/v1/tts/stream",
            json={
                "prompt": prompt,
                "voice": voice,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "audio_chunk" in line or isinstance(data, str):
                            # Base64 encoded audio chunk
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - start_time
                            if isinstance(data, str):
                                audio_bytes = base64.b64decode(data)
                                total_bytes += len(audio_bytes)
                            chunk_count += 1
                    except (json.JSONDecodeError, Exception):
                        pass
                elif line.startswith("event: audio_chunk"):
                    pass  # Next data line will have the chunk
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Estimate audio duration: 24kHz, 16-bit mono = 48000 bytes/sec
        audio_duration = total_bytes / 48000.0
        
        return TTSResult(
            request_id=request_id,
            success=True,
            prompt=prompt,
            audio_duration=audio_duration,
            audio_bytes=total_bytes,
            ttfb=first_chunk_time or 0,
            total_time=total_time,
            start_time=start_time,
            end_time=end_time,
            chunks=chunk_count,
        )
    
    except Exception as e:
        end_time = time.time()
        return TTSResult(
            request_id=request_id,
            success=False,
            prompt=prompt,
            error=str(e),
            total_time=end_time - start_time,
            start_time=start_time,
            end_time=end_time,
        )


async def send_tts_non_streaming_request(
    session: aiohttp.ClientSession,
    request_id: int,
    prompt: str,
    voice: str = "tara",
) -> TTSResult:
    """Send a single non-streaming TTS request."""
    start_time = time.time()
    
    try:
        async with session.post(
            f"{BASE_URL}/v1/tts/generate",
            json={
                "prompt": prompt,
                "voice": voice,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            audio_data = await response.read()
            end_time = time.time()
            
            # Get duration from headers if available
            audio_duration = float(response.headers.get("X-Audio-Duration", 0))
            if audio_duration == 0:
                audio_duration = len(audio_data) / 48000.0  # Estimate
            
            return TTSResult(
                request_id=request_id,
                success=True,
                prompt=prompt,
                audio_duration=audio_duration,
                audio_bytes=len(audio_data),
                ttfb=end_time - start_time,  # Non-streaming: TTFB = total time
                total_time=end_time - start_time,
                start_time=start_time,
                end_time=end_time,
                chunks=1,
            )
    
    except Exception as e:
        end_time = time.time()
        return TTSResult(
            request_id=request_id,
            success=False,
            prompt=prompt,
            error=str(e),
            total_time=end_time - start_time,
            start_time=start_time,
            end_time=end_time,
        )


# ==============================================================================
# Test 1: Batching Efficiency
# ==============================================================================

async def test_batching_efficiency():
    """
    Prove that CUDA streams + async SNAC are working by comparing:
    - Sequential execution time (baseline)
    - Concurrent execution time (with CUDA streams)
    
    If efficiency > 2x, parallelization is working.
    """
    console.print("\n[bold cyan]═══ TEST: TTS Batching Efficiency ═══[/bold cyan]\n")
    
    prompts = [
        "Hello, how are you today?",
        "The weather is beautiful outside.",
        "I love programming in Python.",
        "Machine learning is fascinating.",
    ]
    
    # ============ Sequential Baseline ============
    console.print("[dim]Running sequential baseline (4 requests, one at a time)...[/dim]")
    
    seq_times = []
    seq_audio = 0.0
    
    async with aiohttp.ClientSession() as session:
        for i, prompt in enumerate(prompts):
            start = time.time()
            result = await send_tts_non_streaming_request(session, i, prompt)
            elapsed = time.time() - start
            seq_times.append(elapsed)
            seq_audio += result.audio_duration
            console.print(f"  Request {i+1}: {elapsed:.2f}s, {result.audio_duration:.2f}s audio")
    
    seq_total = sum(seq_times)
    
    # ============ Concurrent (with CUDA streams) ============
    console.print("\n[dim]Running concurrent requests (4 requests simultaneously)...[/dim]")
    
    async with aiohttp.ClientSession() as session:
        start = time.time()
        tasks = [
            send_tts_non_streaming_request(session, i, prompt)
            for i, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        concurrent_total = time.time() - start
    
    concurrent_audio = sum(r.audio_duration for r in results if r.success)
    
    # ============ Calculate Efficiency ============
    efficiency = seq_total / concurrent_total if concurrent_total > 0 else 0
    
    # Display results
    table = Table(title="TTS Batching Efficiency Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Sequential", style="yellow")
    table.add_column("Concurrent", style="green")
    
    table.add_row("Total Time", f"{seq_total:.2f}s", f"{concurrent_total:.2f}s")
    table.add_row("Audio Generated", f"{seq_audio:.2f}s", f"{concurrent_audio:.2f}s")
    table.add_row("Realtime Factor", f"{seq_audio/seq_total:.1f}x", f"{concurrent_audio/concurrent_total:.1f}x")
    
    console.print(table)
    
    # Verdict
    console.print()
    if efficiency >= 2.0:
        console.print(f"[bold green]✅ Batching Efficiency: {efficiency:.1f}x[/bold green]")
        console.print("[green]   CUDA streams + async SNAC are WORKING![/green]")
    elif efficiency >= 1.5:
        console.print(f"[bold yellow]⚠️  Batching Efficiency: {efficiency:.1f}x[/bold yellow]")
        console.print("[yellow]   Partial parallelization (may be GPU-bound)[/yellow]")
    else:
        console.print(f"[bold red]❌ Batching Efficiency: {efficiency:.1f}x[/bold red]")
        console.print("[red]   Parallelization does not appear to be working![/red]")
    
    return efficiency


# ==============================================================================
# Test 2: Load Testing
# ==============================================================================

async def run_load_at_level(
    concurrent_level: int,
    num_requests: int,
    prompts: List[str],
) -> LoadTestResult:
    """Run load test at a specific concurrency level."""
    
    semaphore = asyncio.Semaphore(concurrent_level)
    results: List[TTSResult] = []
    
    async def limited_request(idx: int) -> TTSResult:
        async with semaphore:
            prompt = prompts[idx % len(prompts)]
            async with aiohttp.ClientSession() as session:
                return await send_tts_non_streaming_request(session, idx, prompt)
    
    start_time = time.time()
    tasks = [limited_request(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    wall_clock_time = time.time() - start_time
    
    # Calculate metrics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    ttfbs = [r.ttfb for r in successful if r.ttfb > 0]
    ttfbs.sort()
    
    total_audio = sum(r.audio_duration for r in successful)
    
    return LoadTestResult(
        concurrent_level=concurrent_level,
        total_requests=num_requests,
        successful=len(successful),
        failed=len(failed),
        total_audio_seconds=total_audio,
        wall_clock_time=wall_clock_time,
        avg_ttfb=statistics.mean(ttfbs) if ttfbs else 0,
        p50_ttfb=ttfbs[len(ttfbs)//2] if ttfbs else 0,
        p95_ttfb=ttfbs[int(len(ttfbs)*0.95)] if len(ttfbs) > 1 else (ttfbs[0] if ttfbs else 0),
        realtime_factor=total_audio / wall_clock_time if wall_clock_time > 0 else 0,
        requests_per_second=len(successful) / wall_clock_time if wall_clock_time > 0 else 0,
    )


async def test_load(max_concurrent: int = 16, requests_per_level: int = 8):
    """
    Run load test with increasing concurrency levels.
    Shows how throughput and latency change under load.
    """
    console.print("\n[bold cyan]═══ TEST: TTS Load Testing ═══[/bold cyan]\n")
    
    prompts = [
        "Hello, how are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "Good morning, have a wonderful day ahead!",
        "Technology continues to evolve rapidly.",
        "Music brings joy to our lives.",
        "Learning new things keeps the mind sharp.",
    ]
    
    # Test at various concurrency levels
    levels = [1, 2, 4, 8]
    if max_concurrent > 8:
        levels.append(16)
    if max_concurrent > 16:
        levels.append(32)
    levels = [l for l in levels if l <= max_concurrent]
    
    results: List[LoadTestResult] = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running load tests...", total=len(levels))
        
        for level in levels:
            progress.update(task, description=f"Testing concurrency={level}...")
            result = await run_load_at_level(level, requests_per_level, prompts)
            results.append(result)
            progress.advance(task)
    
    # Display results
    table = Table(title="TTS Load Test Results")
    table.add_column("Concurrent", style="cyan", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("RT Factor", style="green", justify="right")
    table.add_column("Avg TTFB", style="yellow", justify="right")
    table.add_column("P95 TTFB", style="yellow", justify="right")
    table.add_column("Req/s", style="magenta", justify="right")
    
    for r in results:
        table.add_row(
            str(r.concurrent_level),
            str(r.successful),
            str(r.failed) if r.failed else "-",
            f"{r.realtime_factor:.1f}x",
            f"{r.avg_ttfb*1000:.0f}ms",
            f"{r.p95_ttfb*1000:.0f}ms",
            f"{r.requests_per_second:.1f}",
        )
    
    console.print(table)
    
    # Find optimal concurrency
    best = max(results, key=lambda r: r.realtime_factor)
    console.print(f"\n[bold green]📈 Peak Realtime Factor: {best.realtime_factor:.1f}x at concurrency={best.concurrent_level}[/bold green]")
    
    return results


# ==============================================================================
# Test 3: Batch Testing (Maximum Throughput)
# ==============================================================================

async def test_batch(batch_size: int = 20):
    """
    Batch test: Fire all requests simultaneously without rate limiting.
    Measures maximum throughput capacity of the server.
    """
    console.print("\n[bold cyan]═══ TEST: TTS Batch Processing ═══[/bold cyan]\n")
    console.print(f"  Sending {batch_size} requests simultaneously (no rate limit)")
    console.print()
    
    # Longer prompts to stress test
    prompts = [
        "Hello, how are you doing today? I hope you're having a wonderful time.",
        "The quick brown fox jumps over the lazy dog. This is a classic pangram.",
        "Artificial intelligence is transforming the world in remarkable ways.",
        "Python is a versatile programming language loved by developers worldwide.",
        "Good morning! Have a wonderful day ahead filled with joy and success.",
        "Technology continues to evolve rapidly, changing how we live and work.",
        "Music brings joy to our lives and connects people across cultures.",
        "Learning new things keeps the mind sharp and opens new opportunities.",
        "The weather today is absolutely beautiful. Perfect for a long walk.",
        "Thank you for your patience and understanding. It means a lot to me.",
    ]
    
    results: List[TTSResult] = []
    
    console.print("[dim]Firing all requests...[/dim]")
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_tts_non_streaming_request(session, i, prompts[i % len(prompts)])
            for i in range(batch_size)
        ]
        results = await asyncio.gather(*tasks)
    
    wall_clock_time = time.time() - start_time
    
    # Calculate metrics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    total_audio = sum(r.audio_duration for r in successful)
    total_bytes = sum(r.audio_bytes for r in successful)
    
    ttfbs = sorted([r.ttfb for r in successful if r.ttfb > 0])
    total_times = sorted([r.total_time for r in successful])
    
    # Display results table
    table = Table(title="Batch Processing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Successful", f"{len(successful)}/{batch_size}")
    table.add_row("Failed", str(len(failed)) if failed else "0")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Wall Clock Time", f"{wall_clock_time:.2f}s")
    table.add_row("Total Audio Generated", f"{total_audio:.2f}s")
    table.add_row("Total Bytes", f"{total_bytes / 1024 / 1024:.2f} MB")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Realtime Factor", f"[bold]{total_audio/wall_clock_time:.1f}x[/bold]")
    table.add_row("Throughput (req/s)", f"{len(successful)/wall_clock_time:.2f}")
    table.add_row("Throughput (audio s/s)", f"{total_audio/wall_clock_time:.2f}")
    table.add_row("─" * 20, "─" * 20)
    
    if ttfbs:
        table.add_row("Avg Request Time", f"{statistics.mean(total_times)*1000:.0f}ms")
        table.add_row("P50 Request Time", f"{total_times[len(total_times)//2]*1000:.0f}ms")
        table.add_row("P95 Request Time", f"{total_times[int(len(total_times)*0.95)]*1000:.0f}ms")
        table.add_row("Min Request Time", f"{min(total_times)*1000:.0f}ms")
        table.add_row("Max Request Time", f"{max(total_times)*1000:.0f}ms")
    
    console.print(table)
    
    # Show per-request breakdown if small batch
    if batch_size <= 10:
        console.print("\n[bold]Per-Request Breakdown:[/bold]")
        detail_table = Table()
        detail_table.add_column("Req", justify="right")
        detail_table.add_column("Time", justify="right")
        detail_table.add_column("Audio", justify="right", style="green")
        detail_table.add_column("Bytes", justify="right")
        detail_table.add_column("Status")
        
        for i, r in enumerate(results):
            status = "[green]✓[/green]" if r.success else f"[red]✗ {r.error}[/red]"
            detail_table.add_row(
                str(i + 1),
                f"{r.total_time:.2f}s" if r.success else "-",
                f"{r.audio_duration:.1f}s" if r.success else "-",
                f"{r.audio_bytes/1024:.1f}KB" if r.success else "-",
                status,
            )
        console.print(detail_table)
    
    # Verdict
    console.print()
    rt_factor = total_audio / wall_clock_time if wall_clock_time > 0 else 0
    if rt_factor >= 5.0:
        console.print(f"[bold green]🚀 Excellent! Realtime Factor: {rt_factor:.1f}x[/bold green]")
        console.print("[green]   Server is handling batch load very efficiently![/green]")
    elif rt_factor >= 2.0:
        console.print(f"[bold green]✅ Good! Realtime Factor: {rt_factor:.1f}x[/bold green]")
        console.print("[green]   Batch processing is working well.[/green]")
    elif rt_factor >= 1.0:
        console.print(f"[bold yellow]⚠️  Realtime Factor: {rt_factor:.1f}x[/bold yellow]")
        console.print("[yellow]   Server is keeping up but at capacity.[/yellow]")
    else:
        console.print(f"[bold red]❌ Realtime Factor: {rt_factor:.1f}x[/bold red]")
        console.print("[red]   Server is falling behind realtime![/red]")
    
    if failed:
        console.print(f"\n[red]⚠️  {len(failed)} requests failed![/red]")
        for r in failed[:3]:
            console.print(f"  [dim]Error: {r.error}[/dim]")
    
    return results


# ==============================================================================
# Test 4: Staggered Requests
# ==============================================================================

async def test_staggered_requests(
    delay_seconds: float = 0.5,
    num_requests: int = 10,
):
    """
    Send requests with a delay between each one.
    Simulates real-world traffic patterns.
    """
    console.print("\n[bold cyan]═══ TEST: Staggered TTS Requests ═══[/bold cyan]\n")
    console.print(f"  Sending {num_requests} requests with {delay_seconds}s delay between each")
    console.print()
    
    prompts = [
        "Hello, how are you?",
        "What time is it?",
        "Tell me a joke.",
        "Good morning!",
        "How is the weather?",
        "What is your name?",
        "Nice to meet you.",
        "Have a great day!",
        "Thank you so much.",
        "Goodbye for now.",
    ]
    
    results: List[TTSResult] = []
    
    async def send_with_delay(idx: int, delay: float):
        await asyncio.sleep(delay)
        console.print(f"  [dim]→ Request {idx+1} sent at t={delay:.1f}s[/dim]")
        
        async with aiohttp.ClientSession() as session:
            return await send_tts_streaming_request(
                session, idx, prompts[idx % len(prompts)]
            )
    
    global_start = time.time()
    
    tasks = [
        send_with_delay(i, i * delay_seconds)
        for i in range(num_requests)
    ]
    
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - global_start
    
    console.print()
    
    # Calculate metrics
    successful = [r for r in results if r.success]
    total_audio = sum(r.audio_duration for r in successful)
    
    ttfbs = [r.ttfb for r in successful if r.ttfb > 0]
    avg_ttfb = statistics.mean(ttfbs) if ttfbs else 0
    
    # Display results
    table = Table(title="Staggered TTS Requests Results")
    table.add_column("Req", justify="right")
    table.add_column("Sent At", justify="right")
    table.add_column("End At", justify="right", style="cyan")
    table.add_column("TTFB", justify="right", style="yellow")
    table.add_column("Duration", justify="right")
    table.add_column("Audio", justify="right", style="green")
    table.add_column("Status")
    
    for i, r in enumerate(results):
        sent_at = r.start_time - global_start if r.start_time else i * delay_seconds
        end_at = r.end_time - global_start if r.end_time else 0
        
        status = "[green]✓[/green]" if r.success else f"[red]✗[/red]"
        table.add_row(
            str(i + 1),
            f"{sent_at:.2f}s",
            f"{end_at:.2f}s" if r.success else "-",
            f"{r.ttfb*1000:.0f}ms" if r.success else "-",
            f"{r.total_time:.2f}s",
            f"{r.audio_duration:.1f}s" if r.success else "-",
            status,
        )
    
    console.print(table)
    
    # Summary
    console.print()
    console.print(f"[bold]Summary:[/bold]")
    console.print(f"  Total Wall Clock Time: {total_time:.2f}s")
    console.print(f"  Requests Succeeded: {len(successful)}/{num_requests}")
    console.print(f"  Total Audio Generated: {total_audio:.2f}s")
    console.print(f"  Avg TTFB: {avg_ttfb*1000:.0f}ms")
    console.print(f"  Realtime Factor: {total_audio/total_time:.1f}x")
    
    return results


# ==============================================================================
# Main Entry Point
# ==============================================================================

def check_server():
    """Check if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


async def run_all_tests(args):
    """Run all tests."""
    console.print("\n[bold]Checking server...[/bold]", end=" ")
    if not check_server():
        console.print("[red]FAILED[/red]")
        console.print("\nServer not running! Start with:")
        console.print("  python server_vllm.py")
        return
    console.print("[green]OK[/green]")
    
    # Get server info
    try:
        health = requests.get(f"{BASE_URL}/health").json()
        console.print(f"\n[dim]Model: {health.get('model', 'unknown')} | Mode: {health.get('mode', 'unknown')}[/dim]")
    except:
        pass
    
    # Run tests
    await test_batching_efficiency()
    await test_load(max_concurrent=args.max_concurrent)
    await test_batch(batch_size=args.batch_size)
    await test_staggered_requests(delay_seconds=args.delay, num_requests=args.count)
    
    console.print("\n[bold green]✅ All tests completed![/bold green]\n")


def main():
    parser = argparse.ArgumentParser(description="Orpheus TTS Load Testing Suite")
    parser.add_argument(
        "--test", "-t",
        choices=["efficiency", "load", "batch", "staggered", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Max concurrency for load test")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of requests for batch test")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between staggered requests")
    parser.add_argument("--count", type=int, default=10, help="Number of staggered requests")
    
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    console.print("""
[bold cyan]╔══════════════════════════════════════════════════════════════╗
║               Orpheus TTS Load Testing Suite                 ║
╚══════════════════════════════════════════════════════════════╝[/bold cyan]
    """)
    
    if args.test == "all":
        asyncio.run(run_all_tests(args))
    elif args.test == "efficiency":
        asyncio.run(test_batching_efficiency())
    elif args.test == "load":
        asyncio.run(test_load(max_concurrent=args.max_concurrent))
    elif args.test == "batch":
        asyncio.run(test_batch(batch_size=args.batch_size))
    elif args.test == "staggered":
        asyncio.run(test_staggered_requests(delay_seconds=args.delay, num_requests=args.count))


if __name__ == "__main__":
    main()
