#!/usr/bin/env python3
"""
Orpheus TTS Testing & Benchmarking Tool
========================================

Consolidated tool combining all TTS testing capabilities:
- Single request generation (save audio file)
- Streaming with TTFB measurement
- Concurrent testing
- Batching efficiency (CUDA streams verification)
- Load testing (increasing concurrency)
- Batch testing (maximum throughput)
- Staggered requests (real traffic simulation)
- Cost analysis

Usage Examples:
    # Generate single audio file
    python tts_tool.py generate "Hello world" --output hello.wav

    # Test streaming with TTFB
    python tts_tool.py stream "Hello world" --output hello.wav

    # Run efficiency test (proves CUDA streams work)
    python tts_tool.py test efficiency

    # Run load test
    python tts_tool.py test load --max-concurrent 16

    # Run batch test (max throughput)
    python tts_tool.py test batch --batch-size 20

    # Run staggered test
    python tts_tool.py test staggered --delay 0.5 --count 10

    # Run all tests
    python tts_tool.py test all

    # Cost analysis
    python tts_tool.py cost --concurrent 10 --gpu-cost 0.80
"""

import asyncio
import aiohttp
import argparse
import base64
import json
import requests
import statistics
import time
import wave
from dataclasses import dataclass
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs):
            # Strip rich formatting
            text = str(args[0]) if args else ""
            import re
            text = re.sub(r'\[.*?\]', '', text)
            print(text)

BASE_URL = "http://localhost:8000"
console = Console()

# Constants
SAMPLE_RATE = 24000
AUDIO_TOKENS_PER_SEC = 50  # Orpheus generates ~50 tokens for 1 sec of audio
CHARS_PER_AUDIO_HOUR = 45000  # Approx chars in 1 hour of speech


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class TTSResult:
    """Result from a single TTS request."""
    request_id: int
    success: bool
    prompt: str
    audio_duration: float = 0.0
    audio_bytes: int = 0
    ttfb: float = 0.0
    total_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    chunks: int = 0
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
    realtime_factor: float
    requests_per_second: float


# ==============================================================================
# Single Request Functions
# ==============================================================================

def generate_tts_sync(prompt: str, voice: str = "tara", output_file: str = "output.wav") -> Dict:
    """Generate complete TTS audio (non-streaming, synchronous)."""
    url = f"{BASE_URL}/v1/tts/generate"
    
    console.print(f"[dim]Generating TTS for: '{prompt[:50]}...'[/dim]")
    console.print(f"[dim]Voice: {voice}[/dim]")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json={"prompt": prompt, "voice": voice}, timeout=120)
        response.raise_for_status()
        
        total_time = time.time() - start_time
        
        # Save audio file
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        # Get metadata from headers
        duration = float(response.headers.get('X-Audio-Duration', 0))
        gen_time = float(response.headers.get('X-Generation-Time', 0))
        rtf = float(response.headers.get('X-Realtime-Factor', 0))
        
        if duration == 0:
            duration = len(response.content) / 48000.0  # Estimate
        
        console.print(f"\n[green]✓ Complete![/green]")
        console.print(f"  Audio duration: {duration:.2f}s")
        console.print(f"  Generation time: {gen_time:.2f}s")
        console.print(f"  Realtime factor: {rtf:.2f}x")
        console.print(f"\n[cyan]💾 Saved to: {output_file}[/cyan]")
        
        return {"success": True, "duration": duration, "time": total_time, "file": output_file}
        
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return {"success": False, "error": str(e)}


def stream_tts_sync(prompt: str, voice: str = "tara", output_file: str = "output.wav") -> Dict:
    """Stream TTS audio with TTFB measurement."""
    url = f"{BASE_URL}/v1/tts/stream"
    
    console.print(f"[dim]Streaming TTS for: '{prompt[:50]}...'[/dim]")
    console.print(f"[dim]Voice: {voice}[/dim]")
    
    start_time = time.time()
    ttfb = None
    chunk_count = 0
    total_bytes = 0
    
    try:
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            
            event_type = None
            
            with requests.post(url, json={"prompt": prompt, "voice": voice}, stream=True, timeout=120) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    line = line.decode('utf-8')
                    
                    if line.startswith('event:'):
                        event_type = line.split(':', 1)[1].strip()
                        continue
                    
                    if line.startswith('data:'):
                        data = line.split(':', 1)[1].strip()
                        
                        if ttfb is None:
                            ttfb = time.time() - start_time
                            console.print(f"[yellow]⚡ TTFB: {ttfb*1000:.0f}ms[/yellow]")
                        
                        if event_type == 'audio_chunk' or event_type is None:
                            try:
                                audio_bytes = base64.b64decode(data)
                                wf.writeframes(audio_bytes)
                                total_bytes += len(audio_bytes)
                                chunk_count += 1
                                
                                if chunk_count % 10 == 0:
                                    console.print(f"  [dim]Received {chunk_count} chunks...[/dim]")
                            except Exception:
                                pass
                        
                        elif event_type == 'complete':
                            try:
                                metadata = json.loads(data)
                                console.print(f"\n[green]✓ Complete![/green]")
                                console.print(f"  Audio duration: {metadata.get('audio_duration_seconds', 0):.2f}s")
                                console.print(f"  Generation time: {metadata.get('generation_time_seconds', 0):.2f}s")
                                console.print(f"  Realtime factor: {metadata.get('realtime_factor', 0):.2f}x")
                            except json.JSONDecodeError:
                                console.print(f"\n[green]✓ Complete! ({chunk_count} chunks)[/green]")
        
        total_time = time.time() - start_time
        audio_duration = total_bytes / 48000.0
        
        console.print(f"\n[cyan]💾 Saved to: {output_file}[/cyan]")
        
        return {
            "success": True, 
            "duration": audio_duration, 
            "ttfb": ttfb,
            "time": total_time, 
            "chunks": chunk_count,
            "file": output_file
        }
        
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return {"success": False, "error": str(e)}


# ==============================================================================
# Async Request Functions
# ==============================================================================

async def send_tts_request_async(
    session: aiohttp.ClientSession,
    request_id: int,
    prompt: str,
    voice: str = "tara",
) -> TTSResult:
    """Send a single async TTS request."""
    start_time = time.time()
    
    try:
        async with session.post(
            f"{BASE_URL}/v1/tts/generate",
            json={"prompt": prompt, "voice": voice},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            audio_data = await response.read()
            end_time = time.time()
            
            audio_duration = float(response.headers.get('X-Audio-Duration', 0))
            if audio_duration == 0:
                audio_duration = len(audio_data) / 48000.0
            
            return TTSResult(
                request_id=request_id,
                success=True,
                prompt=prompt,
                audio_duration=audio_duration,
                audio_bytes=len(audio_data),
                ttfb=end_time - start_time,
                total_time=end_time - start_time,
                start_time=start_time,
                end_time=end_time,
                chunks=1,
            )
    
    except Exception as e:
        return TTSResult(
            request_id=request_id,
            success=False,
            prompt=prompt,
            error=str(e),
            total_time=time.time() - start_time,
            start_time=start_time,
            end_time=time.time(),
        )


async def send_tts_streaming_async(
    session: aiohttp.ClientSession,
    request_id: int,
    prompt: str,
    voice: str = "tara",
) -> TTSResult:
    """Send a single async streaming TTS request."""
    start_time = time.time()
    first_chunk_time = None
    total_bytes = 0
    chunk_count = 0
    
    try:
        async with session.post(
            f"{BASE_URL}/v1/tts/stream",
            json={"prompt": prompt, "voice": voice},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                    try:
                        data = json.loads(line[6:])
                        if isinstance(data, str):
                            audio_bytes = base64.b64decode(data)
                            total_bytes += len(audio_bytes)
                        chunk_count += 1
                    except:
                        pass
        
        end_time = time.time()
        audio_duration = total_bytes / 48000.0
        
        return TTSResult(
            request_id=request_id,
            success=True,
            prompt=prompt,
            audio_duration=audio_duration,
            audio_bytes=total_bytes,
            ttfb=first_chunk_time or 0,
            total_time=end_time - start_time,
            start_time=start_time,
            end_time=end_time,
            chunks=chunk_count,
        )
    
    except Exception as e:
        return TTSResult(
            request_id=request_id,
            success=False,
            prompt=prompt,
            error=str(e),
            total_time=time.time() - start_time,
            start_time=start_time,
            end_time=time.time(),
        )


# ==============================================================================
# Test Functions
# ==============================================================================

async def test_efficiency():
    """Test batching efficiency (CUDA streams verification)."""
    console.print("\n[bold cyan]═══ TEST: Batching Efficiency ═══[/bold cyan]\n")
    
    prompts = [
        "Hello, how are you today?",
        "The weather is beautiful outside.",
        "I love programming in Python.",
        "Machine learning is fascinating.",
    ]
    
    # Sequential
    console.print("[dim]Running sequential baseline...[/dim]")
    seq_times = []
    seq_audio = 0.0
    
    async with aiohttp.ClientSession() as session:
        for i, prompt in enumerate(prompts):
            start = time.time()
            result = await send_tts_request_async(session, i, prompt)
            elapsed = time.time() - start
            seq_times.append(elapsed)
            seq_audio += result.audio_duration
            console.print(f"  Request {i+1}: {elapsed:.2f}s, {result.audio_duration:.2f}s audio")
    
    seq_total = sum(seq_times)
    
    # Concurrent
    console.print("\n[dim]Running concurrent requests...[/dim]")
    async with aiohttp.ClientSession() as session:
        start = time.time()
        tasks = [send_tts_request_async(session, i, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        concurrent_total = time.time() - start
    
    concurrent_audio = sum(r.audio_duration for r in results if r.success)
    efficiency = seq_total / concurrent_total if concurrent_total > 0 else 0
    
    # Display
    if RICH_AVAILABLE:
        table = Table(title="Batching Efficiency Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Sequential", style="yellow")
        table.add_column("Concurrent", style="green")
        table.add_row("Total Time", f"{seq_total:.2f}s", f"{concurrent_total:.2f}s")
        table.add_row("Audio Generated", f"{seq_audio:.2f}s", f"{concurrent_audio:.2f}s")
        table.add_row("RT Factor", f"{seq_audio/seq_total:.1f}x", f"{concurrent_audio/concurrent_total:.1f}x")
        console.print(table)
    else:
        console.print(f"\nSequential: {seq_total:.2f}s, Concurrent: {concurrent_total:.2f}s")
    
    console.print()
    if efficiency >= 2.0:
        console.print(f"[bold green]✅ Efficiency: {efficiency:.1f}x - CUDA streams WORKING![/bold green]")
    elif efficiency >= 1.5:
        console.print(f"[bold yellow]⚠️  Efficiency: {efficiency:.1f}x - Partial parallelization[/bold yellow]")
    else:
        console.print(f"[bold red]❌ Efficiency: {efficiency:.1f}x - Not parallelized[/bold red]")
    
    return efficiency


async def run_load_level(concurrent: int, num_requests: int, prompts: List[str]) -> LoadTestResult:
    """Run load test at specific concurrency."""
    semaphore = asyncio.Semaphore(concurrent)
    
    async def limited_request(idx: int) -> TTSResult:
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await send_tts_request_async(session, idx, prompts[idx % len(prompts)])
    
    start = time.time()
    results = await asyncio.gather(*[limited_request(i) for i in range(num_requests)])
    wall_time = time.time() - start
    
    successful = [r for r in results if r.success]
    ttfbs = sorted([r.ttfb for r in successful if r.ttfb > 0])
    total_audio = sum(r.audio_duration for r in successful)
    
    return LoadTestResult(
        concurrent_level=concurrent,
        total_requests=num_requests,
        successful=len(successful),
        failed=len(results) - len(successful),
        total_audio_seconds=total_audio,
        wall_clock_time=wall_time,
        avg_ttfb=statistics.mean(ttfbs) if ttfbs else 0,
        p50_ttfb=ttfbs[len(ttfbs)//2] if ttfbs else 0,
        p95_ttfb=ttfbs[int(len(ttfbs)*0.95)] if len(ttfbs) > 1 else (ttfbs[0] if ttfbs else 0),
        realtime_factor=total_audio / wall_time if wall_time > 0 else 0,
        requests_per_second=len(successful) / wall_time if wall_time > 0 else 0,
    )


async def test_load(max_concurrent: int = 16, requests_per_level: int = 8):
    """Run load test with increasing concurrency."""
    console.print("\n[bold cyan]═══ TEST: Load Testing ═══[/bold cyan]\n")
    
    prompts = [
        "Hello, how are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
    ]
    
    levels = [l for l in [1, 2, 4, 8, 16, 32] if l <= max_concurrent]
    results = []
    
    for level in levels:
        console.print(f"[dim]Testing concurrency={level}...[/dim]")
        result = await run_load_level(level, requests_per_level, prompts)
        results.append(result)
    
    # Display
    if RICH_AVAILABLE:
        table = Table(title="Load Test Results")
        table.add_column("Concurrent", justify="right")
        table.add_column("Success", justify="right")
        table.add_column("RT Factor", style="green", justify="right")
        table.add_column("Avg TTFB", style="yellow", justify="right")
        table.add_column("Req/s", style="magenta", justify="right")
        
        for r in results:
            table.add_row(
                str(r.concurrent_level),
                f"{r.successful}/{r.total_requests}",
                f"{r.realtime_factor:.1f}x",
                f"{r.avg_ttfb*1000:.0f}ms",
                f"{r.requests_per_second:.1f}",
            )
        console.print(table)
    
    best = max(results, key=lambda r: r.realtime_factor)
    console.print(f"\n[bold green]📈 Peak: {best.realtime_factor:.1f}x at concurrency={best.concurrent_level}[/bold green]")
    
    return results


async def test_batch(batch_size: int = 20):
    """Batch test: max throughput."""
    console.print("\n[bold cyan]═══ TEST: Batch Processing ═══[/bold cyan]\n")
    console.print(f"[dim]Sending {batch_size} requests simultaneously...[/dim]")
    
    prompts = [
        "Hello, how are you doing today? I hope you're having a wonderful time.",
        "The quick brown fox jumps over the lazy dog. This is a classic pangram.",
        "Artificial intelligence is transforming the world in remarkable ways.",
        "Python is a versatile programming language loved by developers worldwide.",
    ]
    
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [send_tts_request_async(session, i, prompts[i % len(prompts)]) for i in range(batch_size)]
        results = await asyncio.gather(*tasks)
    wall_time = time.time() - start
    
    successful = [r for r in results if r.success]
    total_audio = sum(r.audio_duration for r in successful)
    rt_factor = total_audio / wall_time if wall_time > 0 else 0
    
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Wall Time: {wall_time:.2f}s")
    console.print(f"  Audio Generated: {total_audio:.2f}s")
    console.print(f"  Realtime Factor: [bold]{rt_factor:.1f}x[/bold]")
    console.print(f"  Throughput: {len(successful)/wall_time:.2f} req/s")
    
    if rt_factor >= 5.0:
        console.print(f"\n[bold green]🚀 Excellent throughput![/bold green]")
    elif rt_factor >= 2.0:
        console.print(f"\n[bold green]✅ Good throughput[/bold green]")
    elif rt_factor >= 1.0:
        console.print(f"\n[bold yellow]⚠️  At capacity[/bold yellow]")
    else:
        console.print(f"\n[bold red]❌ Below realtime[/bold red]")
    
    return results


async def test_staggered(delay: float = 0.5, count: int = 10):
    """Staggered requests test."""
    console.print("\n[bold cyan]═══ TEST: Staggered Requests ═══[/bold cyan]\n")
    console.print(f"[dim]Sending {count} requests with {delay}s delay...[/dim]")
    
    prompts = ["Hello!", "How are you?", "What time is it?", "Good morning!"]
    
    async def send_delayed(idx: int):
        await asyncio.sleep(idx * delay)
        console.print(f"  [dim]→ Request {idx+1} at t={idx*delay:.1f}s[/dim]")
        async with aiohttp.ClientSession() as session:
            return await send_tts_streaming_async(session, idx, prompts[idx % len(prompts)])
    
    start = time.time()
    results = await asyncio.gather(*[send_delayed(i) for i in range(count)])
    total_time = time.time() - start
    
    successful = [r for r in results if r.success]
    total_audio = sum(r.audio_duration for r in successful)
    ttfbs = [r.ttfb for r in successful if r.ttfb > 0]
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Wall Time: {total_time:.2f}s")
    console.print(f"  Success: {len(successful)}/{count}")
    console.print(f"  Avg TTFB: {statistics.mean(ttfbs)*1000:.0f}ms" if ttfbs else "  Avg TTFB: N/A")
    console.print(f"  RT Factor: {total_audio/total_time:.1f}x")
    
    return results


async def run_all_tests(args):
    """Run all tests."""
    await test_efficiency()
    await test_load(max_concurrent=args.max_concurrent)
    await test_batch(batch_size=args.batch_size)
    await test_staggered(delay=args.delay, count=args.count)
    console.print("\n[bold green]✅ All tests completed![/bold green]\n")


# ==============================================================================
# Cost Analysis
# ==============================================================================

def run_cost_analysis(concurrent: int = 10, gpu_cost: float = 0.80, num_prompts: int = 20):
    """Run cost analysis."""
    console.print("\n[bold cyan]═══ COST ANALYSIS ═══[/bold cyan]\n")
    
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Text to speech technology has advanced significantly.",
    ]
    prompts = (prompts * (num_prompts // len(prompts) + 1))[:num_prompts]
    
    total_tokens = 0
    start = time.time()
    
    def single_request(prompt):
        try:
            response = requests.post(
                f"{BASE_URL}/v1/tts/generate",
                json={"prompt": prompt, "voice": "tara"},
                timeout=120
            )
            duration = float(response.headers.get('X-Audio-Duration', 0))
            return duration * AUDIO_TOKENS_PER_SEC
        except:
            return 0
    
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(single_request, p) for p in prompts]
        for f in as_completed(futures):
            total_tokens += f.result()
    
    total_time = time.time() - start
    tps = total_tokens / total_time
    rtf = tps / AUDIO_TOKENS_PER_SEC
    audio_hours = rtf
    cost_per_hour = gpu_cost / audio_hours
    cost_per_1k = cost_per_hour / (CHARS_PER_AUDIO_HOUR / 1000)
    
    console.print(f"[bold]Performance:[/bold]")
    console.print(f"  Throughput: {tps:.2f} tokens/sec")
    console.print(f"  Realtime Factor: {rtf:.2f}x")
    console.print(f"  Capacity: {audio_hours:.2f} audio hours/hour")
    
    console.print(f"\n[bold]Cost Analysis (${gpu_cost:.2f}/hr GPU):[/bold]")
    console.print(f"  Cost per Audio Hour: ${cost_per_hour:.4f}")
    console.print(f"  Cost per 1k Chars: ${cost_per_1k:.5f}")


# ==============================================================================
# Health Check
# ==============================================================================

def check_server():
    """Check if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Orpheus TTS Testing & Benchmarking Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate audio (non-streaming)")
    gen_parser.add_argument("prompt", help="Text to convert to speech")
    gen_parser.add_argument("--voice", default="tara", help="Voice name")
    gen_parser.add_argument("--output", "-o", default="output.wav", help="Output file")
    
    # Stream command
    stream_parser = subparsers.add_parser("stream", help="Stream audio with TTFB")
    stream_parser.add_argument("prompt", help="Text to convert to speech")
    stream_parser.add_argument("--voice", default="tara", help="Voice name")
    stream_parser.add_argument("--output", "-o", default="output.wav", help="Output file")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "test_type",
        choices=["efficiency", "load", "batch", "staggered", "all"],
        help="Test type"
    )
    test_parser.add_argument("--max-concurrent", type=int, default=16)
    test_parser.add_argument("--batch-size", type=int, default=20)
    test_parser.add_argument("--delay", type=float, default=0.5)
    test_parser.add_argument("--count", type=int, default=10)
    
    # Cost command
    cost_parser = subparsers.add_parser("cost", help="Cost analysis")
    cost_parser.add_argument("--concurrent", type=int, default=10)
    cost_parser.add_argument("--gpu-cost", type=float, default=0.80)
    cost_parser.add_argument("--num-prompts", type=int, default=20)
    
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    console.print("""
[bold cyan]╔════════════════════════════════════════════════════════════╗
║          Orpheus TTS Testing & Benchmarking Tool           ║
╚════════════════════════════════════════════════════════════╝[/bold cyan]
    """)
    
    if not args.command:
        parser.print_help()
        return
    
    # Check server
    if args.command in ["generate", "stream", "test", "cost"]:
        console.print("[dim]Checking server...[/dim]", end=" ")
        if not check_server():
            console.print("[red]FAILED[/red]\nStart server: python server_vllm.py")
            return
        console.print("[green]OK[/green]\n")
    
    # Execute command
    if args.command == "generate":
        generate_tts_sync(args.prompt, args.voice, args.output)
    
    elif args.command == "stream":
        stream_tts_sync(args.prompt, args.voice, args.output)
    
    elif args.command == "test":
        if args.test_type == "efficiency":
            asyncio.run(test_efficiency())
        elif args.test_type == "load":
            asyncio.run(test_load(args.max_concurrent))
        elif args.test_type == "batch":
            asyncio.run(test_batch(args.batch_size))
        elif args.test_type == "staggered":
            asyncio.run(test_staggered(args.delay, args.count))
        elif args.test_type == "all":
            asyncio.run(run_all_tests(args))
    
    elif args.command == "cost":
        run_cost_analysis(args.concurrent, args.gpu_cost, args.num_prompts)


if __name__ == "__main__":
    main()
