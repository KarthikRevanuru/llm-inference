#!/usr/bin/env python3
"""
vLLM Load Testing Suite

Tests:
1. Batching Efficiency - Proves continuous batching is working
2. Load Testing - Throughput under increasing concurrent load
3. Staggered Requests - Requests with configurable delays (simulates real traffic)

Usage:
    python load_test.py --test efficiency
    python load_test.py --test load --max-concurrent 32
    python load_test.py --test staggered --delay 0.5 --count 10
    python load_test.py --test all
"""

import asyncio
import aiohttp
import requests
import time
import json
import argparse
import statistics
from dataclasses import dataclass, field
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live


BASE_URL = "http://localhost:8000"
console = Console()


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class RequestResult:
    """Result from a single request."""
    request_id: int
    success: bool
    prompt: str
    answer: str = ""  # Generated text
    tokens: int = 0
    ttft: float = 0.0  # Time to first token
    total_time: float = 0.0
    start_time: float = 0.0  # Absolute start time (for overlap calculation)
    end_time: float = 0.0    # Absolute end time (for overlap calculation)
    error: Optional[str] = None


@dataclass
class LoadTestResult:
    """Aggregated results from a load test."""
    concurrent_level: int
    total_requests: int
    successful: int
    failed: int
    total_tokens: int
    wall_clock_time: float
    avg_ttft: float
    p50_ttft: float
    p95_ttft: float
    p99_ttft: float
    throughput_tps: float  # tokens per second
    requests_per_second: float


# ==============================================================================
# Core Request Function
# ==============================================================================

async def send_streaming_request(
    session: aiohttp.ClientSession,
    request_id: int,
    prompt: str,
    max_tokens: int = 50,
) -> RequestResult:
    """Send a single streaming request and collect metrics."""
    start_time = time.time()
    first_token_time = None
    tokens = []
    
    try:
        async with session.post(
            f"{BASE_URL}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "token" in data:
                            if first_token_time is None:
                                first_token_time = time.time() - start_time
                            tokens.append(data["token"])
                    except json.JSONDecodeError:
                        pass
        
        end_time = time.time()
        total_time = end_time - start_time
        generated_text = "".join(tokens)
        
        return RequestResult(
            request_id=request_id,
            success=True,
            prompt=prompt,
            answer=generated_text,
            tokens=len(tokens),
            ttft=first_token_time or 0,
            total_time=total_time,
            start_time=start_time,
            end_time=end_time,
        )
    
    except Exception as e:
        end_time = time.time()
        return RequestResult(
            request_id=request_id,
            success=False,
            prompt=prompt,
            answer="",
            error=str(e),
            total_time=end_time - start_time,
            start_time=start_time,
            end_time=end_time,
        )
            end_time=end_time,
        )


# ==============================================================================
# Test 1: Batching Efficiency
# ==============================================================================

async def test_batching_efficiency():
    """
    Prove that continuous batching is working by comparing:
    - Sequential execution time (baseline)
    - Concurrent execution time (batched)
    
    If efficiency > 2x, batching is working.
    """
    console.print("\n[bold cyan]═══ TEST: Batching Efficiency ═══[/bold cyan]\n")
    
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
    ]
    max_tokens = 30
    
    # ============ Sequential Baseline ============
    console.print("[dim]Running sequential baseline (4 requests, one at a time)...[/dim]")
    
    seq_times = []
    seq_tokens = 0
    
    for i, prompt in enumerate(prompts):
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "stream": False},
            timeout=60,
        )
        elapsed = time.time() - start
        seq_times.append(elapsed)
        result = response.json()
        seq_tokens += result.get("output_tokens", 0)
        console.print(f"  Request {i+1}: {elapsed:.2f}s")
    
    seq_total = sum(seq_times)
    
    # ============ Concurrent Batched ============
    console.print("\n[dim]Running concurrent requests (4 requests simultaneously)...[/dim]")
    
    async with aiohttp.ClientSession() as session:
        start = time.time()
        tasks = [
            send_streaming_request(session, i, prompt, max_tokens)
            for i, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        concurrent_total = time.time() - start
    
    concurrent_tokens = sum(r.tokens for r in results if r.success)
    
    # ============ Calculate Efficiency ============
    efficiency = seq_total / concurrent_total if concurrent_total > 0 else 0
    
    # Display results
    table = Table(title="Batching Efficiency Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Sequential", style="yellow")
    table.add_column("Concurrent", style="green")
    
    table.add_row("Total Time", f"{seq_total:.2f}s", f"{concurrent_total:.2f}s")
    table.add_row("Total Tokens", str(seq_tokens), str(concurrent_tokens))
    table.add_row("Throughput", f"{seq_tokens/seq_total:.1f} tok/s", f"{concurrent_tokens/concurrent_total:.1f} tok/s")
    
    console.print(table)
    
    # Verdict
    console.print()
    if efficiency >= 2.0:
        console.print(f"[bold green]✅ Batching Efficiency: {efficiency:.1f}x[/bold green]")
        console.print("[green]   Continuous batching is WORKING![/green]")
    elif efficiency >= 1.5:
        console.print(f"[bold yellow]⚠️  Batching Efficiency: {efficiency:.1f}x[/bold yellow]")
        console.print("[yellow]   Batching is partially working (may be GPU-bound)[/yellow]")
    else:
        console.print(f"[bold red]❌ Batching Efficiency: {efficiency:.1f}x[/bold red]")
        console.print("[red]   Batching does not appear to be working![/red]")
    
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
    results: List[RequestResult] = []
    
    async def limited_request(idx: int) -> RequestResult:
        async with semaphore:
            prompt = prompts[idx % len(prompts)]
            async with aiohttp.ClientSession() as session:
                return await send_streaming_request(session, idx, prompt, max_tokens=30)
    
    start_time = time.time()
    tasks = [limited_request(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    wall_clock_time = time.time() - start_time
    
    # Calculate metrics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    ttfts = [r.ttft for r in successful if r.ttft > 0]
    ttfts.sort()
    
    total_tokens = sum(r.tokens for r in successful)
    
    return LoadTestResult(
        concurrent_level=concurrent_level,
        total_requests=num_requests,
        successful=len(successful),
        failed=len(failed),
        total_tokens=total_tokens,
        wall_clock_time=wall_clock_time,
        avg_ttft=statistics.mean(ttfts) if ttfts else 0,
        p50_ttft=ttfts[len(ttfts)//2] if ttfts else 0,
        p95_ttft=ttfts[int(len(ttfts)*0.95)] if len(ttfts) > 1 else (ttfts[0] if ttfts else 0),
        p99_ttft=ttfts[int(len(ttfts)*0.99)] if len(ttfts) > 1 else (ttfts[0] if ttfts else 0),
        throughput_tps=total_tokens / wall_clock_time if wall_clock_time > 0 else 0,
        requests_per_second=len(successful) / wall_clock_time if wall_clock_time > 0 else 0,
    )


async def test_load(max_concurrent: int = 32, requests_per_level: int = 16):
    """
    Run load test with increasing concurrency levels.
    Shows how throughput and latency change under load.
    """
    console.print("\n[bold cyan]═══ TEST: Load Testing ═══[/bold cyan]\n")
    
    prompts = [
        "What is machine learning?",
        "Explain cloud computing.",
        "What is an API?",
        "Define artificial intelligence.",
        "What is a database?",
        "Explain containerization.",
        "What is microservices architecture?",
        "Define DevOps.",
    ]
    
    # Test at various concurrency levels
    levels = [1, 2, 4, 8, 16]
    if max_concurrent > 16:
        levels.extend([32])
    if max_concurrent > 32:
        levels.extend([64])
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
    table = Table(title="Load Test Results")
    table.add_column("Concurrent", style="cyan", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Throughput", style="green", justify="right")
    table.add_column("Avg TTFT", style="yellow", justify="right")
    table.add_column("P95 TTFT", style="yellow", justify="right")
    table.add_column("Req/s", style="magenta", justify="right")
    
    for r in results:
        table.add_row(
            str(r.concurrent_level),
            str(r.successful),
            str(r.failed) if r.failed else "-",
            f"{r.throughput_tps:.1f} tok/s",
            f"{r.avg_ttft*1000:.0f}ms",
            f"{r.p95_ttft*1000:.0f}ms",
            f"{r.requests_per_second:.1f}",
        )
    
    console.print(table)
    
    # Find optimal concurrency
    best = max(results, key=lambda r: r.throughput_tps)
    console.print(f"\n[bold green]📈 Peak Throughput: {best.throughput_tps:.1f} tok/s at concurrency={best.concurrent_level}[/bold green]")
    
    return results


# ==============================================================================
# Test 3: Staggered Requests (with Delay)
# ==============================================================================

async def test_staggered_requests(
    delay_seconds: float = 0.5,
    num_requests: int = 10,
    max_tokens: int = 50,
):
    """
    Send requests with a delay between each one.
    Simulates real-world traffic patterns where requests arrive over time.
    
    This tests how vLLM handles requests arriving at different times
    and whether it can dynamically add them to the running batch.
    """
    console.print("\n[bold cyan]═══ TEST: Staggered Requests ═══[/bold cyan]\n")
    console.print(f"  Sending {num_requests} requests with {delay_seconds}s delay between each")
    console.print()
    
    prompts = [
        "What is Python?",
        "Explain JavaScript.",
        "What is Rust?",
        "Define Go language.",
        "What is TypeScript?",
        "Explain Kotlin.",
        "What is Swift?",
        "Define Ruby.",
        "What is C++?",
        "Explain Java.",
    ]
    
    results: List[RequestResult] = []
    
    async def send_with_delay(idx: int, delay: float):
        """Send request after specified delay."""
        await asyncio.sleep(delay)
        console.print(f"  [dim]→ Request {idx+1} sent at t={delay:.1f}s[/dim]")
        
        async with aiohttp.ClientSession() as session:
            result = await send_streaming_request(
                session, idx, prompts[idx % len(prompts)], max_tokens
            )
            return result
    
    global_start = time.time()
    
    # Create tasks with staggered delays
    tasks = [
        send_with_delay(i, i * delay_seconds)
        for i in range(num_requests)
    ]
    
    # Run all concurrently (they'll start at staggered times due to sleep)
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - global_start
    
    console.print()
    
    # Calculate metrics
    successful = [r for r in results if r.success]
    total_tokens = sum(r.tokens for r in successful)
    
    ttfts = [r.ttft for r in successful if r.ttft > 0]
    avg_ttft = statistics.mean(ttfts) if ttfts else 0
    
    # Calculate overlaps for each request
    def get_overlapping_requests(target_idx: int) -> List[int]:
        """Find which other requests were running concurrently with target."""
        target = results[target_idx]
        if not target.success:
            return []
        
        overlapping = []
        for i, r in enumerate(results):
            if i == target_idx or not r.success:
                continue
            # Two requests overlap if one starts before the other ends
            if r.start_time < target.end_time and r.end_time > target.start_time:
                overlapping.append(i + 1)  # 1-indexed for display
        return overlapping
    
    # Display results with end time and overlaps
    table = Table(title="Staggered Requests Results")
    table.add_column("Req", justify="right")
    table.add_column("Sent At", justify="right")
    table.add_column("End At", justify="right", style="cyan")
    table.add_column("TTFT", justify="right", style="yellow")
    table.add_column("Duration", justify="right")
    table.add_column("Tokens", justify="right", style="green")
    table.add_column("Overlaps With", style="magenta")
    table.add_column("Status")
    
    for i, r in enumerate(results):
        sent_at = r.start_time - global_start if r.start_time else i * delay_seconds
        end_at = r.end_time - global_start if r.end_time else 0
        overlaps = get_overlapping_requests(i)
        overlap_str = ",".join(map(str, overlaps[:5])) if overlaps else "-"
        if len(overlaps) > 5:
            overlap_str += f"...+{len(overlaps)-5}"
        
        status = "[green]✓[/green]" if r.success else f"[red]✗[/red]"
        table.add_row(
            str(i + 1),
            f"{sent_at:.2f}s",
            f"{end_at:.2f}s" if r.success else "-",
            f"{r.ttft*1000:.0f}ms" if r.success else "-",
            f"{r.total_time:.2f}s",
            str(r.tokens) if r.success else "-",
            overlap_str,
            status,
        )
    
    console.print(table)
    
    # Timeline visualization
    console.print()
    console.print("[bold]Timeline Visualization:[/bold]")
    console.print("  (each █ = 0.2s, ░ = waiting, █ = processing)")
    console.print()
    
    for i, r in enumerate(results):
        if not r.success:
            continue
        
        sent_at = r.start_time - global_start
        end_at = r.end_time - global_start
        
        # Create timeline bar
        scale = 0.2  # seconds per character
        wait_chars = int(sent_at / scale)
        active_chars = int((end_at - sent_at) / scale)
        
        timeline = "░" * wait_chars + "█" * max(1, active_chars)
        console.print(f"  Req {i+1:2d}: [{timeline}] {sent_at:.1f}s → {end_at:.1f}s")
    
    # Q&A Display
    console.print()
    console.print("[bold]Questions & Answers:[/bold]")
    console.print()
    
    for i, r in enumerate(results):
        if not r.success:
            console.print(f"  [bold cyan]Q{i+1}:[/bold cyan] {r.prompt}")
            console.print(f"  [bold red]A{i+1}:[/bold red] [Error: {r.error}]")
            console.print()
            continue
        
        # Truncate answer if too long
        answer = r.answer.strip()
        if len(answer) > 200:
            answer = answer[:200] + "..."
        
        console.print(f"  [bold cyan]Q{i+1}:[/bold cyan] {r.prompt}")
        console.print(f"  [bold green]A{i+1}:[/bold green] {answer}")
        console.print()
    
    # Summary
    console.print()
    console.print(f"[bold]Summary:[/bold]")
    console.print(f"  Total Wall Clock Time: {total_time:.2f}s")
    console.print(f"  Requests Succeeded: {len(successful)}/{num_requests}")
    console.print(f"  Total Tokens: {total_tokens}")
    console.print(f"  Avg TTFT: {avg_ttft*1000:.0f}ms")
    console.print(f"  Throughput: {total_tokens/total_time:.1f} tok/s")
    
    # Calculate average concurrent requests
    if successful:
        avg_overlaps = statistics.mean(len(get_overlapping_requests(i)) for i in range(len(results)) if results[i].success)
        console.print(f"  Avg Concurrent Requests: {avg_overlaps + 1:.1f}")
    
    # Analysis: Did late requests benefit from batching?
    if len(results) >= 4:
        early_ttft = statistics.mean([r.ttft for r in results[:2] if r.success and r.ttft])
        late_ttft = statistics.mean([r.ttft for r in results[-2:] if r.success and r.ttft])
        
        console.print()
        if late_ttft < early_ttft * 1.5:
            console.print("[green]✅ Late requests were batched with earlier ones (similar TTFT)[/green]")
        else:
            console.print("[yellow]⚠️  Late requests had higher TTFT (may have waited for batch slot)[/yellow]")
    
    return results


# ==============================================================================
# Main Entry Point
# ==============================================================================

def check_server():
    """Check if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


async def run_all_tests(args):
    """Run all tests."""
    console.print("\n[bold]Checking server...[/bold]", end=" ")
    if not check_server():
        console.print("[red]FAILED[/red]")
        console.print("\nServer not running! Start with:")
        console.print("  python llm_server.py --model facebook/opt-350m")
        return
    console.print("[green]OK[/green]")
    
    # Get server info
    stats = requests.get(f"{BASE_URL}/stats").json()
    console.print(f"\n[dim]Model: {stats['model']} | Backend: {stats['backend']}[/dim]")
    
    # Run tests
    await test_batching_efficiency()
    await test_load(max_concurrent=args.max_concurrent)
    await test_staggered_requests(delay_seconds=args.delay, num_requests=args.count)
    
    console.print("\n[bold green]✅ All tests completed![/bold green]\n")


def main():
    parser = argparse.ArgumentParser(description="vLLM Load Testing Suite")
    parser.add_argument(
        "--test", "-t",
        choices=["efficiency", "load", "staggered", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--max-concurrent", type=int, default=32, help="Max concurrency for load test")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between staggered requests")
    parser.add_argument("--count", type=int, default=10, help="Number of staggered requests")
    
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    console.print("""
[bold cyan]╔══════════════════════════════════════════════════════════════╗
║                  vLLM Load Testing Suite                      ║
╚══════════════════════════════════════════════════════════════╝[/bold cyan]
    """)
    
    if args.test == "all":
        asyncio.run(run_all_tests(args))
    elif args.test == "efficiency":
        asyncio.run(test_batching_efficiency())
    elif args.test == "load":
        asyncio.run(test_load(max_concurrent=args.max_concurrent))
    elif args.test == "staggered":
        asyncio.run(test_staggered_requests(delay_seconds=args.delay, num_requests=args.count))


if __name__ == "__main__":
    main()
