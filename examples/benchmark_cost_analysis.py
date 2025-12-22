"""
Benchmark & Cost Analysis Script for Orpheus-3B TTS
===================================================

This script runs two types of benchmarks to verify performance and cost assumptions:

1.  **Offline Batch Benchmark**:
    - Uses `vllm.LLM` directly (no server).
    - Measures raw GPU throughput (tokens/second).
    - Best for calculating maximum theoretical capacity.

2.  **Online Streaming Benchmark**:
    - Hits your running API (`http://localhost:8000`).
    - Measures Time-To-First-Byte (TTFB) and Real-Time Factor (RTF).
    - Best for verifying user experience and server overhead.

Usage:
    # Run Offline Benchmark (Requires GPU)
    python benchmark_cost_analysis.py --mode offline --model canopylabs/orpheus-3b-0.1-ft

    # Run Online Benchmark (Requires Server Running)
    python benchmark_cost_analysis.py --mode online --url http://localhost:8000 --concurrent 10
"""

import argparse
import time
import statistics
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants for Cost Calculation
AUDIO_TOKENS_PER_SEC = 50  # Orpheus generates ~50 tokens for 1 sec of audio
CHARS_PER_AUDIO_HOUR = 45000  # Approx chars in 1 hour of speech

class CostCalculator:
    @staticmethod
    def print_report(tokens_per_sec: float, gpu_hourly_cost: float):
        """Calculates and prints a financial report."""
        
        # 1. Calculate Real-Time Factor (RTF)
        # How many seconds of audio can we generate in 1 wall-clock second?
        rtf = tokens_per_sec / AUDIO_TOKENS_PER_SEC
        
        # 2. Calculate Capacity
        audio_hours_per_hour = rtf
        
        # 3. Calculate Cost
        cost_per_audio_hour = gpu_hourly_cost / audio_hours_per_hour
        cost_per_1k_chars = cost_per_audio_hour / (CHARS_PER_AUDIO_HOUR / 1000)
        
        print(f"\n{'='*60}")
        print(f"💰 FINANCIAL REPORT (Based on ${gpu_hourly_cost:.2f}/hr GPU)")
        print(f"{'='*60}")
        print(f"Performance Metrics:")
        print(f"  - Throughput:       {tokens_per_sec:.2f} tokens/sec")
        print(f"  - Real-Time Factor: {rtf:.2f}x (1 sec work = {rtf:.2f} sec audio)")
        print(f"  - Capacity:         {audio_hours_per_hour:.2f} hours audio / hour")
        
        print(f"\nCost Analysis:")
        print(f"  - Cost per Audio Hour:   ${cost_per_audio_hour:.4f}")
        print(f"  - Cost per 1k Chars:     ${cost_per_1k_chars:.5f}")
        print(f"{'='*60}\n")

class OfflineBenchmark:
    def __init__(self, model_name: str, quantization: str = None):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            print("Error: vLLM not installed. Run `pip install vllm`.")
            exit(1)
            
        print(f"Loading model {model_name} (Quantization: {quantization})...")
        self.llm = LLM(model=model_name, quantization=quantization, dtype="auto")
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)

    def run(self, prompts: List[str]):
        print(f"Running Offline Batch Benchmark on {len(prompts)} prompts...")
        
        start_time = time.time()
        outputs = self.llm.generate(prompts, self.sampling_params)
        total_time = time.time() - start_time
        
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tokens_per_sec = total_tokens / total_time
        
        print(f"\nResults:")
        print(f"  - Processed {len(prompts)} prompts in {total_time:.2f}s")
        print(f"  - Total Tokens: {total_tokens}")
        print(f"  - Speed: {tokens_per_sec:.2f} tokens/sec")
        
        return tokens_per_sec

class OnlineBenchmark:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def single_request(self, prompt: str) -> Dict:
        url = f"{self.base_url}/v1/tts/generate"
        start_time = time.time()
        try:
            response = requests.post(url, json={"prompt": prompt, "voice": "tara"}, timeout=120)
            response.raise_for_status()
            duration = float(response.headers.get('X-Audio-Duration', 0))
            # Estimate tokens if header missing (duration * 50)
            tokens = duration * AUDIO_TOKENS_PER_SEC 
            return {"success": True, "tokens": tokens, "time": time.time() - start_time}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run(self, prompts: List[str], concurrency: int):
        print(f"Running Online Streaming Benchmark (Concurrency: {concurrency})...")
        
        total_tokens = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self.single_request, p) for p in prompts]
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    total_tokens += result['tokens']
                else:
                    print(f"Request failed: {result['error']}")

        total_time = time.time() - start_time
        tokens_per_sec = total_tokens / total_time
        
        print(f"\nResults:")
        print(f"  - Processed {len(prompts)} requests in {total_time:.2f}s")
        print(f"  - Total Estimated Tokens: {total_tokens:.0f}")
        print(f"  - Effective Speed: {tokens_per_sec:.2f} tokens/sec")
        
        return tokens_per_sec

def main():
    parser = argparse.ArgumentParser(description="Orpheus TTS Benchmark & Cost Analysis")
    parser.add_argument('--mode', choices=['offline', 'online'], required=True, help='Benchmark mode')
    parser.add_argument('--model', default='canopylabs/orpheus-3b-0.1-ft', help='Model name (offline only)')
    parser.add_argument('--quantization', default=None, help='Quantization (e.g., fp8) (offline only)')
    parser.add_argument('--url', default='http://localhost:8000', help='API URL (online only)')
    parser.add_argument('--concurrent', type=int, default=10, help='Concurrency (online only)')
    parser.add_argument('--gpu-cost', type=float, default=0.80, help='Hourly GPU cost (default: $0.80 for L4)')
    parser.add_argument('--num-prompts', type=int, default=20, help='Number of prompts to run')
    
    args = parser.parse_args()
    
    # Test Data
    base_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Text to speech technology has advanced significantly.",
        "This is a test of the emergency broadcast system.",
        "Deep learning models require significant compute power."
    ]
    # Multiply prompts to hit requested number
    prompts = (base_prompts * (args.num_prompts // len(base_prompts) + 1))[:args.num_prompts]

    if args.mode == 'offline':
        bench = OfflineBenchmark(args.model, args.quantization)
        tps = bench.run(prompts)
    else:
        bench = OnlineBenchmark(args.url)
        tps = bench.run(prompts, args.concurrent)

    CostCalculator.print_report(tps, args.gpu_cost)

if __name__ == '__main__':
    main()
