"""Concurrent request testing for Orpheus TTS API."""
import argparse
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import requests


class ConcurrentTester:
    """Test concurrent request handling and measure performance."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def single_request(self, request_id: int, prompt: str, voice: str) -> Dict:
        """
        Make a single TTS request and measure performance.
        
        Returns:
            Dictionary with timing metrics
        """
        url = f"{self.base_url}/v1/tts/generate"
        
        payload = {
            "prompt": prompt,
            "voice": voice
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            total_time = time.time() - start_time
            
            # Get metadata from headers
            duration = float(response.headers.get('X-Audio-Duration', 0))
            gen_time = float(response.headers.get('X-Generation-Time', 0))
            rtf = float(response.headers.get('X-Realtime-Factor', 0))
            
            return {
                'request_id': request_id,
                'success': True,
                'total_time': total_time,
                'audio_duration': duration,
                'generation_time': gen_time,
                'realtime_factor': rtf,
                'error': None
            }
            
        except Exception as e:
            return {
                'request_id': request_id,
                'success': False,
                'total_time': time.time() - start_time,
                'error': str(e)
            }
    
    def run_concurrent_test(
        self,
        num_requests: int,
        max_workers: int,
        prompts: List[str],
        voices: List[str]
    ):
        """
        Run concurrent requests and report statistics.
        
        Args:
            num_requests: Number of requests to make
            max_workers: Maximum concurrent workers
            prompts: List of prompts to cycle through
            voices: List of voices to cycle through
        """
        print(f"\n{'='*60}")
        print(f"CONCURRENT REQUEST TEST")
        print(f"{'='*60}")
        print(f"Total requests: {num_requests}")
        print(f"Max concurrent: {max_workers}")
        print(f"{'='*60}\n")
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            futures = []
            for i in range(num_requests):
                prompt = prompts[i % len(prompts)]
                voice = voices[i % len(voices)]
                future = executor.submit(self.single_request, i, prompt, voice)
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    print(f"✓ Request {result['request_id']}: "
                          f"{result['audio_duration']:.2f}s audio in "
                          f"{result['generation_time']:.2f}s "
                          f"(RTF: {result['realtime_factor']:.2f}x)")
                else:
                    print(f"✗ Request {result['request_id']}: {result['error']}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Successful: {len(successful)}/{num_requests}")
        print(f"Failed: {len(failed)}/{num_requests}")
        
        if successful:
            gen_times = [r['generation_time'] for r in successful]
            audio_durations = [r['audio_duration'] for r in successful]
            rtfs = [r['realtime_factor'] for r in successful]
            
            print(f"\nGeneration Time Statistics:")
            print(f"  Mean: {statistics.mean(gen_times):.2f}s")
            print(f"  Median: {statistics.median(gen_times):.2f}s")
            print(f"  Min: {min(gen_times):.2f}s")
            print(f"  Max: {max(gen_times):.2f}s")
            
            print(f"\nRealtime Factor Statistics:")
            print(f"  Mean: {statistics.mean(rtfs):.2f}x")
            print(f"  Median: {statistics.median(rtfs):.2f}x")
            print(f"  Min: {min(rtfs):.2f}x")
            print(f"  Max: {max(rtfs):.2f}x")
            
            total_audio = sum(audio_durations)
            print(f"\nThroughput:")
            print(f"  Total audio generated: {total_audio:.2f}s")
            print(f"  Audio per second: {total_audio/total_time:.2f}s/s")
            print(f"  Requests per second: {len(successful)/total_time:.2f} req/s")
        
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Concurrent TTS Testing")
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--num-requests', type=int, default=5, help='Number of requests')
    parser.add_argument('--concurrent', type=int, default=5, help='Max concurrent requests')
    
    args = parser.parse_args()
    
    # Test prompts of varying lengths
    prompts = [
        "Hello, this is a short test.",
        "The weather today is absolutely beautiful, with clear blue skies and a gentle breeze.",
        "Artificial intelligence is transforming the way we interact with technology, making it more natural and intuitive.",
        "In the realm of text-to-speech synthesis, achieving human-like quality has been a long-standing challenge that recent advances in deep learning are finally addressing.",
    ]
    
    # Test different voices
    voices = ["tara", "leah", "jess", "leo"]
    
    tester = ConcurrentTester(args.url)
    tester.run_concurrent_test(args.num_requests, args.concurrent, prompts, voices)


if __name__ == '__main__':
    main()
