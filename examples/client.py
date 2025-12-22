"""Example client for Orpheus TTS streaming API."""
import argparse
import base64
import time
import wave
import requests
import json
from typing import Optional


class OrpheusTTSClient:
    """Client for interacting with Orpheus TTS API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def stream_tts(
        self,
        prompt: str,
        voice: str = "tara",
        output_file: str = "output.wav",
        measure_ttfb: bool = False
    ):
        """
        Stream TTS audio and save to file.
        
        Args:
            prompt: Text to convert to speech
            voice: Voice name
            output_file: Output WAV file path
            measure_ttfb: Whether to measure and print TTFB
        """
        url = f"{self.base_url}/v1/tts/stream"
        
        payload = {
            "prompt": prompt,
            "voice": voice
        }
        
        print(f"Streaming TTS for: '{prompt[:50]}...'")
        print(f"Voice: {voice}")
        
        start_time = time.time()
        ttfb = None
        
        # Open WAV file for writing
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            
            chunk_count = 0
            
            # Stream response
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    # Parse SSE format
                    line = line.decode('utf-8')
                    
                    if line.startswith('event:'):
                        event_type = line.split(':', 1)[1].strip()
                        continue
                    
                    if line.startswith('data:'):
                        data = line.split(':', 1)[1].strip()
                        
                        # Record TTFB on first chunk
                        if ttfb is None:
                            ttfb = time.time() - start_time
                            if measure_ttfb:
                                print(f"⚡ TTFB: {ttfb:.3f}s")
                        
                        # Handle different event types
                        if event_type == 'audio_chunk':
                            # Decode base64 audio chunk
                            audio_bytes = base64.b64decode(data)
                            wf.writeframes(audio_bytes)
                            chunk_count += 1
                            
                            if chunk_count % 10 == 0:
                                print(f"  Received {chunk_count} chunks...")
                        
                        elif event_type == 'complete':
                            # Parse completion metadata
                            metadata = json.loads(data)
                            print(f"\n✓ Complete!")
                            print(f"  Audio duration: {metadata['audio_duration_seconds']:.2f}s")
                            print(f"  Generation time: {metadata['generation_time_seconds']:.2f}s")
                            print(f"  Realtime factor: {metadata['realtime_factor']:.2f}x")
                            print(f"  Total chunks: {metadata['total_chunks']}")
                        
                        elif event_type == 'error':
                            print(f"✗ Error: {data}")
                            return
        
        print(f"\n💾 Saved to: {output_file}")
    
    def generate_tts(
        self,
        prompt: str,
        voice: str = "tara",
        output_file: str = "output.wav"
    ):
        """
        Generate complete TTS audio (non-streaming).
        
        Args:
            prompt: Text to convert to speech
            voice: Voice name
            output_file: Output WAV file path
        """
        url = f"{self.base_url}/v1/tts/generate"
        
        payload = {
            "prompt": prompt,
            "voice": voice
        }
        
        print(f"Generating TTS for: '{prompt[:50]}...'")
        print(f"Voice: {voice}")
        
        start_time = time.time()
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Save audio file
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        # Print metadata from headers
        duration = float(response.headers.get('X-Audio-Duration', 0))
        gen_time = float(response.headers.get('X-Generation-Time', 0))
        rtf = float(response.headers.get('X-Realtime-Factor', 0))
        
        print(f"\n✓ Complete!")
        print(f"  Audio duration: {duration:.2f}s")
        print(f"  Generation time: {gen_time:.2f}s")
        print(f"  Realtime factor: {rtf:.2f}x")
        print(f"\n💾 Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Orpheus TTS Client")
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--prompt', default='Hello! This is a test of the Orpheus text to speech system. It sounds pretty natural, right?', help='Text to convert to speech')
    parser.add_argument('--voice', default='tara', help='Voice name')
    parser.add_argument('--output', default='output.wav', help='Output file path')
    parser.add_argument('--mode', choices=['stream', 'single'], default='stream', help='Streaming or single request')
    parser.add_argument('--measure-ttfb', action='store_true', help='Measure and display TTFB')
    
    args = parser.parse_args()
    
    client = OrpheusTTSClient(args.url)
    
    if args.mode == 'stream':
        client.stream_tts(args.prompt, args.voice, args.output, args.measure_ttfb)
    else:
        client.generate_tts(args.prompt, args.voice, args.output)


if __name__ == '__main__':
    main()
