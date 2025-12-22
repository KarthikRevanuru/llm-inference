# Orpheus-3B TTS Deployment

High-performance streaming Text-to-Speech API using [Orpheus-3B](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) with optimized GPU utilization and minimal Time-to-First-Byte (TTFB).

## Features

- ⚡ **Streaming Audio**: Server-Sent Events (SSE) for minimal TTFB (~200-300ms)
- 🚀 **Concurrent Requests**: vLLM continuous batching handles multiple requests efficiently
- 🎯 **Full GPU Utilization**: Optimized batching and memory management
- 📊 **Prometheus Metrics**: Monitor TTFB, throughput, GPU usage, and more
- 🎭 **8 Voices**: Natural-sounding English voices with emotion tags
- 🐳 **Docker Ready**: Easy deployment with GPU passthrough

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- NVIDIA drivers and CUDA toolkit installed

### Local Installation

1. **Clone and install dependencies**:
```bash
git clone <your-repo-url>
cd exo-juno
pip install -r requirements.txt
```

2. **Configure settings** (optional):
```bash
cp config.ini.example config.ini
# Edit config.ini to customize settings
```

3. **Start the server**:
```bash
python server.py
```

The server will start on `http://localhost:8000` and automatically download the model on first run.

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## API Usage

### Streaming Endpoint (Recommended)

Stream audio chunks as they're generated for minimal latency:

**Python**:
```python
import requests
import base64
import wave

url = "http://localhost:8000/v1/tts/stream"
payload = {
    "prompt": "Hello! This is a test of streaming TTS.",
    "voice": "tara"
}

with wave.open("output.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    
    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line.startswith(b'data:'):
                data = line.decode('utf-8').split(':', 1)[1].strip()
                # Decode base64 audio chunk
                audio_bytes = base64.b64decode(data)
                wf.writeframes(audio_bytes)
```

**cURL**:
```bash
curl -X POST http://localhost:8000/v1/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world!", "voice": "tara"}'
```

### Non-Streaming Endpoint

Generate complete audio file:

**Python**:
```python
import requests

url = "http://localhost:8000/v1/tts/generate"
payload = {
    "prompt": "Hello! This is a complete audio file.",
    "voice": "tara"
}

response = requests.post(url, json=payload)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

**cURL**:
```bash
curl -X POST http://localhost:8000/v1/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world!", "voice": "tara"}' \
  --output output.wav
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text to convert to speech (max 1000 chars) |
| `voice` | string | `"tara"` | Voice name: tara, leah, jess, leo, dan, mia, zac, zoe |
| `temperature` | float | `1.0` | Sampling temperature (0.1-2.0) |
| `top_p` | float | `0.95` | Top-p sampling (0.0-1.0) |
| `repetition_penalty` | float | `1.1` | Repetition penalty (1.0-2.0, must be ≥1.1) |

### Available Voices

- **tara** - Natural, conversational (recommended)
- **leah** - Warm, friendly
- **jess** - Clear, professional
- **leo** - Deep, authoritative
- **dan** - Casual, relaxed
- **mia** - Energetic, upbeat
- **zac** - Smooth, calm
- **zoe** - Bright, expressive

### Emotion Tags

Add emotion to speech with tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`

```python
payload = {
    "prompt": "That's hilarious! <laugh> I can't believe it!",
    "voice": "tara"
}
```

## Example Clients

### Basic Streaming Client

```bash
python examples/client.py \
  --prompt "Your text here" \
  --voice tara \
  --mode stream \
  --measure-ttfb
```

### Concurrent Load Testing

Test concurrent request handling:

```bash
python examples/test_concurrent.py \
  --num-requests 10 \
  --concurrent 5
```

This will:
- Send 10 requests with 5 concurrent workers
- Measure TTFB, generation time, and realtime factor
- Report statistics and throughput

## Performance Tuning

### GPU Memory Utilization

Adjust in `config.ini` or `config.py`:
```python
GPU_MEMORY_UTILIZATION=0.90  # Use 90% of GPU memory
```

### Concurrent Request Handling

Control batch size and concurrent sequences:
```python
MAX_NUM_SEQS=8        # Max concurrent requests
MAX_BATCH_SIZE=8      # Batch size for continuous batching
```

**Guidelines**:
- Higher values = more concurrent requests but higher memory usage
- Start with 8 and adjust based on GPU memory
- Monitor GPU usage with `nvidia-smi`

### Multi-GPU Support

For multiple GPUs, set tensor parallelism:
```python
TENSOR_PARALLEL_SIZE=2  # Number of GPUs
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

**Key Metrics**:
- `tts_ttfb_seconds` - Time to first byte histogram
- `tts_request_duration_seconds` - Total request duration
- `tts_generation_speed_ratio` - Realtime factor (audio_duration / generation_time)
- `tts_active_requests` - Current active requests
- `tts_gpu_memory_bytes` - GPU memory usage
- `tts_requests_total` - Total requests by endpoint and status

## Troubleshooting

### Model Download Issues

The model (~6GB) downloads automatically on first run. If download fails:

```bash
# Pre-download the model
python -c "from huggingface_hub import snapshot_download; snapshot_download('canopylabs/orpheus-3b-0.1-ft')"
```

### GPU Out of Memory

Reduce concurrent requests or GPU memory utilization:
```python
MAX_NUM_SEQS=4
GPU_MEMORY_UTILIZATION=0.80
```

### Slow Generation

Check GPU utilization:
```bash
watch -n 1 nvidia-smi
```

If GPU utilization is low:
- Increase `MAX_NUM_SEQS` for more batching
- Check if CPU is bottleneck
- Ensure CUDA is properly installed

### vLLM Version Issues

The deployment uses vLLM 0.7.3 (stable version). If you encounter issues:
```bash
pip install vllm==0.7.3 --force-reinstall
```

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP/SSE
       ▼
┌─────────────────────────────┐
│      FastAPI Server         │
│  - Request validation       │
│  - Async request handling   │
│  - SSE streaming            │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│     Model Manager           │
│  - Singleton pattern        │
│  - Thread-safe access       │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│    vLLM Engine              │
│  - Continuous batching      │
│  - GPU memory management    │
│  - Streaming generation     │
└─────────────────────────────┘
```

## Configuration Reference

See [config.py](config.py) for all available settings.

**Key Settings**:
- `MODEL_NAME` - HuggingFace model identifier
- `MAX_MODEL_LEN` - Maximum sequence length (default: 2048)
- `MAX_NUM_SEQS` - Maximum concurrent sequences (default: 8)
- `GPU_MEMORY_UTILIZATION` - GPU memory fraction (default: 0.90)
- `HOST` / `PORT` - Server binding (default: 0.0.0.0:8000)

## License

This deployment code is provided as-is. The Orpheus-3B model has its own license - see [HuggingFace model card](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft).

## Credits

- **Model**: [Orpheus-3B](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) by Canopy Labs
- **Inference Engine**: [vLLM](https://github.com/vllm-project/vllm)
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
