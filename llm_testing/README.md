# Small LLM Testing (vLLM Backend)

A framework for testing small language models using **vLLM** with native **continuous batching**, **async streaming**, and **high concurrency**.

## 🚀 Quick Start

```bash
# Install dependencies (requires NVIDIA GPU)
pip install -r requirements.txt

# Start the server
python llm_server.py --model facebook/opt-350m

# In another terminal, run tests
python test_server.py
```

## ⚡ vLLM Advantages

| Feature | How vLLM Implements It |
|---------|------------------------|
| **Continuous Batching** | Dynamic batching during generation, not just at start |
| **PagedAttention** | Efficient memory management for high concurrency |
| **Async Streaming** | Native async token streaming |
| **High Throughput** | 10-100x faster than naive HuggingFace |

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Single prompt (supports `stream: true`) |
| `/batch` | POST | Multiple prompts with continuous batching |
| `/stats` | GET | Server statistics |
| `/models` | GET | List available models |

## 📦 Files

| File | Description |
|------|-------------|
| `llm_server.py` | FastAPI + vLLM AsyncLLMEngine server |
| `llm_client.py` | Python client (sync + async) |
| `llm_runner.py` | Standalone vLLM runner (no server) |
| `test_llm.py` | Interactive CLI |
| `test_server.py` | Test suite for batch/stream/concurrent |
| `config.py` | Model catalog |

## 🖥️ Server Options

```bash
python llm_server.py \
    --model facebook/opt-350m \    # Model name
    --tensor-parallel 1 \          # GPUs for tensor parallelism
    --gpu-memory 0.85 \            # GPU memory utilization
    --max-model-len 2048 \         # Max context length
    --port 8000
```

## 📊 Available Models

```bash
python test_llm.py --list
```

| Model | Size | Notes |
|-------|------|-------|
| `facebook/opt-350m` | 350M | Default, good compatibility |
| `EleutherAI/pythia-410m` | 410M | Excellent vLLM support |
| `microsoft/phi-2` | 2.7B | High quality |
| `Qwen/Qwen2-0.5B-Instruct` | 0.5B | Fast |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Good balance |

## 🧪 Testing

```bash
# Run all tests
python test_server.py

# Run specific test
python test_server.py --test batch
python test_server.py --test stream
python test_server.py --test throughput
python test_server.py --test concurrent-stream
```

## 🔗 cURL Examples

```bash
# Single request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 50}'

# Batch
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["What is AI?", "What is ML?"], "max_tokens": 50}'

# Streaming
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story", "stream": true}'
```

## ⚠️ Requirements

- **NVIDIA GPU** with CUDA support
- Python 3.8+
- vLLM 0.6.0+
