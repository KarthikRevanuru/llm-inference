"""
Configuration for Small LLM Testing (vLLM backend)
"""

# Popular small LLMs to test (< 4B parameters)
# These are all compatible with vLLM
SMALL_LLMS = {
    # Phi family (Microsoft)
    "phi-2": "microsoft/phi-2",  # 2.7B
    
    # TinyLlama
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B
    
    # Qwen family
    "qwen2-0.5b": "Qwen/Qwen2-0.5B-Instruct",  # 0.5B
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",  # 1.5B
    
    # Gemma family (Google)
    "gemma-2b": "google/gemma-2b-it",  # 2B
    
    # StableLM
    "stablelm-zephyr-3b": "stabilityai/stablelm-zephyr-3b",  # 3B
    
    # SmolLM (Hugging Face) - may need --trust-remote-code
    "smollm-135m": "HuggingFaceTB/SmolLM-135M-Instruct",  # 135M
    "smollm-360m": "HuggingFaceTB/SmolLM-360M-Instruct",  # 360M
    "smollm-1.7b": "HuggingFaceTB/SmolLM-1.7B-Instruct",  # 1.7B
    
    # OPT (Meta) - good for testing, very compatible
    "opt-125m": "facebook/opt-125m",  # 125M
    "opt-350m": "facebook/opt-350m",  # 350M
    "opt-1.3b": "facebook/opt-1.3b",  # 1.3B
    
    # Pythia (EleutherAI) - excellent vLLM compatibility
    "pythia-160m": "EleutherAI/pythia-160m",  # 160M
    "pythia-410m": "EleutherAI/pythia-410m",  # 410M
    "pythia-1b": "EleutherAI/pythia-1b",  # 1B
    "pythia-1.4b": "EleutherAI/pythia-1.4b",  # 1.4B
}

# Default model for quick testing (good vLLM compatibility)
DEFAULT_MODEL = "facebook/opt-350m"

# vLLM Sampling parameters
DEFAULT_SAMPLING_PARAMS = {
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
}

# vLLM Engine settings
DEFAULT_ENGINE_ARGS = {
    "trust_remote_code": True,
    "dtype": "auto",  # Will use float16 on GPU
    "gpu_memory_utilization": 0.85,
    "max_model_len": 2048,
}
