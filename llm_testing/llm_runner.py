"""
vLLM-based LLM Runner

A simple interface for loading and testing small language models using vLLM.
Supports offline inference and async generation.
"""

import time
from typing import Optional, Dict, Any, List, AsyncGenerator
import asyncio

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from config import (
    SMALL_LLMS,
    DEFAULT_MODEL,
    DEFAULT_SAMPLING_PARAMS,
    DEFAULT_ENGINE_ARGS,
)


class VLLMRunner:
    """
    A class to load and run language models using vLLM for testing.
    Supports both sync (offline) and async modes.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        trust_remote_code: bool = True,
    ):
        """
        Initialize the vLLM runner.
        
        Args:
            model_name: Key from SMALL_LLMS or a HuggingFace model ID
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            trust_remote_code: Whether to trust remote code
        """
        self.model_name = model_name
        self.model_id = SMALL_LLMS.get(model_name, model_name)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        
        self.llm: Optional[LLM] = None
        self.load_time = 0
        
    def load(self) -> None:
        """Load the model using vLLM."""
        print(f"🚀 Loading model with vLLM: {self.model_id}")
        print(f"   Tensor Parallel Size: {self.tensor_parallel_size}")
        print(f"   GPU Memory Utilization: {self.gpu_memory_utilization}")
        
        start_time = time.time()
        
        self.llm = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            dtype="auto",
        )
        
        self.load_time = time.time() - start_time
        print(f"✅ Model loaded in {self.load_time:.2f}s")
        self._print_model_info()
    
    def _print_model_info(self) -> None:
        """Print model information."""
        if self.llm is None:
            return
        
        print(f"\n{'='*50}")
        print(f"Model: {self.model_id}")
        print(f"Backend: vLLM")
        print(f"Max Model Length: {self.max_model_len}")
        print(f"GPU Memory Utilization: {self.gpu_memory_utilization}")
        print(f"{'='*50}\n")
    
    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text from a single prompt.
        
        Args:
            prompt: Input text prompt
            **kwargs: Override sampling parameters
            
        Returns:
            Dict with generated text and metrics
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Merge sampling params
        params = {**DEFAULT_SAMPLING_PARAMS, **kwargs}
        sampling_params = SamplingParams(
            max_tokens=params.get("max_tokens", 256),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 50),
            repetition_penalty=params.get("repetition_penalty", 1.1),
        )
        
        # Generate
        start_time = time.time()
        outputs = self.llm.generate([prompt], sampling_params)
        generation_time = time.time() - start_time
        
        output = outputs[0]
        generated_text = output.outputs[0].text
        output_tokens = len(output.outputs[0].token_ids)
        input_tokens = len(output.prompt_token_ids)
        
        return {
            "generated_text": generated_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_time": generation_time,
            "tokens_per_second": output_tokens / generation_time if generation_time > 0 else 0,
        }
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts in a single batch.
        vLLM handles batching efficiently with continuous batching.
        
        Args:
            prompts: List of input prompts
            **kwargs: Override sampling parameters
            
        Returns:
            List of result dicts
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Merge sampling params
        params = {**DEFAULT_SAMPLING_PARAMS, **kwargs}
        sampling_params = SamplingParams(
            max_tokens=params.get("max_tokens", 256),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 50),
            repetition_penalty=params.get("repetition_penalty", 1.1),
        )
        
        # Generate batch
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        total_time = time.time() - start_time
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            output_tokens = len(output.outputs[0].token_ids)
            input_tokens = len(output.prompt_token_ids)
            
            results.append({
                "generated_text": generated_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "generation_time": total_time / len(prompts),
                "tokens_per_second": output_tokens / (total_time / len(prompts)) if total_time > 0 else 0,
            })
        
        return results
    
    def benchmark(
        self,
        prompts: List[str],
        num_runs: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Benchmark the model with multiple prompts.
        
        Args:
            prompts: List of prompts to test
            num_runs: Number of runs per prompt
            **kwargs: Override sampling parameters
            
        Returns:
            Benchmark results
        """
        results = []
        
        for _ in range(num_runs):
            batch_results = self.generate_batch(prompts, **kwargs)
            results.extend(batch_results)
        
        # Calculate averages
        total_tokens = sum(r["output_tokens"] for r in results)
        total_time = sum(r["generation_time"] for r in results)
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        
        return {
            "model": self.model_id,
            "backend": "vLLM",
            "load_time": self.load_time,
            "total_prompts": len(prompts) * num_runs,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "avg_tokens_per_second": avg_tps,
        }
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        
        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        print("Model unloaded.")


def list_available_models() -> Dict[str, str]:
    """List all available small LLMs."""
    print("\n📚 Available Small LLMs (vLLM compatible):\n")
    for key, model_id in SMALL_LLMS.items():
        print(f"  • {key}: {model_id}")
    print()
    return SMALL_LLMS


if __name__ == "__main__":
    # Quick test
    list_available_models()
    
    print("\n🚀 Quick Test with OPT-350M (vLLM):\n")
    
    runner = VLLMRunner("opt-350m")
    runner.load()
    
    result = runner.generate("Explain what machine learning is in simple terms:")
    
    print(f"\nPrompt: Explain what machine learning is in simple terms:")
    print(f"\nGenerated ({result['output_tokens']} tokens in {result['generation_time']:.2f}s):")
    print(f"{result['generated_text']}")
    print(f"\n⚡ Speed: {result['tokens_per_second']:.1f} tokens/sec")
    
    runner.unload()
