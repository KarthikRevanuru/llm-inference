#!/usr/bin/env python3
"""
Interactive vLLM Testing Script

Usage:
    python test_llm.py                    # Uses default model (opt-350m)
    python test_llm.py --model phi-2      # Specific model
    python test_llm.py --list             # List available models
"""

import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from llm_runner import VLLMRunner, list_available_models
from config import SMALL_LLMS, DEFAULT_MODEL


console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Test small language models with vLLM")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to test (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Single prompt to run (non-interactive mode)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.85,
        help="GPU memory utilization (default: 0.85)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode"
    )
    
    return parser.parse_args()


def run_benchmark(runner: VLLMRunner, args):
    """Run benchmark tests."""
    console.print("\n[bold cyan]🔬 Running Benchmark (vLLM)...[/bold cyan]\n")
    
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What are the benefits of exercise?",
        "Describe the process of photosynthesis.",
    ]
    
    results = runner.benchmark(test_prompts, num_runs=2, max_tokens=args.max_tokens)
    
    console.print(Panel.fit(
        f"[bold]Model:[/bold] {results['model']}\n"
        f"[bold]Backend:[/bold] {results['backend']}\n"
        f"[bold]Load Time:[/bold] {results['load_time']:.2f}s\n"
        f"[bold]Total Prompts:[/bold] {results['total_prompts']}\n"
        f"[bold]Total Tokens:[/bold] {results['total_tokens']}\n"
        f"[bold]Avg Tokens/sec:[/bold] {results['avg_tokens_per_second']:.1f}",
        title="📊 Benchmark Results",
        border_style="green",
    ))


def run_interactive(runner: VLLMRunner, args):
    """Run interactive mode."""
    console.print(Panel.fit(
        "[bold green]Interactive vLLM Testing[/bold green]\n\n"
        "Commands:\n"
        "  • Type your prompt and press Enter\n"
        "  • Type 'exit' or 'quit' to stop\n"
        "  • Type 'info' to show model info",
        border_style="blue",
    ))
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            
            if user_input.lower() == "info":
                runner._print_model_info()
                continue
            
            if not user_input.strip():
                continue
            
            # Generate response
            result = runner.generate(
                user_input,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            
            console.print(f"\n[bold magenta]Assistant:[/bold magenta] {result['generated_text']}")
            console.print(
                f"\n[dim]({result['output_tokens']} tokens, "
                f"{result['generation_time']:.2f}s, "
                f"{result['tokens_per_second']:.1f} tok/s)[/dim]"
            )
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def run_single_prompt(runner: VLLMRunner, prompt: str, args):
    """Run a single prompt."""
    console.print(f"\n[bold cyan]Prompt:[/bold cyan] {prompt}\n")
    
    result = runner.generate(
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    console.print(Panel(
        result["generated_text"],
        title="Response",
        border_style="green",
    ))
    console.print(
        f"[dim]Metrics: {result['output_tokens']} tokens, "
        f"{result['generation_time']:.2f}s, "
        f"{result['tokens_per_second']:.1f} tok/s[/dim]"
    )


def main():
    args = parse_args()
    
    # List models and exit
    if args.list:
        list_available_models()
        sys.exit(0)
    
    # Validate model
    if args.model not in SMALL_LLMS and not args.model.startswith(("huggingface/", "/")):
        console.print(f"[yellow]Warning: '{args.model}' not in predefined list. Using as HuggingFace model ID.[/yellow]")
    
    # Initialize and load model
    console.print(Panel.fit(
        f"[bold]Model:[/bold] {args.model}\n"
        f"[bold]Backend:[/bold] vLLM\n"
        f"[bold]GPU Memory:[/bold] {args.gpu_memory}",
        title="🚀 Loading Model",
        border_style="blue",
    ))
    
    runner = VLLMRunner(
        model_name=args.model,
        gpu_memory_utilization=args.gpu_memory,
    )
    
    try:
        runner.load()
        
        if args.benchmark:
            run_benchmark(runner, args)
        elif args.prompt:
            run_single_prompt(runner, args.prompt, args)
        else:
            run_interactive(runner, args)
            
    finally:
        runner.unload()


if __name__ == "__main__":
    main()
