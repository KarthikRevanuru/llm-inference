#!/usr/bin/env python3
"""Setup script for Orpheus TTS - handles GPU vendor selection and dependencies."""
import subprocess
import sys
import os


def get_gpu_choice():
    """Prompt user for GPU vendor selection."""
    print("\n" + "=" * 50)
    print("  Orpheus TTS Setup")
    print("=" * 50)
    print("\nSelect your GPU vendor:\n")
    print("  1. NVIDIA (CUDA) - Recommended, production-ready")
    print("  2. AMD (ROCm)    - Experimental, requires ROCm 6.0+")
    print("  3. CPU only      - No GPU acceleration")
    print()
    
    while True:
        choice = input("Enter choice [1/2/3]: ").strip()
        if choice in ("1", "2", "3"):
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def detect_gpu():
    """Try to auto-detect GPU vendor."""
    # Check for NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return "nvidia"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check for AMD ROCm
    try:
        result = subprocess.run(
            ["rocm-smi"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return "amd"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None


def install_dependencies(gpu_type: str):
    """Install dependencies based on GPU type."""
    print(f"\nInstalling dependencies for {gpu_type.upper()}...")
    
    if gpu_type == "nvidia":
        # Standard CUDA installation
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
    elif gpu_type == "amd":
        # ROCm installation
        print("\n⚠️  AMD ROCm support is experimental!")
        print("    Ensure ROCm 6.0+ is installed on your system.\n")
        
        # Install PyTorch for ROCm first
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "--index-url", "https://download.pytorch.org/whl/rocm6.0"
        ], check=True)
        
        # Install remaining dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-rocm.txt"
        ], check=True)
        
    else:  # cpu
        print("\n⚠️  CPU-only mode: Generation will be very slow!\n")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-cpu.txt"
        ], check=True)
    
    print("\n✅ Dependencies installed successfully!")


def create_config():
    """Create config.ini from example if it doesn't exist."""
    if not os.path.exists("config.ini"):
        if os.path.exists("config.ini.example"):
            import shutil
            shutil.copy("config.ini.example", "config.ini")
            print("✅ Created config.ini from config.ini.example")
        else:
            print("ℹ️  No config.ini.example found, using defaults")


def main():
    """Main setup flow."""
    # Try auto-detection first
    detected = detect_gpu()
    
    if detected:
        print(f"\n🔍 Detected {detected.upper()} GPU")
        use_detected = input(f"Use {detected.upper()}? [Y/n]: ").strip().lower()
        if use_detected in ("", "y", "yes"):
            gpu_type = detected
        else:
            choice = get_gpu_choice()
            gpu_type = {"1": "nvidia", "2": "amd", "3": "cpu"}[choice]
    else:
        print("\n🔍 No GPU detected automatically")
        choice = get_gpu_choice()
        gpu_type = {"1": "nvidia", "2": "amd", "3": "cpu"}[choice]
    
    # Install dependencies
    install_dependencies(gpu_type)
    
    # Create config
    create_config()
    
    print("\n" + "=" * 50)
    print("  Setup Complete!")
    print("=" * 50)
    print("\nTo start the server:")
    print("  python server.py")
    print()


if __name__ == "__main__":
    main()
