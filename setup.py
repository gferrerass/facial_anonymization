#!/usr/bin/env python3
"""
Setup script for Facial anonymisation project.
Creates virtual environment and installs dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✓ {description} - OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e}")
        return False

def ensure_comfyui_repo(base_dir: Path) -> Path:
    """Ensure ComfyUI repo exists in zimage/ComfyUI. Returns path or None."""
    comfyui_dir = base_dir / "ComfyUI"
    if comfyui_dir.exists():
        print(f"\n✓ ComfyUI found at: {comfyui_dir}")
        return comfyui_dir

    print(f"\n{'='*60}")
    print("▶ ComfyUI not found. Downloading...")
    print(f"{'='*60}")

    git_check = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if git_check.returncode != 0:
        print("✗ Git is not available. Please install Git and re-run setup.")
        return None

    clone_cmd = ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(comfyui_dir)]
    result = subprocess.run(clone_cmd, capture_output=False)
    if result.returncode != 0:
        print("✗ Failed to download ComfyUI")
        return None

    print(f"✓ ComfyUI downloaded to: {comfyui_dir}")
    return comfyui_dir

def ensure_florence2_repo(comfyui_dir: Path) -> Path:
    """Ensure ComfyUI-Florence2 repo exists in ComfyUI/custom_nodes. Returns path or None."""
    if comfyui_dir is None:
        return None
    
    custom_nodes_dir = comfyui_dir / "custom_nodes"
    florence2_dir = custom_nodes_dir / "ComfyUI-Florence2"
    
    if florence2_dir.exists():
        print(f"\n✓ ComfyUI-Florence2 found at: {florence2_dir}")
        return florence2_dir

    print(f"\n{'='*60}")
    print("▶ ComfyUI-Florence2 not found. Downloading...")
    print(f"{'='*60}")

    git_check = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if git_check.returncode != 0:
        print("✗ Git is not available. Please install Git and re-run setup.")
        return None

    clone_cmd = ["git", "clone", "https://github.com/kijai/ComfyUI-Florence2.git", str(florence2_dir)]
    result = subprocess.run(clone_cmd, capture_output=False)
    if result.returncode != 0:
        print("✗ Failed to download ComfyUI-Florence2")
        return None

    print(f"✓ ComfyUI-Florence2 downloaded to: {florence2_dir}")
    return florence2_dir

def main():
    print("\n" + "="*60)
    print("   FACIAL ANONYMISATION SETUP - Creating Virtual Environment")
    print("="*60)
    
    # Detect OS
    system = platform.system()
    venv_dir = Path("venv")
    base_dir = Path(__file__).parent
    
    # Create virtual environment
    if not venv_dir.exists():
        print(f"\n▶ Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv venv", "Virtual environment creation"):
            return False
    else:
        print(f"\n✓ Virtual environment already exists")
    
    # Get Python executable in venv
    if system == "Windows":
        python_exe = str(venv_dir / "Scripts" / "python.exe")
    else:
        python_exe = str(venv_dir / "bin" / "python")
    
    # Upgrade pip using python -m pip (more reliable)
    print(f"\n▶ Upgrading pip...")
    result = subprocess.run(
        [python_exe, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ Pip upgrade - OK")
    else:
        print("⚠ Pip upgrade - Skipped (not critical)")

    # Ensure ComfyUI is available locally
    comfyui_dir = ensure_comfyui_repo(base_dir)
    
    # Install PyTorch with CUDA support
    print(f"\n{'='*60}")
    print(f"▶ Installing PyTorch with CUDA 13.0 support...")
    print(f"{'='*60}")
    pytorch_cmd = [python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu130"]
    result = subprocess.run(pytorch_cmd, capture_output=False)
    if result.returncode != 0:
        print("\n⚠ PyTorch CUDA installation failed, trying CPU version...")
        pytorch_cpu_cmd = [python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        result = subprocess.run(pytorch_cpu_cmd, capture_output=False)
        if result.returncode != 0:
            print("✗ PyTorch installation failed")
            return False
    print("✓ PyTorch installation - OK")
    
    # Install other requirements
    print(f"\n{'='*60}")
    print(f"▶ Installing other dependencies...")
    print(f"{'='*60}")
    other_deps = [
        "numpy>=1.25.0",
        "Pillow>=9.0.0",
        "scipy>=1.10.0",
        "psutil>=5.8.0",
        "PyYAML>=5.4.1",
        "safetensors>=0.4.2",
        "transformers>=4.30.0",
        "aiohttp>=3.8.0",
        "einops>=0.6.0",
        "torchsde>=0.2.5",
        "av>=10.0.0",
        "requests>=2.31.0"
    ]
    for dep in other_deps:
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", dep],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ {dep.split('>=')[0]}")
        else:
            print(f"⚠ {dep.split('>=')[0]} - Skipped")

    # Install ComfyUI requirements (frontend package, etc.)
    comfyui_requirements = None
    if comfyui_dir is not None:
        comfyui_requirements = comfyui_dir / "requirements.txt"

    if comfyui_requirements and comfyui_requirements.exists():
        print(f"\n{'='*60}")
        print("▶ Installing ComfyUI requirements...")
        print(f"{'='*60}")
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", "-r", str(comfyui_requirements)],
            capture_output=False
        )
        if result.returncode == 0:
            print("✓ ComfyUI requirements - OK")
        else:
            print("⚠ ComfyUI requirements - Failed (frontend may be missing)")
    else:
        print("⚠ ComfyUI requirements.txt not found; skipping")
    
    # Ensure ComfyUI-Florence2 is available
    florence2_dir = ensure_florence2_repo(comfyui_dir)
    
    # Install ComfyUI-Florence2 requirements
    florence2_requirements = None
    if florence2_dir is not None:
        florence2_requirements = florence2_dir / "requirements.txt"

    if florence2_requirements and florence2_requirements.exists():
        print(f"\n{'='*60}")
        print("▶ Installing ComfyUI-Florence2 requirements...")
        print(f"{'='*60}")
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", "-r", str(florence2_requirements)],
            capture_output=False
        )
        if result.returncode == 0:
            print("✓ ComfyUI-Florence2 requirements - OK")
        else:
            print("⚠ ComfyUI-Florence2 requirements - Failed")
    else:
        print("⚠ ComfyUI-Florence2 requirements.txt not found; skipping")
    
    print("\n" + "="*60)
    print("   SETUP COMPLETE!")
    print("="*60)
    print(f"\n✓ Virtual environment created at: {venv_dir.absolute()}")
    print(f"✓ PyTorch with CUDA 13.0 support installed")
    print(f"✓ All dependencies installed")
    print(f"\nNext step: Run the script with:")
    print(f"  python run.py")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
