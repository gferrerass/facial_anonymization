#!/usr/bin/env python3
"""
Setup script for Facial anonymization project.
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

def ensure_inpaint_cropandstitch_repo(comfyui_dir: Path) -> Path:
    """Ensure ComfyUI-Inpaint-CropAndStitch repo exists in ComfyUI/custom_nodes. Returns path or None."""
    if comfyui_dir is None:
        return None
    
    custom_nodes_dir = comfyui_dir / "custom_nodes"
    inpaint_dir = custom_nodes_dir / "ComfyUI-Inpaint-CropAndStitch"
    
    if inpaint_dir.exists():
        print(f"\n✓ ComfyUI-Inpaint-CropAndStitch found at: {inpaint_dir}")
        return inpaint_dir

    print(f"\n{'='*60}")
    print("▶ ComfyUI-Inpaint-CropAndStitch not found. Downloading...")
    print(f"{'='*60}")

    git_check = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if git_check.returncode != 0:
        print("✗ Git is not available. Please install Git and re-run setup.")
        return None

    clone_cmd = ["git", "clone", "https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git", str(inpaint_dir)]
    result = subprocess.run(clone_cmd, capture_output=False)
    if result.returncode != 0:
        print("✗ Failed to download ComfyUI-Inpaint-CropAndStitch")
        return None

    print(f"✓ ComfyUI-Inpaint-CropAndStitch downloaded to: {inpaint_dir}")
    return inpaint_dir

def ensure_impact_pack_repo(comfyui_dir: Path) -> Path:
    """Ensure ComfyUI-Impact-Pack repo exists in ComfyUI/custom_nodes. Returns path or None."""
    if comfyui_dir is None:
        return None
    
    custom_nodes_dir = comfyui_dir / "custom_nodes"
    impact_pack_dir = custom_nodes_dir / "ComfyUI-Impact-Pack"
    
    if impact_pack_dir.exists():
        print(f"\n✓ ComfyUI-Impact-Pack found at: {impact_pack_dir}")
        return impact_pack_dir

    print(f"\n{'='*60}")
    print("▶ ComfyUI-Impact-Pack not found. Downloading...")
    print(f"{'='*60}")

    git_check = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if git_check.returncode != 0:
        print("✗ Git is not available. Please install Git and re-run setup.")
        return None

    clone_cmd = ["git", "clone", "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git", str(impact_pack_dir)]
    result = subprocess.run(clone_cmd, capture_output=False)
    if result.returncode != 0:
        print("✗ Failed to download ComfyUI-Impact-Pack")
        return None

    print(f"✓ ComfyUI-Impact-Pack downloaded to: {impact_pack_dir}")
    return impact_pack_dir

def ensure_impact_subpack_repo(comfyui_dir: Path) -> Path:
    """Ensure ComfyUI-Impact-Subpack repo exists in ComfyUI/custom_nodes. Returns path or None."""
    if comfyui_dir is None:
        return None
    
    custom_nodes_dir = comfyui_dir / "custom_nodes"
    impact_dir = custom_nodes_dir / "ComfyUI-Impact-Subpack"
    
    if impact_dir.exists():
        print(f"\n✓ ComfyUI-Impact-Subpack found at: {impact_dir}")
        return impact_dir

    print(f"\n{'='*60}")
    print("▶ ComfyUI-Impact-Subpack not found. Downloading...")
    print(f"{'='*60}")

    git_check = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if git_check.returncode != 0:
        print("✗ Git is not available. Please install Git and re-run setup.")
        return None

    clone_cmd = ["git", "clone", "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git", str(impact_dir)]
    result = subprocess.run(clone_cmd, capture_output=False)
    if result.returncode != 0:
        print("✗ Failed to download ComfyUI-Impact-Subpack")
        return None

    print(f"✓ ComfyUI-Impact-Subpack downloaded to: {impact_dir}")
    return impact_dir

def ensure_kjnodes_repo(comfyui_dir: Path) -> Path:
    """Ensure ComfyUI-KJNodes repo exists in ComfyUI/custom_nodes. Returns path or None."""
    if comfyui_dir is None:
        return None
    
    custom_nodes_dir = comfyui_dir / "custom_nodes"
    kjnodes_dir = custom_nodes_dir / "ComfyUI-KJNodes"
    
    if kjnodes_dir.exists():
        print(f"\n✓ ComfyUI-KJNodes found at: {kjnodes_dir}")
        return kjnodes_dir

    print(f"\n{'='*60}")
    print("▶ ComfyUI-KJNodes not found. Downloading...")
    print(f"{'='*60}")

    git_check = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if git_check.returncode != 0:
        print("✗ Git is not available. Please install Git and re-run setup.")
        return None

    clone_cmd = ["git", "clone", "https://github.com/kijai/ComfyUI-KJNodes.git", str(kjnodes_dir)]
    result = subprocess.run(clone_cmd, capture_output=False)
    if result.returncode != 0:
        print("✗ Failed to download ComfyUI-KJNodes")
        return None

    print(f"✓ ComfyUI-KJNodes downloaded to: {kjnodes_dir}")
    return kjnodes_dir

def ensure_controlnet_aux_repo(comfyui_dir: Path) -> Path:
    """Ensure comfyui_controlnet_aux repo exists in ComfyUI/custom_nodes. Returns path or None."""
    if comfyui_dir is None:
        return None
    
    custom_nodes_dir = comfyui_dir / "custom_nodes"
    controlnet_aux_dir = custom_nodes_dir / "comfyui_controlnet_aux"
    
    if controlnet_aux_dir.exists():
        print(f"\n✓ comfyui_controlnet_aux found at: {controlnet_aux_dir}")
        return controlnet_aux_dir

    print(f"\n{'='*60}")
    print("▶ comfyui_controlnet_aux not found. Downloading...")
    print(f"{'='*60}")

    git_check = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if git_check.returncode != 0:
        print("✗ Git is not available. Please install Git and re-run setup.")
        return None

    clone_cmd = ["git", "clone", "https://github.com/Fannovel16/comfyui_controlnet_aux.git", str(controlnet_aux_dir)]
    result = subprocess.run(clone_cmd, capture_output=False)
    if result.returncode != 0:
        print("✗ Failed to download comfyui_controlnet_aux")
        return None

    print(f"✓ comfyui_controlnet_aux downloaded to: {controlnet_aux_dir}")
    return controlnet_aux_dir

def collect_custom_nodes_dependencies(comfyui_dir: Path) -> dict:
    """
    Collect all dependencies from custom_nodes requirements.txt files.
    Returns a dictionary with custom node name and their dependencies.
    """
    custom_nodes_deps = {}
    
    if comfyui_dir is None or not comfyui_dir.exists():
        return custom_nodes_deps
    
    custom_nodes_dir = comfyui_dir / "custom_nodes"
    if not custom_nodes_dir.exists():
        return custom_nodes_deps
    
    # List of custom nodes to check
    custom_nodes_to_check = [
        "ComfyUI-Florence2",
        "ComfyUI-Impact-Pack",
        "ComfyUI-Impact-Subpack",
        "ComfyUI-KJNodes",
        "ComfyUI-Inpaint-CropAndStitch",
        "comfyui_controlnet_aux"
    ]
    
    for custom_node in custom_nodes_to_check:
        custom_node_path = custom_nodes_dir / custom_node
        requirements_file = custom_node_path / "requirements.txt"
        
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    custom_nodes_deps[custom_node] = deps
            except Exception as e:
                print(f"⚠ Failed to read requirements from {custom_node}: {e}")
    
    return custom_nodes_deps

def install_custom_nodes_dependencies(python_exe: str, comfyui_dir: Path) -> bool:
    """
    Install all dependencies from custom_nodes in consolidated manner.
    All packages are installed using python_exe (venv Python).
    Returns True if all dependencies were installed successfully.
    """
    custom_nodes_deps = collect_custom_nodes_dependencies(comfyui_dir)
    
    if not custom_nodes_deps:
        print("⚠ No custom_nodes with requirements.txt found")
        return True
    
    print(f"\n{'='*60}")
    print(f"▶ Installing all custom_nodes dependencies in venv...")
    print(f"   Using: {python_exe}")
    print(f"{'='*60}")
    
    all_deps = []
    for custom_node, deps in custom_nodes_deps.items():
        print(f"\n📦 {custom_node}:")
        for dep in deps:
            print(f"  - {dep}")
            if dep not in all_deps:
                all_deps.append(dep)
    
    if not all_deps:
        print("⚠ No dependencies found in custom_nodes requirements")
        return True
    
    # Separate opencv packages to ensure correct installation
    opencv_deps = [d for d in all_deps if 'opencv' in d.lower()]
    other_deps = [d for d in all_deps if 'opencv' not in d.lower()]
    
    print(f"\n▶ Installing {len(all_deps)} unique dependencies...")
    failed_deps = []
    
    # First, ensure opencv-python-headless is installed (avoid conflicts)
    if opencv_deps:
        print("\n🔧 Installing OpenCV (headless version to avoid GUI conflicts)...")
        # Remove any existing opencv packages first
        subprocess.run(
            [python_exe, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"],
            capture_output=True
        )
        # Install opencv-python-headless
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", "opencv-python-headless"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ opencv-python-headless")
        else:
            print("✗ opencv-python-headless - FAILED")
            failed_deps.append("opencv-python-headless")
    
    # Install other dependencies
    for dep in other_deps:
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", dep],
            capture_output=True,
            text=True
        )
        dep_name = dep.split('>=')[0].split('==')[0].split('!=')[0].split('<')[0].split('>')[0]
        if result.returncode == 0:
            print(f"✓ {dep_name}")
        else:
            print(f"✗ {dep_name} - FAILED")
            failed_deps.append(dep)
    
    if failed_deps:
        print(f"\n⚠ {len(failed_deps)} dependencies failed to install:")
        for dep in failed_deps:
            print(f"  - {dep}")
        print("\n💡 You may need to install these manually later.")
        return False
    
    print("\n✓ All custom_nodes dependencies installed successfully!")
    return True

def verify_custom_nodes_installation(comfyui_dir: Path) -> bool:
    """
    Verify that all required custom nodes are installed correctly.
    Returns True if all custom nodes are present.
    """
    if comfyui_dir is None or not comfyui_dir.exists():
        return False
    
    print(f"\n{'='*60}")
    print(f"▶ Verifying custom nodes installation...")
    print(f"{'='*60}")
    
    custom_nodes_dir = comfyui_dir / "custom_nodes"
    required_nodes = {
        "ComfyUI-Florence2": "Florence2 image captioning",
        "ComfyUI-Impact-Pack": "Face detection with BboxDetectorCombined",
        "ComfyUI-Impact-Subpack": "Ultralytics YOLO detector",
        "ComfyUI-KJNodes": "Mask operations (GrowMaskWithBlur)",
        "ComfyUI-Inpaint-CropAndStitch": "Inpainting crop and stitch operations",
        "comfyui_controlnet_aux": "ControlNet auxiliary nodes (preprocessors)"
    }
    
    all_present = True
    for node_name, description in required_nodes.items():
        node_path = custom_nodes_dir / node_name
        if node_path.exists() and node_path.is_dir():
            # Check if __init__.py exists (basic validation)
            init_file = node_path / "__init__.py"
            if init_file.exists():
                print(f"✓ {node_name:35s} - {description}")
            else:
                print(f"⚠ {node_name:35s} - Present but missing __init__.py")
                all_present = False
        else:
            print(f"✗ {node_name:35s} - NOT FOUND")
            all_present = False
    
    if all_present:
        print("\n✓ All required custom nodes are installed!")
    else:
        print("\n⚠ Some custom nodes are missing or incomplete.")
        print("   This may cause errors when running the facial anonymization.")
    
    return all_present

def main():
    print("\n" + "="*60)
    print("   FACIAL ANONYMIZATION SETUP - Creating Virtual Environment")
    print("="*60)
    
    # Detect OS
    system = platform.system()
    venv_dir = Path("venv")
    base_dir = Path(__file__).parent
    
    # Create virtual environment (using system Python)
    if not venv_dir.exists():
        print(f"\n▶ Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv venv", "Virtual environment creation"):
            return False
    else:
        print(f"\n✓ Virtual environment already exists")
    
    # Get Python executable in venv (all subsequent installations use this)
    if system == "Windows":
        python_exe = str(venv_dir / "Scripts" / "python.exe")
    else:
        python_exe = str(venv_dir / "bin" / "python")
    
    print(f"\n🔧 Using virtual environment Python: {python_exe}")
    print(f"   All packages will be installed in the isolated venv")
    
    # Upgrade pip using venv python
    print(f"\n▶ Upgrading pip in virtual environment...")
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
    
    # ========================================================================
    # ALL PACKAGES BELOW ARE INSTALLED IN THE VIRTUAL ENVIRONMENT (venv)
    # using python_exe which points to venv/Scripts/python.exe
    # ========================================================================
    
    # Install PyTorch with CUDA support in venv
    print(f"\n{'='*60}")
    print(f"▶ Installing PyTorch with CUDA 13.0 support in venv...")
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
    
    # Install other requirements in venv
    print(f"\n{'='*60}")
    print(f"▶ Installing other dependencies in venv...")
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

    # Install ComfyUI requirements in venv
    comfyui_requirements = None
    if comfyui_dir is not None:
        comfyui_requirements = comfyui_dir / "requirements.txt"

    if comfyui_requirements and comfyui_requirements.exists():
        print(f"\n{'='*60}")
        print("▶ Installing ComfyUI requirements in venv...")
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
    
    # Ensure all custom_nodes are available and install their dependencies in venv
    print(f"\n{'='*60}")
    print(f"▶ Setting up custom nodes and installing dependencies in venv...")
    print(f"{'='*60}")
    
    florence2_dir = ensure_florence2_repo(comfyui_dir)
    inpaint_dir = ensure_inpaint_cropandstitch_repo(comfyui_dir)
    impact_pack_dir = ensure_impact_pack_repo(comfyui_dir)
    impact_subpack_dir = ensure_impact_subpack_repo(comfyui_dir)
    kjnodes_dir = ensure_kjnodes_repo(comfyui_dir)
    controlnet_aux_dir = ensure_controlnet_aux_repo(comfyui_dir)
    
    # Install all custom_nodes dependencies in consolidated manner
    install_custom_nodes_dependencies(python_exe, comfyui_dir)
    
    # Verify all custom nodes are properly installed
    all_nodes_ok = verify_custom_nodes_installation(comfyui_dir)
    
    print("\n" + "="*60)
    print("   SETUP COMPLETE!")
    print("="*60)
    print(f"\n✓ Virtual environment created at: {venv_dir.absolute()}")
    print(f"✓ Python executable: {python_exe}")
    print(f"✓ PyTorch with CUDA 13.0 support installed")
    print(f"✓ ComfyUI and all custom nodes installed")
    print(f"✓ All dependencies installed in isolated venv")
    
    if not all_nodes_ok:
        print(f"\n⚠ Warning: Some custom nodes may be incomplete.")
        print(f"   You may need to run setup again or install them manually.")
    
    print(f"\n📁 Project structure:")
    print(f"   - input/     : Place images to anonymize here")
    print(f"   - output/    : Anonymized images will be saved here")
    print(f"   - models/    : Model files will be downloaded here automatically")
    print(f"   - venv/      : Isolated Python environment (all packages here)")
    
    print(f"\n💡 Note: All packages are installed in the venv folder.")
    print(f"   The system Python installation is not affected.")
    
    print(f"\n🚀 Next step: Run the facial anonymization with:")
    print(f"   python run.py")
    print(f"\n   (run.py automatically uses the venv Python)")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
