#!/usr/bin/env python3
"""
Run script for Facial Anonymization project.
Activates virtual environment and runs main.py with optional parameters.
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Batch facial anonymization with face inpainting using ComfyUI pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py
  python run.py --strength 0.8 --denoise 0.7
  python run.py --input photo.jpg --output results
  python run.py --input custom_input --output custom_output --max-images 5
  python run.py --strength 0.5 --denoise 0.5 --input ./photos --output ./results
        """
    )
    
    parser.add_argument(
        "--strength",
        type=float,
        default=0.7,
        help="ControlNet edge guidance strength (0.0-1.0). Higher = stronger edge guidance. Default: 0.7"
    )
    
    parser.add_argument(
        "--denoise",
        type=float,
        default=0.6,
        help="Denoising/generation strength (0.0-1.0). Higher = more facial changes. Default: 0.6"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="input",
        help="Input directory or image file path to anonymize (absolute or relative). Default: input"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for anonymized images (absolute or relative path). Default: output"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process from directory. Default: all images"
    )
    
    args = parser.parse_args()
    print("\n" + "="*60)
    print("   FACIAL ANONYMIZATION - Starting Workflow")
    print("="*60)
    
    venv_dir = Path("venv")
    system = platform.system()
    
    # Check if virtual environment exists
    if not venv_dir.exists():
        print(f"\n✗ Virtual environment not found!")
        print(f"Please run setup first:")
        print(f"  python setup.py")
        sys.exit(1)
    
    # Get Python executable in venv
    if system == "Windows":
        python_exe = str(venv_dir / "Scripts" / "python.exe")
    else:
        python_exe = str(venv_dir / "bin" / "python")
    
    if not Path(python_exe).exists():
        print(f"\n✗ Python executable not found at: {python_exe}")
        print(f"Virtual environment may be corrupted. Try running setup again:")
        print(f"  python setup.py")
        sys.exit(1)
    
    print(f"\n✓ Virtual environment found")
    print(f"✓ Python: {python_exe}")
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"   ControlNet strength: {args.strength}")
    print(f"   Denoise strength: {args.denoise}")
    print(f"   Input directory: {args.input}")
    print(f"   Output directory: {args.output}")
    if args.max_images:
        print(f"   Max images: {args.max_images}")
    print(f"\nStarting Facial Anonymization workflow...\n")
    
    # Build command with arguments
    cmd = [python_exe, "main.py"]
    cmd.extend(["--strength", str(args.strength)])
    cmd.extend(["--denoise", str(args.denoise)])
    cmd.extend(["--input", args.input])
    cmd.extend(["--output", args.output])
    if args.max_images:
        cmd.extend(["--max-images", str(args.max_images)])
    
    # Run main.py
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            check=False
        )
        sys.exit(result.returncode)
    except Exception as e:
        print(f"\n✗ Error running main.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
