#!/usr/bin/env python3
"""
Run script for Facial Anonymization project.
Activates virtual environment and runs main2.py
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

def main():
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
    print(f"\nStarting Facial Anonymization workflow...\n")
    
    # Run main.py
    try:
        result = subprocess.run(
            [python_exe, "main.py"],
            cwd=Path(__file__).parent,
            check=False
        )
        sys.exit(result.returncode)
    except Exception as e:
        print(f"\n✗ Error running main.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
