#!/usr/bin/env python3
"""
AI-Powered 5G Open RAN Optimizer - Setup Script
==============================================

This script sets up the complete environment for the AI-Powered 5G Open RAN Optimizer.
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data", "models", "logs", "predictions",
        "computing_datasets/datasets_unpin",
        "computing_datasets/datasets_pin", 
        "energy_datasets",
        "application_datasets"
    ]
    
    print("📁 Creating project directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("✅ All directories created!")

def main():
    """Main setup function"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              AI-Powered 5G Open RAN Optimizer               ║
    ║                      SETUP SCRIPT                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        return
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    SETUP COMPLETED! 🎉                      ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🚀 Your AI-Powered 5G Open RAN Optimizer is ready!
    
    Next steps:
    1. Run: python demo.py (for full demonstration)
    2. Or use individual components:
       - python synthetic_data.py (generate test data)
       - python src/preprocess.py --input <data> --output <output>
       - python src/train.py --input <data>
       - python src/evaluate.py --input <data> --models models
       - python src/optimize.py --input <data> --output <output>
    
    📚 Documentation available in docs/ folder
    🐛 Issues? Check logs/ folder for detailed information
    """)

if __name__ == "__main__":
    main()
