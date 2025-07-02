#!/usr/bin/env python3
"""
AI-Powered 5G Open RAN Optimizer - Complete Demo Script
======================================================

This script demonstrates the complete functionality of the AI-Powered 5G Open RAN Optimizer.
It runs through all the major components:
1. Data generation
2. Data preprocessing 
3. Model training
4. Model evaluation
5. Network optimization
"""

import os
import subprocess
import sys
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            print(f"[SUCCESS] {description}")
        else:
            print(f"[FAILED] {description}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"[FAILED] {description} - Exception: {e}")
        return False
    
    return True

def main():
    """Main demo function"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              AI-Powered 5G Open RAN Optimizer               ║
    ║                         FULL DEMO                           ║
    ╚══════════════════════════════════════════════════════════════╝
    
    This demo will showcase all components of the 5G optimization system:
    - Network Anomaly Detection
    - Predictive Network Planning  
    - Dynamic Network Optimization
    - Energy Efficiency Optimization
    """)
    
    start_time = datetime.now()
    
    # Step 1: Generate synthetic data
    if not run_command("python synthetic_data.py", "Generating Synthetic 5G Network Data"):
        return
    
    # Step 2: Data preprocessing
    if not run_command(
        'python src/preprocess.py --input "computing_datasets/datasets_unpin/realistic_computing.csv" --output "demo_preprocessed.csv"',
        "Preprocessing Network Data"
    ):
        return
    
    # Step 3: Train models
    if not run_command(
        'python src/train.py --input "demo_preprocessed.csv"',
        "Training AI Models for Network Optimization"
    ):
        return
    
    # Step 4: Evaluate models
    if not run_command(
        'python src/evaluate.py --input "demo_preprocessed.csv" --models "models"',
        "Evaluating Model Performance"
    ):
        return
    
    # Step 5: Run optimization
    if not run_command(
        'python src/optimize.py --input "computing_datasets/datasets_unpin/realistic_computing.csv" --output "demo_optimized.csv"',
        "Optimizing Network Resources"
    ):
        return
    
    # Step 6: Run full pipeline
    if not run_command(
        'python src/main.py "energy_datasets/dataset_ul.csv"',
        "Running Complete AI Pipeline"
    ):
        return
    
    # Demo complete
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                      DEMO COMPLETED! 🎉                     ║
    ╚══════════════════════════════════════════════════════════════╝
    
    ✅ All components tested successfully!
    ⏱️  Total demo time: {duration.total_seconds():.1f} seconds
    
    📊 Generated Files:
    - demo_preprocessed.csv: Cleaned and transformed data
    - demo_optimized.csv: Network optimization recommendations
    - models/: Trained AI models
    - predictions/: Network performance predictions
    - logs/: Detailed execution logs
    
    🚀 Key Features Demonstrated:
    ✓ Synthetic 5G network data generation
    ✓ Advanced data preprocessing and cleaning
    ✓ Machine learning model training
    ✓ Performance evaluation and metrics
    ✓ Real-time network optimization
    ✓ Energy efficiency recommendations
    ✓ Load balancing suggestions
    
    📈 Next Steps:
    1. Integrate with real 5G network data sources
    2. Deploy models to production environment
    3. Set up real-time monitoring dashboard
    4. Configure automated optimization triggers
    
    For more information, see the README.md file.
    """)

if __name__ == "__main__":
    main()
