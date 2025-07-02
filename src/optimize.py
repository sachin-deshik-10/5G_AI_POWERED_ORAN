import argparse
import pandas as pd
import numpy as np
import os
from models.predictive_network_planning.predict import make_predictions
from utils.logger import Logger
from datetime import datetime

def optimize_network_resources(data):
    """
    Optimize network resources based on predictions
    
    Args:
        data: Input DataFrame with predictions
    
    Returns:
        DataFrame with optimization recommendations
    """
    result = data.copy()
    
    # Simple optimization rules
    if 'predicted_dl_kbps' in result.columns:
        # Optimize CPU allocation based on predicted throughput
        result['recommended_cpu_cores'] = np.where(
            result['predicted_dl_kbps'] > 50000, 'Core 4-7', 'Core 0-3'
        )
        
        # Optimize MCS based on throughput predictions
        result['recommended_mcs_dl'] = np.where(
            result['predicted_dl_kbps'] > 70000, 
            np.minimum(result.get('mcs_dl_i', 15) + 2, 31),
            np.maximum(result.get('mcs_dl_i', 15) - 1, 0)
        )
        
        # Energy efficiency recommendations
        result['power_optimization'] = np.where(
            result['predicted_dl_kbps'] < 20000, 'Reduce Power', 'Normal Power'
        )
        
        # Load balancing recommendations
        high_load_mask = result['predicted_dl_kbps'] > 80000
        result['load_balancing'] = np.where(
            high_load_mask, 'Redistribute Load', 'Normal Operation'
        )
    
    return result

def main(args):
    """Main optimization function"""
    
    # Set up logger
    log_file = f"optimization_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    logger = Logger(log_file)
    
    try:
        # Load input data
        logger.log(f"Loading input data from: {args.input}")
        df = pd.read_csv(args.input)
        logger.log(f"Loaded data with shape: {df.shape}")
        
        # Make predictions
        logger.log("Making predictions...")
        predictions = make_predictions(df)
        
        # Optimize resources
        logger.log("Optimizing network resources...")
        optimized_data = optimize_network_resources(predictions)
        
        # Save optimized data
        logger.log(f"Saving optimized data to: {args.output}")
        optimized_data.to_csv(args.output, index=False)
        
        logger.log("Optimization completed successfully")
        print(f"Optimization completed. Results saved to {args.output}")
        
        # Print summary
        if 'predicted_dl_kbps' in optimized_data.columns:
            avg_predicted = optimized_data['predicted_dl_kbps'].mean()
            print(f"Average predicted throughput: {avg_predicted:.2f} kbps")
            
        if 'recommended_cpu_cores' in optimized_data.columns:
            cpu_recommendations = optimized_data['recommended_cpu_cores'].value_counts()
            print("CPU Core Recommendations:")
            for core, count in cpu_recommendations.items():
                print(f"  {core}: {count} instances")
        
        if 'power_optimization' in optimized_data.columns:
            power_recommendations = optimized_data['power_optimization'].value_counts()
            print("Power Optimization Recommendations:")
            for power, count in power_recommendations.items():
                print(f"  {power}: {count} instances")
            
    except Exception as e:
        error_msg = f"Error during optimization: {e}"
        logger.log(error_msg, "error")
        print(error_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI-Powered 5G Open RAN Optimizer optimization")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input real-time data file path")
    parser.add_argument("--models", type=str, default="models", 
                       help="Path to the trained models directory")
    parser.add_argument("--output", type=str, required=True, 
                       help="Output optimized data file path")
    args = parser.parse_args()
    main(args)
