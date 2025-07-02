import argparse
import pandas as pd
import os
from models.predictive_network_planning.predict import train_model
from utils.logger import Logger
from datetime import datetime

def main(args):
    """Main training function"""
    
    # Set up logger
    log_file = f"training_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    logger = Logger(log_file)
    
    try:
        # Load training data
        logger.log(f"Loading training data from: {args.input}")
        df = pd.read_csv(args.input)
        logger.log(f"Loaded data with shape: {df.shape}")
        
        # Train predictive network planning model
        logger.log("Training predictive network planning model...")
        model = train_model(df)
        
        if model is not None:
            logger.log("Model training completed successfully")
            print("Model training completed successfully")
        else:
            logger.log("Model training failed")
            print("Model training failed")
            
    except Exception as e:
        error_msg = f"Error during training: {e}"
        logger.log(error_msg, "error")
        print(error_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train machine learning models")
    parser.add_argument("--input", type=str, required=True, 
                       help="Path to training data CSV file")
    parser.add_argument("--output", type=str, default="models", 
                       help="Output directory for trained models")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    main(args)
