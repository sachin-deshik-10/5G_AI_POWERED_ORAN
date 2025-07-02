import argparse
import os
from datetime import datetime
from utils.logger import Logger
from data_preparation.data_extraction import extract_data
from data_preparation.data_cleaning import clean_data
from data_preparation.data_transformation import transform_data
from models.predictive_network_planning.predict import make_predictions

def main(args):
    # Set up logger
    log_file = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    logger = Logger(log_file)
    
    # Extract/load data
    logger.log("Loading data...")
    raw_data = extract_data(args.data_file)
    
    # Clean data
    logger.log("Cleaning data...")
    cleaned_data = clean_data(raw_data)
    
    # Save cleaned data temporarily
    temp_cleaned_path = "temp_cleaned_data.csv"
    cleaned_data.to_csv(temp_cleaned_path, index=False)
    
    # Transform data
    logger.log("Transforming data...")
    temp_transformed_path = "temp_transformed_data.csv"
    transform_data(temp_cleaned_path, temp_transformed_path)
    
    # Load transformed data for predictions
    import pandas as pd
    transformed_data = pd.read_csv(temp_transformed_path)
    
    # Make predictions
    logger.log("Making predictions...")
    predictions = make_predictions(transformed_data)
    
    # Save predictions to file
    logger.log("Saving predictions to file...")
    os.makedirs("predictions", exist_ok=True)
    predictions_file = f"predictions/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
    predictions.to_csv(predictions_file, index=False)
    
    logger.log(f"Finished. Predictions saved to {predictions_file}")
    
    # Clean up temporary files
    if os.path.exists(temp_cleaned_path):
        os.remove(temp_cleaned_path)
    if os.path.exists(temp_transformed_path):
        os.remove(temp_transformed_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI-Powered 5G OpenRAN Optimizer")
    parser.add_argument("data_file", type=str, help="Path to the raw data file")
    args = parser.parse_args()
    
    main(args)

