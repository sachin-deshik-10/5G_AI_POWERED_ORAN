import argparse
import pandas as pd
from data_preparation.data_extraction import DataExtractor
from data_preparation.data_cleaning import clean_data
from data_preparation.data_transformation import transform_data

def main(args):
    """Main preprocessing function"""
    
    # Initialize data extractor
    extractor = DataExtractor()
    
    # Load data
    if args.input.startswith("http"):
        # If input is a URL, download and extract
        url_parts = args.input.split(",")
        if len(url_parts) == 2:
            url, filename = url_parts
            extractor.extract_data(url, filename)
            raw_data_path = f"{extractor.data_dir}/{filename.replace('.zip', '.csv')}"
        else:
            print("Error: URL input should be in format 'url,filename'")
            return
    else:
        # Local file path
        raw_data_path = args.input
    
    print(f"Loading data from: {raw_data_path}")
    
    # Read and clean data
    try:
        raw_data = extractor.read_csv(raw_data_path)
        print(f"Loaded data with shape: {raw_data.shape}")
        
        # Clean data
        print("Cleaning data...")
        cleaned_data = clean_data(raw_data)
        print(f"Cleaned data shape: {cleaned_data.shape}")
        
        # Save cleaned data temporarily for transformation
        temp_cleaned_path = "temp_cleaned_data.csv"
        cleaned_data.to_csv(temp_cleaned_path, index=False)
        
        # Transform data
        print("Transforming data...")
        transform_data(temp_cleaned_path, args.output)
        
        print(f"Data preprocessing completed. Output saved to {args.output}")
        
        # Clean up temporary file
        import os
        if os.path.exists(temp_cleaned_path):
            os.remove(temp_cleaned_path)
            
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing pipeline")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input raw data file path or URL,filename")
    parser.add_argument("--output", type=str, required=True, 
                       help="Output preprocessed data file path")
    args = parser.parse_args()
    main(args)
