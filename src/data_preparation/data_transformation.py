import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml
import os

def read_config(config_file):
    """Read configuration from YAML file"""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def transform_data(input_path: str, output_path: str) -> None:
    """
    Transform data based on configuration
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save transformed data
    """
    # Load data from input file
    df = pd.read_csv(input_path)
    
    # Check if config file exists
    config_file = "config.yml"
    if os.path.exists(config_file):
        try:
            config = read_config(config_file)
            
            # Apply data transformations if configured
            if "data_transformation" in config and "columns" in config["data_transformation"]:
                for col_config in config["data_transformation"]["columns"]:
                    col_name = col_config["name"]
                    transform_type = col_config["type"]
                    
                    if col_name in df.columns:
                        if transform_type == "log":
                            df[col_name] = df[col_name].apply(lambda x: np.log(x) if x > 0 else 0)
                        elif transform_type == "sqrt":
                            df[col_name] = df[col_name].apply(lambda x: np.sqrt(x) if x >= 0 else 0)
                        elif transform_type == "inverse":
                            df[col_name] = df[col_name].apply(lambda x: 1/x if x > 0 else 0)
        except Exception as e:
            print(f"Warning: Could not apply config-based transformations: {e}")
    
    # Scale numeric data
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # Save transformed data to output file
    df.to_csv(output_path, index=False)
    print(f"Data transformation completed. Output saved to {output_path}")

