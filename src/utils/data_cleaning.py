# src/utils/data_cleaning.py

import pandas as pd

def drop_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows with null values from the given DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame with rows containing null values removed.
    """
    # Drop rows with any null values
    df_cleaned = df.dropna()
    return df_cleaned
