# tests/test_1.py
import pandas as pd
from src.utils.data_cleaning import drop_null_rows
import sys
import os

# Get the project root directory and add 'src' to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_cleaning import drop_null_rows  # Now it should work

df = pd.read_csv("computing_datasets/datasets_unpin/realistic_computing.csv")
df_cleaned = drop_null_rows(df)
df_cleaned.to_csv("cleaned_data.csv", index=False)

print("✅ Data Cleaning Test Passed!")
