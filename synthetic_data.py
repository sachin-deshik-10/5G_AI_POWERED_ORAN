import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta




# Set random seed for reproducibility
np.random.seed(42)

# Create dataset folders
folders = ["computing_datasets/datasets_unpin", "computing_datasets/datasets_pin", "energy_datasets", "application_datasets"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Function to generate timestamps over multiple months
def generate_dates(n, start_date="2024-01-01"):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    return [start + timedelta(minutes=i*5) for i in range(n)]

# Generate Computing Dataset (10,000 rows)
def generate_computing_data(filename, n=10000):
    data = {
        "mcs_dl_i": np.random.randint(1, 30, n),
        "mcs_ul_i": np.random.randint(1, 30, n),
        "dl_kbps_i": np.random.randint(500, 100000, n),  # More realistic traffic loads
        "ul_kbps_i": np.random.randint(500, 50000, n),
        "cpu_set": np.random.choice(["Core 0-3", "Core 4-7"], n),
        "cpu_i": np.clip(np.random.normal(0.5, 0.2, n), 0, 1),  # Gaussian distribution
        "explode": np.random.choice([0, 1], n, p=[0.95, 0.05])  # 5% failure rate
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

generate_computing_data("computing_datasets/datasets_unpin/realistic_computing.csv")
generate_computing_data("computing_datasets/datasets_pin/realistic_computing.csv")

# Generate Energy Dataset (10,000 rows)
def generate_energy_data(filename, n=10000):
    timestamps = generate_dates(n)
    power_consumption = np.clip(np.random.normal(50, 10, n), 10, 100)  # Avg. 50W, ±10W variance
    
    data = {
        "date": timestamps,
        "cpu_platform": np.random.choice(["Intel Xeon", "AMD EPYC"], n),
        "BW": np.random.choice([10, 20, 50, 100], n),
        "UL/DL": np.random.choice(["UL", "DL", "DLUL"], n, p=[0.4, 0.4, 0.2]),
        "TM": np.random.randint(1, 10, n),
        "traffic_load": np.random.randint(500, 20000, n),
        "pm_power": power_consumption,
        "rapl_power": power_consumption * np.random.uniform(0.6, 0.9, n),  # CPU power ~60-90% of total
        "clockspeed": np.random.randint(2000, 4000, n),
        "failed_experiment": np.random.choice([0, 1], n, p=[0.97, 0.03])  # 3% failure rate
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

generate_energy_data("energy_datasets/dataset_ul.csv")
generate_energy_data("energy_datasets/dataset_dlul.csv")

# Generate Application Dataset (10,000 rows)
def generate_application_data(filename, n=10000):
    timestamps = generate_dates(n)
    
    data = {
        "date_exp": timestamps,
        "gpu_platform": np.random.choice(["NVIDIA A100", "NVIDIA RTX 3090"], n),
        "BW": np.random.choice([10, 20, 50, 100], n),
        "img_resolution": np.random.randint(50, 100, n),
        "gpu_power": np.random.randint(50, 300, n),
        "av_end2end_delay": np.round(np.random.exponential(50, n), 2),  # Longer tail for delays
        "av_gpu_delay": np.round(np.random.uniform(1, 50, n), 2),
        "av_num_obj": np.random.randint(1, 15, n),
        "AP1": np.round(np.random.uniform(0.5, 1.0, n), 2),  # Higher precision, realistic values
        "powermeter_av": np.round(np.random.uniform(10, 150, n), 2),
        "rapl_av": np.round(np.random.uniform(5, 80, n), 2),
        "clocksp_av": np.random.randint(2000, 4000, n)
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

generate_application_data("application_datasets/realistic_application.csv")

print("✅ Large, realistic datasets generated successfully!")
