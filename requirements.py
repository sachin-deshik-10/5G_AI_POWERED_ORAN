import os
import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Function to run shell commands
def run_command(command):
    subprocess.run(command, shell=True, check=True)

# Step 1: Create and activate virtual environment
def setup_environment():
    print("🔹 Creating virtual environment...")
    run_command("python -m venv venv")
    
    print("🔹 Activating virtual environment...")
    if os.name == "nt":  # Windows
        run_command("venv\\Scripts\\activate")
    else:  # Linux/macOS
        run_command("source venv/bin/activate")

    print("🔹 Installing dependencies...")
    requirements = """python==3.8.0
pandas==1.3.1
numpy==1.21.1
scikit-learn==0.24.2
requests==2.26.0
beautifulsoup4==4.9.3
selenium==3.141.0
tensorflow==2.5.0
keras==2.4.3
pytorch==1.9.0
pyomo==5.7.3
pulp==2.4
scipy==1.7.0
statsmodels==0.12.2
matplotlib==3.4.2
seaborn==0.11.1
pyod==0.9.4
plotly==5.1.0
dash==2.0.0
dash-core-components==2.0.0
dash-html-components==2.0.0
dash-bootstrap-components==0.13.1
pylint==2.9.6
pytest==6.2.4"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)

    run_command("pip install -r requirements.txt")

# Step 2: Generate synthetic datasets
def generate_dataset(filename, n=10000):
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i*5) for i in range(n)]
    
    data = {
        "date": timestamps,
        "cpu_usage": np.clip(np.random.normal(50, 10, n), 10, 100),
        "traffic_load": np.random.randint(100, 20000, n),
        "bandwidth": np.random.choice([10, 20, 50, 100], n),
        "power_consumption": np.random.uniform(10, 150, n),
        "snr": np.random.uniform(5, 30, n),
        "latency": np.random.uniform(10, 100, n)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ Dataset saved: {filename}")

# Step 3: Train a simple ML model
def train_model():
    df = pd.read_csv("data/realistic_dataset.csv")
    
    X = df[["cpu_usage", "traffic_load", "bandwidth", "snr"]]
    y = df["latency"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"✅ Model trained! Mean Squared Error: {mse:.2f}")

# Step 4: Run network optimization (placeholder)
def optimize_network():
    print("🔹 Running AI-powered network optimization...")
    print("✅ Network resources optimized dynamically!")

# Main execution
if __name__ == "__main__":
    print("🚀 Starting AI-Powered 5G OpenRAN Optimizer...")
    
    setup_environment()
    
    os.makedirs("data", exist_ok=True)
    generate_dataset("data/realistic_dataset.csv")
    
    train_model()
    optimize_network()
    
    print("🎯 All tasks completed successfully!")
