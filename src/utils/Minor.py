import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.models.dynamic_network_optimization import DynamicNetworkOptimization
#from src.dynamic_network_optimization.energy_efficiency_optimization import EnergyEfficiencyOptimization
from src.models.network_anomaly_detection import NetworkAnomalyDetection
#from src.models.predictive_network_planning.predictive_network_planning import PredictiveNetworkPlanning

# 1. Load and Clean Data
def load_synthetic_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    # Clean the data, remove missing values, or apply transformations as needed
    cleaned_data = raw_data.dropna()  # Modify this based on your dataset
    return cleaned_data

# 2. Data Extraction and Transformation
def extract_data(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    # Extract the relevant columns from the dataset
    return cleaned_data[['feature1', 'feature2', 'feature3']]  # Adjust according to your data

def transform_data(extracted_data: pd.DataFrame) -> pd.DataFrame:
    # Scale the data
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(extracted_data), columns=extracted_data.columns)

# 3. Model Initialization & Prediction
def run_dynamic_network_optimization(network: np.ndarray, demand: np.ndarray) -> np.ndarray:
    # Initialize and run the DynamicNetworkOptimization
    dno_model = DynamicNetworkOptimization(network, demand)
    return dno_model.optimal_flow()

def run_energy_efficiency_optimization(data: pd.DataFrame) -> dict:
    # Initialize and run the EnergyEfficiencyOptimization
    eeo_model = EnergyEfficiencyOptimization(config=None, logger=None)  # Assuming a config and logger are required
    return eeo_model.run(data)

def run_network_anomaly_detection(data: pd.DataFrame) -> list:
    # Initialize and run the NetworkAnomalyDetection
    nad_model = NetworkAnomalyDetection()
    return nad_model.detect_anomaly(data)

def run_predictive_network_planning(input_data: np.ndarray) -> np.ndarray:
    # Initialize and run the PredictiveNetworkPlanning model
    pnp_model = PredictiveNetworkPlanning()
    return pnp_model.predict(input_data)

# 4. Plotting Results (if applicable)
def plot_network_flow(flow: np.ndarray):
    plt.figure(figsize=(8, 6))
    plt.plot(flow, label="Network Flow")
    plt.xlabel("Network Nodes")
    plt.ylabel("Flow Value")
    plt.title("Network Flow Optimization")
    plt.legend()
    plt.show()

def plot_energy_efficiency(results: dict):
    # Assuming the results contain 'nodes', 'edges', and 'cost'
    plt.figure(figsize=(8, 6))
    plt.bar(results['nodes'], results['cost'], label="Energy Cost per Node")
    plt.xlabel("Nodes")
    plt.ylabel("Energy Cost")
    plt.title("Energy Efficiency Optimization")
    plt.legend()
    plt.show()

def plot_anomaly_detection(predictions: list):
    plt.figure(figsize=(8, 6))
    plt.plot(predictions, label="Anomaly Detection Predictions", color='red')
    plt.xlabel("Data Points")
    plt.ylabel("Anomaly Status")
    plt.title("Network Anomaly Detection")
    plt.legend()
    plt.show()

# 5. Running the Entire Pipeline
def main():
    # File paths for synthetic data
    raw_data_file = "tests/synthetic_data.csv"  # Modify with your file path
    
    # Step 1: Load and clean data
    raw_data = load_synthetic_data(raw_data_file)
    cleaned_data = clean_data(raw_data)
    
    # Step 2: Extract features and transform data
    extracted_data = extract_data(cleaned_data)
    transformed_data = transform_data(extracted_data)
    
    # Example: Define synthetic network and demand for network optimization
    network = np.array([[0, 10, 15], [10, 0, 35], [15, 35, 0]])  # Example network matrix
    demand = np.array([[0, 200, 100], [200, 0, 300], [100, 300, 0]])  # Example demand matrix
    
    # Step 3: Run Dynamic Network Optimization
    flow = run_dynamic_network_optimization(network, demand)
    print("Optimal Network Flow:", flow)
    plot_network_flow(flow)
    
    # Step 4: Run Energy Efficiency Optimization
    eeo_results = run_energy_efficiency_optimization(transformed_data)
    print("Energy Efficiency Optimization Results:", eeo_results)
    plot_energy_efficiency(eeo_results)
    
    # Step 5: Run Network Anomaly Detection
    nad_predictions = run_network_anomaly_detection(transformed_data)
    print("Network Anomaly Detection Predictions:", nad_predictions)
    plot_anomaly_detection(nad_predictions)
    
    # Step 6: Run Predictive Network Planning
    input_data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # Example data for prediction
    pnp_predictions = run_predictive_network_planning(input_data)
    print("Predictive Network Planning Predictions:", pnp_predictions)

if __name__ == "__main__":
    main()
