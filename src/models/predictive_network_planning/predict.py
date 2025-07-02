import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def make_predictions(data):
    """
    Make predictions using trained model or create simple predictions
    
    Args:
        data: Input DataFrame
    
    Returns:
        DataFrame with predictions
    """
    try:
        # Try to load pre-trained model
        model_path = "models/predictive_network_planning_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Prepare features (assuming we predict dl_kbps_i based on other features)
            features = ['mcs_dl_i', 'mcs_ul_i', 'ul_kbps_i', 'cpu_i']
            available_features = [col for col in features if col in data.columns]
            
            if available_features:
                X = data[available_features]
                predictions = model.predict(X)
                
                result = data.copy()
                result['predicted_dl_kbps'] = predictions
                return result
    except Exception as e:
        print(f"Could not load model: {e}")
    
    # Fallback: Create simple rule-based predictions
    result = data.copy()
    
    # Simple prediction based on MCS and current throughput
    if 'mcs_dl_i' in data.columns and 'ul_kbps_i' in data.columns:
        result['predicted_dl_kbps'] = (
            data['mcs_dl_i'] * 2000 + data['ul_kbps_i'] * 0.8
        ).clip(lower=0)
    else:
        result['predicted_dl_kbps'] = np.random.uniform(10000, 100000, len(data))
    
    return result

def train_model(data):
    """
    Train a predictive model
    
    Args:
        data: Training DataFrame
    """
    try:
        # Prepare features and target
        features = ['mcs_dl_i', 'mcs_ul_i', 'ul_kbps_i', 'cpu_i']
        target = 'dl_kbps_i'
        
        available_features = [col for col in features if col in data.columns]
        
        if target not in data.columns or len(available_features) == 0:
            print("Required columns not found for training")
            return None
        
        X = data[available_features]
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained successfully. MSE: {mse:.2f}, R2: {r2:.2f}")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/predictive_network_planning_model.pkl")
        
        return model
        
    except Exception as e:
        print(f"Error training model: {e}")
        return None

