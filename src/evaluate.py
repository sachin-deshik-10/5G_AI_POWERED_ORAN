import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from utils.logger import Logger
from datetime import datetime

def evaluate_model(data, model_path):
    """
    Evaluate trained model on test data
    
    Args:
        data: Test DataFrame
        model_path: Path to trained model
    
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Load trained model
        model = joblib.load(model_path)
        
        # Prepare features and target
        features = ['mcs_dl_i', 'mcs_ul_i', 'ul_kbps_i', 'cpu_i']
        target = 'dl_kbps_i'
        
        available_features = [col for col in features if col in data.columns]
        
        if target not in data.columns or len(available_features) == 0:
            print("Required columns not found for evaluation")
            return None
        
        X = data[available_features]
        y_true = data[target]
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_samples': len(y_true)
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def main(args):
    """Main evaluation function"""
    
    # Set up logger
    log_file = f"evaluation_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    logger = Logger(log_file)
    
    try:
        # Load evaluation data
        logger.log(f"Loading evaluation data from: {args.input}")
        df = pd.read_csv(args.input)
        logger.log(f"Loaded data with shape: {df.shape}")
        
        # Check if model exists
        if not os.path.exists(args.models):
            logger.log(f"Model directory not found: {args.models}", "error")
            print(f"Model directory not found: {args.models}")
            return
        
        # Evaluate predictive network planning model
        model_path = os.path.join(args.models, "predictive_network_planning_model.pkl")
        
        if os.path.exists(model_path):
            logger.log("Evaluating predictive network planning model...")
            metrics = evaluate_model(df, model_path)
            
            if metrics:
                logger.log("Model evaluation completed successfully")
                print("Model Evaluation Results:")
                print(f"MSE: {metrics['mse']:.4f}")
                print(f"RMSE: {metrics['rmse']:.4f}")
                print(f"MAE: {metrics['mae']:.4f}")
                print(f"R²: {metrics['r2']:.4f}")
                print(f"Samples: {metrics['n_samples']}")
                
                # Save results
                results_file = f"evaluation_results_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
                with open(results_file, 'w') as f:
                    f.write("Model Evaluation Results\n")
                    f.write("========================\n")
                    f.write(f"MSE: {metrics['mse']:.4f}\n")
                    f.write(f"RMSE: {metrics['rmse']:.4f}\n")
                    f.write(f"MAE: {metrics['mae']:.4f}\n")
                    f.write(f"R²: {metrics['r2']:.4f}\n")
                    f.write(f"Samples: {metrics['n_samples']}\n")
                
                logger.log(f"Results saved to {results_file}")
            else:
                logger.log("Model evaluation failed")
                print("Model evaluation failed")
        else:
            logger.log(f"Model not found: {model_path}", "error")
            print(f"Model not found: {model_path}")
            
    except Exception as e:
        error_msg = f"Error during evaluation: {e}"
        logger.log(error_msg, "error")
        print(error_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI-Powered 5G Open RAN Optimizer models")
    parser.add_argument("--input", type=str, required=True, 
                       help="Evaluation data file path")
    parser.add_argument("--models", type=str, required=True, 
                       help="Path to the trained models directory")
    args = parser.parse_args()
    main(args)
