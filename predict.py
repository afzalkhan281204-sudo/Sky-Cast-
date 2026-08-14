import json
import os
import sys
import numpy as np

from lstm_core.dataset import ScratchStandardScaler, WeatherDatasetGenerator
from lstm_core.uncertainty import UncertaintyEstimator
from train import deserialize_model

def load_trained_system(weights_path: str = "trained_weights.json"):
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found. Please run 'python train.py' first.")
        sys.exit(1)
        
    with open(weights_path, "r") as f:
        payload = json.load(f)
        
    scaler = ScratchStandardScaler()
    scaler.mean = np.array(payload['scaler_mean'])
    scaler.std = np.array(payload['scaler_std'])
    
    models = [deserialize_model(m_dict) for m_dict in payload['models']]
    estimator = UncertaintyEstimator(models=models, scaler=scaler, target_col_idx=0)
    
    return estimator, scaler, payload

def run_cli_forecast():
    print("=" * 70)
    print("      SCRATCH NUMPY LSTM WEATHER FORECASTER - 7-DAY INFERENCE")
    print("=" * 70)
    
    estimator, scaler, payload = load_trained_system()
    feature_names = payload['feature_names']
    lookback = payload['lookback']
    
    # Generate 30 days of recent weather observation
    generator = WeatherDatasetGenerator(n_days=lookback + 10, seed=100)
    raw_data, _ = generator.generate()
    recent_30_days = raw_data[-lookback:]  # (30, 6)
    
    # Scale sequence
    scaled_sequence = scaler.transform(recent_30_days)[np.newaxis, ...]  # (1, 30, 6)
    
    print(f"Input Weather History: Last {lookback} Days Observations")
    print(f"Recent T_max range: {np.min(recent_30_days[:, 0]):.1f}°C to {np.max(recent_30_days[:, 0]):.1f}°C (Average: {np.mean(recent_30_days[:, 0]):.1f}°C)\n")
    
    # Predict 7-day forecast with uncertainty estimation
    results = estimator.predict_with_uncertainty(scaled_sequence, num_mc_samples=100, confidence_level=0.95)
    
    mean_fc = results['mean_forecast'][0]
    epi_std = results['epistemic_std'][0]
    alea_std = results['aleatoric_std'][0]
    tot_std = results['total_std'][0]
    lower_95 = results['lower_bound'][0]
    upper_95 = results['upper_bound'][0]
    
    print("+" + "-"*68 + "+")
    print(f"| Day |  Fcst T_max (°C)  | Epistemic σ | Aleatoric σ | Total σ | 95% Conf. Interval  |")
    print("+" + "-"*68 + "+")
    
    for d in range(7):
        print(f"|  +{d+1}  |    {mean_fc[d]:6.2f} °C    |   {epi_std[d]:5.2f} °C   |   {alea_std[d]:5.2f} °C   | {tot_std[d]:5.2f} °C | [{lower_95[d]:5.1f}°C - {upper_95[d]:5.1f}°C] |")
        
    print("+" + "-"*68 + "+")
    print(f"\nUncertainty Summary:")
    print(f"  • Average Epistemic (Model) Uncertainty:  {np.mean(epi_std):.2f}°C")
    print(f"  • Average Aleatoric (Data Noise) Uncertainty: {np.mean(alea_std):.2f}°C")
    print(f"  • Overall Average 95% Interval Width:    {np.mean(upper_95 - lower_95):.2f}°C")
    print("=" * 70)

if __name__ == "__main__":
    run_cli_forecast()
