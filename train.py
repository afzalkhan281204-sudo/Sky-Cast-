import os
import json
import numpy as np
import time

from lstm_core.dataset import WeatherDatasetGenerator, ScratchStandardScaler, create_sliding_windows
from lstm_core.model import WeatherForecaster
from lstm_core.optimizer import ScratchAdamOptimizer
from lstm_core.uncertainty import UncertaintyEstimator

def serialize_model(model: WeatherForecaster):
    """Converts model NumPy arrays to serializable lists."""
    return {
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'horizon': model.horizon,
        'dropout_rate': model.dropout_rate,
        'W_mu': model.W_mu.tolist(),
        'b_mu': model.b_mu.tolist(),
        'W_var': model.W_var.tolist(),
        'b_var': model.b_var.tolist(),
        'lstm_W': model.lstm.W.tolist(),
        'lstm_b': model.lstm.b.tolist()
    }

def deserialize_model(data: dict) -> WeatherForecaster:
    """Restores model from serialized dictionary."""
    model = WeatherForecaster(
        input_dim=data['input_dim'],
        hidden_dim=data['hidden_dim'],
        horizon=data['horizon'],
        dropout_rate=data['dropout_rate'],
        seed=42
    )
    model.W_mu = np.array(data['W_mu'])
    model.b_mu = np.array(data['b_mu'])
    model.W_var = np.array(data['W_var'])
    model.b_var = np.array(data['b_var'])
    model.lstm.W = np.array(data['lstm_W'])
    model.lstm.b = np.array(data['lstm_b'])
    return model

def train_weather_forecaster(epochs: int = 45, batch_size: int = 32, lr: float = 0.005, ensemble_size: int = 3):
    print("=" * 65)
    print("      TRAINING SCRATCH NUMPY LSTM WEATHER FORECASTER")
    print("=" * 65)
    
    start_time = time.time()
    
    # 1. Dataset Generation
    generator = WeatherDatasetGenerator(n_days=1460, seed=42)
    raw_data, feature_names = generator.generate()
    print(f"[1/5] Generated {len(raw_data)} days of multi-variate weather time series.")
    print(f"      Features ({len(feature_names)}): {', '.join(feature_names)}")
    
    # 2. Scaling & Windowing
    scaler = ScratchStandardScaler()
    scaled_data = scaler.fit_transform(raw_data)
    
    lookback = 30
    horizon = 7
    X, Y = create_sliding_windows(scaled_data, target_col=0, lookback=lookback, horizon=horizon)
    
    # Train / Val Split (80% / 20%)
    split_idx = int(len(X) * 0.8)
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val, Y_val = X[split_idx:], Y[split_idx:]
    
    print(f"[2/5] Created sliding windows: Lookback={lookback} days -> Horizon={horizon} days")
    print(f"      Train Samples: {len(X_train)} | Validation Samples: {len(X_val)}")
    
    models = []
    training_history = []
    
    print(f"[3/5] Training Deep Ensemble of {ensemble_size} Scratch LSTM Models...")
    
    for model_idx in range(ensemble_size):
        seed = 42 + model_idx * 10
        model = WeatherForecaster(
            input_dim=X.shape[-1],
            hidden_dim=24,
            horizon=horizon,
            dropout_rate=0.15,
            seed=seed
        )
        optimizer = ScratchAdamOptimizer(lr=lr, max_grad_norm=1.0)
        
        history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_mae': []}
        num_batches = int(np.ceil(len(X_train) / batch_size))
        
        print(f"\n  --- Model {model_idx + 1}/{ensemble_size} ---")
        
        for epoch in range(1, epochs + 1):
            # Shuffle training batches
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]
            
            epoch_loss = 0.0
            
            for b in range(num_batches):
                b_start = b * batch_size
                b_end = min(len(X_train), (b + 1) * batch_size)
                
                xb = X_shuffled[b_start:b_end]
                yb = Y_shuffled[b_start:b_end]
                
                mu, var = model.forward(xb, training=True)
                loss, dmu, ds_var = model.compute_loss(yb, use_nll=True)
                model.backward(dmu, ds_var)
                optimizer.step(model.get_params())
                
                epoch_loss += loss * len(xb)
                
            train_loss = epoch_loss / len(X_train)
            
            # Validation Evaluation
            val_mu, val_var = model.forward(X_val, training=False)
            val_loss, _, _ = model.compute_loss(Y_val, use_nll=True)
            
            # Unscaled MAE in °C
            val_mu_deg = scaler.inverse_transform_target(val_mu, 0)
            Y_val_deg = scaler.inverse_transform_target(Y_val, 0)
            val_mae = np.mean(np.abs(val_mu_deg - Y_val_deg))
            
            history['epoch'].append(epoch)
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['val_mae'].append(float(val_mae))
            
            if epoch % 10 == 0 or epoch == epochs:
                print(f"    Epoch {epoch:02d}/{epochs:02d} | Train NLL: {train_loss:.4f} | Val NLL: {val_loss:.4f} | Val MAE: {val_mae:.2f}°C")
                
        models.append(model)
        training_history.append(history)
        
    print(f"\n[4/5] Evaluating Ensemble Uncertainty Quantifier on Validation Set...")
    estimator = UncertaintyEstimator(models=models, scaler=scaler, target_col_idx=0)
    eval_res = estimator.predict_with_uncertainty(X_val, num_mc_samples=50)
    
    val_pred_mean = eval_res['mean_forecast']
    val_true_deg = scaler.inverse_transform_target(Y_val, 0)
    
    ensemble_mae = np.mean(np.abs(val_pred_mean - val_true_deg))
    ensemble_rmse = np.sqrt(np.mean((val_pred_mean - val_true_deg) ** 2))
    
    # Coverage probability of 95% CI
    in_ci = (val_true_deg >= eval_res['lower_bound']) & (val_true_deg <= eval_res['upper_bound'])
    picp_95 = np.mean(in_ci) * 100.0
    mean_interval_width = np.mean(eval_res['upper_bound'] - eval_res['lower_bound'])
    
    print(f"      Ensemble Validation MAE  : {ensemble_mae:.2f}°C")
    print(f"      Ensemble Validation RMSE : {ensemble_rmse:.2f}°C")
    print(f"      95% CI Coverage (PICP)  : {picp_95:.1f}%")
    print(f"      Mean 95% Interval Width  : {mean_interval_width:.2f}°C")
    
    # 5. Save Artifacts
    print(f"\n[5/5] Saving model weights and scaler parameters to trained_weights.json...")
    serialized_payload = {
        'scaler_mean': scaler.mean.tolist(),
        'scaler_std': scaler.std.tolist(),
        'feature_names': feature_names,
        'lookback': lookback,
        'horizon': horizon,
        'ensemble_mae': float(ensemble_mae),
        'ensemble_rmse': float(ensemble_rmse),
        'picp_95': float(picp_95),
        'models': [serialize_model(m) for m in models],
        'history': training_history[0],  # Primary model history
        'raw_data_sample': raw_data[-100:].tolist()  # Recent 100 days sample for demo
    }
    
    save_path = "trained_weights.json"
    with open(save_path, "w") as f:
        json.dump(serialized_payload, f, indent=2)
        
    elapsed = time.time() - start_time
    print(f"\nSUCCESS! Training completed in {elapsed:.2f} seconds.")
    print(f"Saved weights payload to: {os.path.abspath(save_path)}")
    print("=" * 65)

if __name__ == "__main__":
    train_weather_forecaster()
