import numpy as np
from lstm_core.model import WeatherForecaster
from lstm_core.dataset import ScratchStandardScaler

class UncertaintyEstimator:
    """
    Quantifies and decomposes 7-day temperature forecast uncertainty into:
    1. Epistemic Uncertainty (Model uncertainty via MC Dropout or Ensemble)
    2. Aleatoric Uncertainty (Inherent weather noise via Variance Head)
    3. Total Predictive Uncertainty & Confidence Intervals (90%, 95%)
    """
    def __init__(self, models: list, scaler: ScratchStandardScaler, target_col_idx: int = 0):
        """
        Parameters:
        -----------
        models : list of WeatherForecaster
            One model (for MC Dropout) or a list of trained ensemble models.
        scaler : ScratchStandardScaler
            Fitted scaler to unscale targets back to real degrees Celsius (°C).
        """
        if not isinstance(models, list):
            models = [models]
        self.models = models
        self.scaler = scaler
        self.target_col_idx = target_col_idx

    def predict_with_uncertainty(self, X_input: np.ndarray, num_mc_samples: int = 100, confidence_level: float = 0.95):
        """
        Runs Monte Carlo inference across model(s) and returns unscaled point forecasts and uncertainty metrics.
        
        Parameters:
        -----------
        X_input : np.ndarray of shape (1, T, D) or (B, T, D)
            Scaled historical weather sequence
        num_mc_samples : int
            Number of stochastic forward passes for MC Dropout
        confidence_level : float
            0.90 for 90% CI or 0.95 for 95% CI
            
        Returns:
        --------
        dict containing:
            - mean_forecast: (B, horizon) in °C
            - epistemic_std: (B, horizon) in °C
            - aleatoric_std: (B, horizon) in °C
            - total_std: (B, horizon) in °C
            - lower_bound: (B, horizon) in °C
            - upper_bound: (B, horizon) in °C
        """
        if X_input.ndim == 2:
            X_input = X_input[np.newaxis, ...]
            
        B, T, D = X_input.shape
        horizon = self.models[0].horizon
        
        all_mu_scaled = []
        all_var_scaled = []
        
        # Scale factors to convert normalized variance to physical °C² variance
        std_target = self.scaler.std[self.target_col_idx]
        mean_target = self.scaler.mean[self.target_col_idx]
        
        samples_per_model = max(1, num_mc_samples // len(self.models))
        
        for model in self.models:
            for _ in range(samples_per_model):
                # Force training=True to keep dropout active during inference
                mu_s, var_s = model.forward(X_input, training=True)
                all_mu_scaled.append(mu_s)
                all_var_scaled.append(var_s)
                
        # Shape: (M, B, horizon)
        all_mu_scaled = np.array(all_mu_scaled)
        all_var_scaled = np.array(all_var_scaled)
        
        # Convert scaled means to physical °C
        all_mu_deg = all_mu_scaled * std_target + mean_target
        # Convert scaled variance to physical °C²: Var(aX) = a^2 * Var(X)
        all_var_deg = all_var_scaled * (std_target ** 2)
        
        # 1. Predictive Mean Forecast: E[mu]
        mean_forecast = np.mean(all_mu_deg, axis=0)  # (B, horizon)
        
        # 2. Epistemic Variance: Var(mu) across stochastic passes/models
        epistemic_var = np.var(all_mu_deg, axis=0)  # (B, horizon)
        epistemic_std = np.sqrt(epistemic_var)
        
        # 3. Aleatoric Variance: E[var] average predicted data noise
        aleatoric_var = np.mean(all_var_deg, axis=0)  # (B, horizon)
        aleatoric_std = np.sqrt(aleatoric_var)
        
        # 4. Total Variance: Var_tot = Var_epi + Var_alea
        total_var = epistemic_var + aleatoric_var
        total_std = np.sqrt(total_var)
        
        # Z-score multiplier for confidence intervals
        z_multiplier = 1.96 if confidence_level >= 0.95 else 1.645
        
        lower_bound = mean_forecast - z_multiplier * total_std
        upper_bound = mean_forecast + z_multiplier * total_std
        
        return {
            'mean_forecast': mean_forecast,
            'epistemic_std': epistemic_std,
            'aleatoric_std': aleatoric_std,
            'total_std': total_std,
            'epistemic_var': epistemic_var,
            'aleatoric_var': aleatoric_var,
            'total_var': total_var,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'z_multiplier': z_multiplier,
            'confidence_level': confidence_level
        }
