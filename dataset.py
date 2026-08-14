import numpy as np

class ScratchStandardScaler:
    """
    Pure NumPy Standard Scaler for 2D and 3D sequence arrays.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """Fit mean and std on 2D (N, D) or 3D (B, T, D) data."""
        if data.ndim == 3:
            # Flatten batch and time
            flat = data.reshape(-1, data.shape[-1])
        else:
            flat = data
            
        self.mean = np.mean(flat, axis=0)
        self.std = np.std(flat, axis=0)
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform_target(self, scaled_target: np.ndarray, feature_idx: int = 0) -> np.ndarray:
        """Inverse transforms target variable (e.g. T_max at feature_idx 0)."""
        mean = self.mean[feature_idx]
        std = self.std[feature_idx]
        return scaled_target * std + mean


class WeatherDatasetGenerator:
    """
    Generates realistic multi-variate weather time series with physical dynamics:
    - Annual seasonality (harmonics)
    - Auto-regressive thermal inertia
    - Diurnal temperature range & humidity cross-correlations
    - Weather front perturbations (stochastic cold/warm waves)
    """
    def __init__(self, n_days: int = 1460, seed: int = 42):
        self.n_days = n_days
        self.seed = seed

    def generate(self):
        np.random.seed(self.seed)
        t = np.arange(self.n_days)
        
        # Base annual temperature cycle (Mean ~20C, amplitude ~12C)
        annual_phase = 2.0 * np.pi * t / 365.25
        base_temp = 20.0 + 12.0 * np.sin(annual_phase - 1.2)
        
        # Multi-day atmospheric wave perturbations (synoptic weather patterns)
        synoptic_wave = 4.0 * np.sin(2.0 * np.pi * t / 14.0) + 2.0 * np.cos(2.0 * np.pi * t / 5.5)
        
        # AR(1) thermal memory
        ar_noise = np.zeros(self.n_days)
        for i in range(1, self.n_days):
            ar_noise[i] = 0.75 * ar_noise[i-1] + np.random.normal(0.0, 1.5)
            
        t_max = base_temp + synoptic_wave + ar_noise + 3.0
        t_min = t_max - np.random.uniform(6.0, 12.0, size=self.n_days)
        
        # Relative humidity: inversely related to temperature + noise
        humidity = np.clip(75.0 - 1.2 * (t_max - 20.0) + np.random.normal(0, 8.0, size=self.n_days), 20.0, 98.0)
        
        # Atmospheric pressure: High pressure during cold clear days, low pressure during storms
        pressure = 1013.25 - 0.3 * synoptic_wave + np.random.normal(0, 4.0, size=self.n_days)
        
        # Solar radiation (W/m^2): Driven by season & humidity (clouds)
        solar = np.clip((250.0 + 120.0 * np.sin(annual_phase)) * (1.0 - 0.005 * humidity) + np.random.normal(0, 20.0, size=self.n_days), 10.0, 400.0)
        
        # Wind speed (km/h)
        wind = np.clip(12.0 + 0.5 * np.abs(synoptic_wave) + np.random.exponential(4.0, size=self.n_days), 2.0, 60.0)
        
        # Stack features: [T_max, T_min, Humidity, Pressure, Solar, Wind]
        data = np.column_stack([t_max, t_min, humidity, pressure, solar, wind])
        
        feature_names = ['T_max (°C)', 'T_min (°C)', 'Humidity (%)', 'Pressure (hPa)', 'Solar (W/m²)', 'Wind (km/h)']
        return data, feature_names


def create_sliding_windows(data: np.ndarray, target_col: int = 0, lookback: int = 30, horizon: int = 7):
    """
    Creates input sequences (X) and multi-step targets (Y).
    
    Parameters:
    -----------
    data : np.ndarray of shape (N, D)
        Multi-variate weather time-series
    target_col : int
        Index of the target variable to predict (0 for T_max)
    lookback : int
        Historical lookback window T_in (e.g. 30 days)
    horizon : int
        Forecast horizon T_out (e.g. 7 days)
        
    Returns:
    --------
    X : np.ndarray of shape (Num_samples, lookback, D)
    Y : np.ndarray of shape (Num_samples, horizon)
    """
    N, D = data.shape
    num_samples = N - lookback - horizon + 1
    
    X = np.zeros((num_samples, lookback, D))
    Y = np.zeros((num_samples, horizon))
    
    for i in range(num_samples):
        X[i] = data[i : i + lookback]
        Y[i] = data[i + lookback : i + lookback + horizon, target_col]
        
    return X, Y
