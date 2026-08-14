import numpy as np
from lstm_core.lstm_cell import ScratchLSTMLayer, sigmoid

def softplus(x):
    """Numerically stable softplus function: ln(1 + exp(x))."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

class WeatherForecaster:
    """
    Scratch NumPy Sequence-to-Vector LSTM Weather Model with Dual Output Heads.
    
    Predicts 7-day maximum daily temperature (T_max):
    1. Mean head: \hat{\mu} \in R^{B \times 7}
    2. Variance head: \hat{\sigma}^2 \in R^{B \times 7} (for aleatoric uncertainty)
    """
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int = 7, dropout_rate: float = 0.1, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.dropout_rate = dropout_rate
        
        if seed is not None:
            np.random.seed(seed)
            
        # LSTM Layer
        self.lstm = ScratchLSTMLayer(input_dim, hidden_dim, dropout_rate=dropout_rate, seed=seed)
        
        # Dense Mean Head: (horizon, hidden_dim)
        limit_m = np.sqrt(6.0 / (hidden_dim + horizon))
        self.W_mu = np.random.uniform(-limit_m, limit_m, (horizon, hidden_dim))
        self.b_mu = np.zeros((horizon,))
        
        # Dense Variance Head: (horizon, hidden_dim)
        self.W_var = np.random.uniform(-limit_m, limit_m, (horizon, hidden_dim))
        self.b_var = np.zeros((horizon,))
        
        # Gradients
        self.dW_mu = np.zeros_like(self.W_mu)
        self.db_mu = np.zeros_like(self.b_mu)
        self.dW_var = np.zeros_like(self.W_var)
        self.db_var = np.zeros_like(self.b_var)
        
        self.cache = None

    def forward(self, X: np.ndarray, training: bool = True):
        """
        Forward pass over input sequence X (B, T, D).
        
        Returns:
        --------
        mu : np.ndarray of shape (B, horizon)
            Predicted mean maximum temperatures for next 'horizon' days
        var : np.ndarray of shape (B, horizon)
            Predicted variance (aleatoric uncertainty) for next 'horizon' days
        """
        B, T, D = X.shape
        
        # LSTM Forward
        H_all, (h_last, c_last) = self.lstm.forward(X, training=training)
        
        # Linear projection for Mean: (B, horizon)
        mu = h_last @ self.W_mu.T + self.b_mu
        
        # Linear projection for Log-variance pre-activation: (B, horizon)
        s_var = h_last @ self.W_var.T + self.b_var
        
        # Softplus activation for positive variance
        var = softplus(s_var) + 1e-4
        
        self.cache = {
            'X': X,
            'H_all': H_all,
            'h_last': h_last,
            'c_last': c_last,
            'mu': mu,
            's_var': s_var,
            'var': var
        }
        
        return mu, var

    def compute_loss(self, y_true: np.ndarray, use_nll: bool = True):
        """
        Compute Gaussian NLL or MSE loss and return loss scalar + gradients w.r.t mu and s_var.
        
        Parameters:
        -----------
        y_true : np.ndarray of shape (B, horizon)
            Ground truth target temperatures
        use_nll : bool
            If True, uses Gaussian Negative Log-Likelihood loss (NLL).
            If False, uses Mean Squared Error (MSE) loss.
        """
        mu = self.cache['mu']
        var = self.cache['var']
        s_var = self.cache['s_var']
        B, K = y_true.shape
        
        if use_nll:
            # Gaussian NLL: 0.5 * [ (y - mu)^2 / var + ln(var) + ln(2*pi) ]
            nll_elementwise = 0.5 * (((y_true - mu) ** 2) / var + np.log(var) + np.log(2.0 * np.pi))
            loss = np.mean(nll_elementwise)
            
            # Gradients
            # dL / dmu = (mu - y) / (var * B * K)
            dmu = (mu - y_true) / (var * B * K)
            
            # dL / dvar = 0.5 * (1/var - (y - mu)^2 / var^2) / (B * K)
            dvar = 0.5 * (1.0 / var - ((y_true - mu) ** 2) / (var ** 2)) / (B * K)
            
            # dvar / ds_var = sigmoid(s_var)
            ds_var = dvar * sigmoid(s_var)
        else:
            # MSE Loss
            diff = mu - y_true
            loss = np.mean(diff ** 2)
            dmu = (2.0 * diff) / (B * K)
            ds_var = np.zeros_like(s_var)
            
        return loss, dmu, ds_var

    def backward(self, dmu: np.ndarray, ds_var: np.ndarray):
        """
        Backward pass through output heads and LSTM layer.
        """
        h_last = self.cache['h_last']
        H_all = self.cache['H_all']
        B, T, H = H_all.shape
        
        # Gradients for Mean Head
        self.dW_mu = dmu.T @ h_last  # (horizon, H)
        self.db_mu = np.sum(dmu, axis=0)  # (horizon,)
        
        # Gradients for Variance Head
        self.dW_var = ds_var.T @ h_last  # (horizon, H)
        self.db_var = np.sum(ds_var, axis=0)  # (horizon,)
        
        # Upstream gradient to final hidden state h_last
        dh_last = dmu @ self.W_mu + ds_var @ self.W_var  # (B, H)
        
        # Upstream gradient to all hidden states (zero everywhere except last timestep)
        dH_all = np.zeros_like(H_all)
        
        # Backprop through LSTM layer
        dX, dh0, dc0 = self.lstm.backward(dH_all, dh_last=dh_last)
        
        return dX

    def get_params(self):
        """Returns list of (parameter, gradient) tuples for optimizer."""
        return [
            (self.W_mu, self.dW_mu),
            (self.b_mu, self.db_mu),
            (self.W_var, self.dW_var),
            (self.b_var, self.db_var),
            (self.lstm.W, self.lstm.dW),
            (self.lstm.b, self.lstm.db)
        ]
