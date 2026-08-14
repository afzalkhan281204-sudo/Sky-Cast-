import numpy as np

def sigmoid(x):
    """Numerically stable sigmoid activation."""
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def sigmoid_backward(sigmoid_out):
    """Derivative of sigmoid given activation output."""
    return sigmoid_out * (1.0 - sigmoid_out)

def tanh_backward(tanh_out):
    """Derivative of tanh given activation output."""
    return 1.0 - tanh_out ** 2


class ScratchLSTMLayer:
    """
    Pure NumPy LSTM Layer implementation with exact analytical BPTT backward pass.
    
    Supports:
    - Multi-batch sequence processing
    - Exact backpropagation through time (BPTT)
    - Optional dropout mask for Monte Carlo (MC) dropout uncertainty estimation
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.0, seed: int = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        if seed is not None:
            np.random.seed(seed)
            
        # Xavier / Glorot Initialization
        # Weight shape: (4 * hidden_dim, input_dim + hidden_dim)
        # Order of gates: [forget_gate, input_gate, candidate_cell, output_gate]
        concat_dim = input_dim + hidden_dim
        limit = np.sqrt(6.0 / (concat_dim + hidden_dim))
        
        self.W = np.random.uniform(-limit, limit, (4 * hidden_dim, concat_dim))
        self.b = np.zeros((4 * hidden_dim,))
        
        # Initialize forget gate bias to 1.0 (helps gradient flow through long sequences)
        self.b[:hidden_dim] = 1.0
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Cache for backpropagation
        self.cache = None

    def forward(self, X: np.ndarray, h0: np.ndarray = None, c0: np.ndarray = None, training: bool = True):
        """
        Forward pass through the LSTM layer over sequence length T.
        
        Parameters:
        -----------
        X : np.ndarray of shape (B, T, D)
            Input sequence tensor (Batch size, Sequence length, Input features)
        h0 : np.ndarray of shape (B, H), optional
            Initial hidden state (default zeros)
        c0 : np.ndarray of shape (B, H), optional
            Initial cell state (default zeros)
        training : bool
            Whether in training mode (applies dropout if dropout_rate > 0)
            
        Returns:
        --------
        H_all : np.ndarray of shape (B, T, H)
            All hidden states across time
        (h_last, c_last) : tuple of np.ndarray
            Final hidden state (B, H) and final cell state (B, H)
        """
        B, T, D = X.shape
        H = self.hidden_dim
        
        if h0 is None:
            h0 = np.zeros((B, H))
        if c0 is None:
            c0 = np.zeros((B, H))
            
        # Storage across time
        H_all = np.zeros((B, T, H))
        C_all = np.zeros((B, T, H))
        
        # Gate caches for BPTT
        gates_cache = []
        z_cache = []
        dropout_masks = []
        
        h_prev = h0
        c_prev = c0
        
        for t in range(T):
            x_t = X[:, t, :]  # (B, D)
            z_t = np.hstack([x_t, h_prev])  # (B, D + H)
            
            # Linear projection: A_t = z_t @ W.T + b -> (B, 4H)
            A_t = z_t @ self.W.T + self.b
            
            # Split into 4 gates
            a_f = A_t[:, :H]
            a_i = A_t[:, H:2*H]
            a_c = A_t[:, 2*H:3*H]
            a_o = A_t[:, 3*H:]
            
            # Apply non-linearities
            f_t = sigmoid(a_f)
            i_t = sigmoid(a_i)
            c_tilde_t = np.tanh(a_c)
            o_t = sigmoid(a_o)
            
            # Cell state update
            c_t = f_t * c_prev + i_t * c_tilde_t
            
            # Hidden state update
            tanh_c_t = np.tanh(c_t)
            h_t = o_t * tanh_c_t
            
            # Apply Dropout if enabled
            mask = None
            if self.dropout_rate > 0.0:
                if training:
                    mask = (np.random.rand(*h_t.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
                    h_t = h_t * mask
                else:
                    # In test mode without MC dropout, mask is 1s
                    mask = np.ones_like(h_t)
            
            dropout_masks.append(mask)
            H_all[:, t, :] = h_t
            C_all[:, t, :] = c_t
            
            gates_cache.append((f_t, i_t, c_tilde_t, o_t, tanh_c_t))
            z_cache.append(z_t)
            
            h_prev = h_t
            c_prev = c_t
            
        # Store cache for backward pass
        self.cache = {
            'X': X,
            'h0': h0,
            'c0': c0,
            'H_all': H_all,
            'C_all': C_all,
            'gates_cache': gates_cache,
            'z_cache': z_cache,
            'dropout_masks': dropout_masks
        }
        
        return H_all, (h_prev, c_prev)

    def backward(self, dH_all: np.ndarray, dh_last: np.ndarray = None, dc_last: np.ndarray = None):
        """
        Backward pass through the LSTM layer using Backpropagation Through Time (BPTT).
        
        Parameters:
        -----------
        dH_all : np.ndarray of shape (B, T, H)
            Upstream gradient w.r.t all hidden states
        dh_last : np.ndarray of shape (B, H), optional
            Upstream gradient w.r.t the final hidden state
        dc_last : np.ndarray of shape (B, H), optional
            Upstream gradient w.r.t the final cell state
            
        Returns:
        --------
        dX : np.ndarray of shape (B, T, D)
            Gradient w.r.t input sequence X
        dh0 : np.ndarray of shape (B, H)
            Gradient w.r.t initial hidden state
        dc0 : np.ndarray of shape (B, H)
            Gradient w.r.t initial cell state
        """
        X = self.cache['X']
        h0 = self.cache['h0']
        c0 = self.cache['c0']
        C_all = self.cache['C_all']
        gates_cache = self.cache['gates_cache']
        z_cache = self.cache['z_cache']
        dropout_masks = self.cache['dropout_masks']
        
        B, T, D = X.shape
        H = self.hidden_dim
        
        # Reset parameter gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        dX = np.zeros_like(X)
        
        dh_next = np.zeros((B, H)) if dh_last is None else dh_last.copy()
        dc_next = np.zeros((B, H)) if dc_last is None else dc_last.copy()
        
        for t in reversed(range(T)):
            h_t_grad = dH_all[:, t, :] + dh_next
            
            # Undo dropout if applied
            mask = dropout_masks[t]
            if mask is not None:
                h_t_grad = h_t_grad * mask
                
            f_t, i_t, c_tilde_t, o_t, tanh_c_t = gates_cache[t]
            c_prev = c0 if t == 0 else C_all[:, t - 1, :]
            
            # Gradient of cell state
            # dh_t = o_t * tanh(c_t) -> dc_t = dc_next + dh_t * o_t * (1 - tanh(c_t)^2)
            dc_t = dc_next + h_t_grad * o_t * (1.0 - tanh_c_t ** 2)
            
            # Gate pre-activation gradients
            da_o = h_t_grad * tanh_c_t * sigmoid_backward(o_t)
            da_c = dc_t * i_t * tanh_backward(c_tilde_t)
            da_i = dc_t * c_tilde_t * sigmoid_backward(i_t)
            da_f = dc_t * c_prev * sigmoid_backward(f_t)
            
            # Concatenate gate derivatives: (B, 4H)
            dA_t = np.hstack([da_f, da_i, da_c, da_o])
            
            # Accumulate weight & bias gradients
            z_t = z_cache[t]  # (B, D + H)
            self.dW += dA_t.T @ z_t  # (4H, D + H)
            self.db += np.sum(dA_t, axis=0)  # (4H,)
            
            # Gradient w.r.t input z_t
            dz_t = dA_t @ self.W  # (B, D + H)
            
            # Split dz_t into dx_t and dh_{t-1}
            dX[:, t, :] = dz_t[:, :D]
            dh_next = dz_t[:, D:]
            
            # Gradient w.r.t previous cell state
            dc_next = dc_t * f_t
            
        dh0 = dh_next
        dc0 = dc_next
        
        return dX, dh0, dc0
