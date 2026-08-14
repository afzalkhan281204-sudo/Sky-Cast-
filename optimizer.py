import numpy as np

class ScratchAdamOptimizer:
    """
    Pure NumPy implementation of the Adam Optimizer with Gradient Clipping.
    """
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, max_grad_norm: float = 1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.t = 0
        
        # State moments for parameters
        self.m = {}
        self.v = {}

    def clip_gradients(self, param_grad_pairs):
        """Clips total gradient norm across all parameters to max_grad_norm."""
        if self.max_grad_norm is None or self.max_grad_norm <= 0:
            return
            
        total_sq_norm = 0.0
        for _, grad in param_grad_pairs:
            total_sq_norm += np.sum(grad ** 2)
            
        total_norm = np.sqrt(total_sq_norm)
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (total_norm + 1e-6)
            for _, grad in param_grad_pairs:
                grad *= scale

    def step(self, param_grad_pairs):
        """
        Updates parameters given a list of (parameter, gradient) tuples.
        """
        self.t += 1
        self.clip_gradients(param_grad_pairs)
        
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)
        
        for idx, (param, grad) in enumerate(param_grad_pairs):
            if idx not in self.m:
                self.m[idx] = np.zeros_like(param)
                self.v[idx] = np.zeros_like(param)
                
            m = self.m[idx]
            v = self.v[idx]
            
            # Update biased 1st and 2nd moment estimates
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
            
            self.m[idx] = m
            self.v[idx] = v
            
            # Apply update
            param -= lr_t * m / (np.sqrt(v) + self.eps)
