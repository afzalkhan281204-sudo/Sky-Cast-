import numpy as np
from lstm_core.model import WeatherForecaster

def check_lstm_gradients(batch_size: int = 2, seq_len: int = 5, input_dim: int = 3, hidden_dim: int = 4, horizon: int = 7, eps: float = 1e-5):
    """
    Performs analytical vs finite-difference numerical gradient check on Scratch WeatherForecaster model.
    Returns max relative error across all parameters.
    """
    np.random.seed(123)
    model = WeatherForecaster(input_dim=input_dim, hidden_dim=hidden_dim, horizon=horizon, dropout_rate=0.0, seed=123)
    
    X = np.random.randn(batch_size, seq_len, input_dim)
    Y = np.random.randn(batch_size, horizon)
    
    # 1. Analytical Forward & Backward Pass
    model.forward(X, training=False)
    loss, dmu, ds_var = model.compute_loss(Y, use_nll=True)
    model.backward(dmu, ds_var)
    
    param_grad_pairs = model.get_params()
    max_rel_error = 0.0
    errors_summary = []
    
    for idx, (param, analytical_grad) in enumerate(param_grad_pairs):
        num_grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            ix = it.multi_index
            old_val = param[ix]
            
            # f(w + eps)
            param[ix] = old_val + eps
            model.forward(X, training=False)
            loss_plus, _, _ = model.compute_loss(Y, use_nll=True)
            
            # f(w - eps)
            param[ix] = old_val - eps
            model.forward(X, training=False)
            loss_minus, _, _ = model.compute_loss(Y, use_nll=True)
            
            # Reset parameter
            param[ix] = old_val
            
            # Central finite difference
            num_grad[ix] = (loss_plus - loss_minus) / (2.0 * eps)
            
            it.iternext()
            
        # Compute relative error
        diff = np.abs(analytical_grad - num_grad)
        denom = np.maximum(1e-8, np.abs(analytical_grad) + np.abs(num_grad))
        rel_error = np.max(diff / denom)
        
        if rel_error > max_rel_error:
            max_rel_error = rel_error
            
        errors_summary.append((idx, param.shape, rel_error))
        
    return max_rel_error, errors_summary

if __name__ == "__main__":
    max_err, summary = check_lstm_gradients()
    print("--- LSTM Gradient Verification ---")
    for idx, shape, err in summary:
        print(f"Param {idx} (shape {shape}): Max Relative Error = {err:.8e}")
    print(f"Overall Max Relative Error: {max_err:.8e}")
    if max_err < 1e-5:
        print("GRADIENT CHECK PASSED SUCCESSFULY!")
    else:
        print("GRADIENT CHECK FAILED!")
