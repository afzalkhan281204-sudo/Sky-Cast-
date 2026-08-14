import http.server
import socketserver
import json
import os
import urllib.parse
import numpy as np

from lstm_core.dataset import ScratchStandardScaler, WeatherDatasetGenerator
from lstm_core.uncertainty import UncertaintyEstimator
from train import deserialize_model

PORT = 8080
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

class WeatherAppHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/api/forecast':
            self.handle_get_forecast()
        elif parsed_path.path == '/api/historical':
            self.handle_get_historical()
        else:
            # Serve static files
            super().do_GET()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/api/simulate':
            self.handle_post_simulate()
        else:
            self.send_error(404, "Endpoint not found")

    def handle_get_forecast(self):
        weights_path = os.path.join(os.path.dirname(__file__), 'trained_weights.json')
        if not os.path.exists(weights_path):
            self.send_json({'error': 'Model not trained yet. Please run train.py.'}, status=500)
            return

        with open(weights_path, 'r') as f:
            payload = json.load(f)

        scaler = ScratchStandardScaler()
        scaler.mean = np.array(payload['scaler_mean'])
        scaler.std = np.array(payload['scaler_std'])
        models = [deserialize_model(m_dict) for m_dict in payload['models']]
        estimator = UncertaintyEstimator(models=models, scaler=scaler, target_col_idx=0)

        # Get recent 30 days from raw dataset sample
        raw_sample = np.array(payload['raw_data_sample'])  # (100, 6)
        lookback = payload['lookback']
        recent_30 = raw_sample[-lookback:]

        # Run inference
        scaled_input = scaler.transform(recent_30)[np.newaxis, ...]
        res = estimator.predict_with_uncertainty(scaled_input, num_mc_samples=100)

        response_data = {
            'lookback_days': recent_30[:, 0].tolist(),
            'lookback_min': recent_30[:, 1].tolist(),
            'lookback_humidity': recent_30[:, 2].tolist(),
            'lookback_pressure': recent_30[:, 3].tolist(),
            'mean_forecast': res['mean_forecast'][0].tolist(),
            'epistemic_std': res['epistemic_std'][0].tolist(),
            'aleatoric_std': res['aleatoric_std'][0].tolist(),
            'total_std': res['total_std'][0].tolist(),
            'lower_bound': res['lower_bound'][0].tolist(),
            'upper_bound': res['upper_bound'][0].tolist(),
            'ensemble_mae': payload['ensemble_mae'],
            'ensemble_rmse': payload['ensemble_rmse'],
            'picp_95': payload['picp_95'],
            'history': payload['history']
        }
        self.send_json(response_data)

    def handle_post_simulate(self):
        content_len = int(self.headers.get('Content-Length', 0))
        post_body = self.rfile.read(content_len)
        data = json.loads(post_body.decode('utf-8'))

        weights_path = os.path.join(os.path.dirname(__file__), 'trained_weights.json')
        if not os.path.exists(weights_path):
            self.send_json({'error': 'Model not trained yet'}, status=500)
            return

        with open(weights_path, 'r') as f:
            payload = json.load(f)

        scaler = ScratchStandardScaler()
        scaler.mean = np.array(payload['scaler_mean'])
        scaler.std = np.array(payload['scaler_std'])
        models = [deserialize_model(m_dict) for m_dict in payload['models']]
        estimator = UncertaintyEstimator(models=models, scaler=scaler, target_col_idx=0)

        # Baseline recent 30 days
        raw_sample = np.array(payload['raw_data_sample'])
        lookback = payload['lookback']
        modified_30 = raw_sample[-lookback:].copy()

        # Apply simulation deltas from user
        temp_delta = float(data.get('temp_delta', 0.0))
        humidity_delta = float(data.get('humidity_delta', 0.0))
        pressure_delta = float(data.get('pressure_delta', 0.0))
        noise_level = float(data.get('noise_level', 0.0))

        # Apply perturbation to recent 7 days of lookback
        modified_30[-7:, 0] += temp_delta
        modified_30[-7:, 2] = np.clip(modified_30[-7:, 2] + humidity_delta, 10.0, 100.0)
        modified_30[-7:, 3] += pressure_delta

        if noise_level > 0:
            np.random.seed(int(time.time() * 1000) % 100000)
            modified_30[-7:] += np.random.normal(0, noise_level, modified_30[-7:].shape)

        scaled_input = scaler.transform(modified_30)[np.newaxis, ...]
        res = estimator.predict_with_uncertainty(scaled_input, num_mc_samples=100)

        response_data = {
            'modified_tmax': modified_30[:, 0].tolist(),
            'mean_forecast': res['mean_forecast'][0].tolist(),
            'epistemic_std': res['epistemic_std'][0].tolist(),
            'aleatoric_std': res['aleatoric_std'][0].tolist(),
            'total_std': res['total_std'][0].tolist(),
            'lower_bound': res['lower_bound'][0].tolist(),
            'upper_bound': res['upper_bound'][0].tolist()
        }
        self.send_json(response_data)

    def handle_get_historical(self):
        weights_path = os.path.join(os.path.dirname(__file__), 'trained_weights.json')
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                payload = json.load(f)
            self.send_json({'raw_data': payload['raw_data_sample'], 'feature_names': payload['feature_names']})
        else:
            self.send_json({'error': 'Data not ready'}, status=404)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

def run_server():
    server_address = ('', PORT)
    with socketserver.TCPServer(server_address, WeatherAppHandler) as httpd:
        print(f"Weather App Server running at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == '__main__':
    run_server()
