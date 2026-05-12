import os
import logging

import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np

# ==== CONFIG PATHS ====
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR    = os.path.join(BASE_DIR, "models_saved")
MODEL_PATH  = os.path.join(SAVE_DIR, "lstm_spike_predictor.pt")
SCALER_PATH = os.path.join(SAVE_DIR, "scaler.pkl")

WINDOW_SIZE = 50
NUM_FEATURES = 6

logger = logging.getLogger("BoomPredictor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Architecture strictement identique à train.py (MLP, malgré le nom historique LSTM)
class SpikePredictorMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SpikePredictorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * WINDOW_SIZE, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


class BoomPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.scaler = None
        self.model = SpikePredictorMLP(input_size=NUM_FEATURES).to(self.device)
        self.model.eval()

        # 1. Chargement du normalisateur
        if os.path.exists(SCALER_PATH):
            try:
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("Scaler chargé depuis %s", SCALER_PATH)
            except Exception as e:
                logger.warning("Impossible de charger le scaler (%s). Inférence non normalisée.", e)
        else:
            logger.warning("Scaler introuvable (%s). Lance d'abord ai_model/train.py.", SCALER_PATH)

        # 2. Chargement du modèle PyTorch
        if os.path.exists(MODEL_PATH):
            try:
                state = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                self.model.eval()
                self.model_loaded = True
                logger.info("[Cerveau IA] -> Connecté et prêt à prédire (device=%s).", self.device)
            except Exception as e:
                logger.warning("Échec du chargement du modèle (%s). Inférence avec poids aléatoires.", e)
        else:
            logger.warning("Modèle PT introuvable (%s). Lance d'abord ai_model/train.py.", MODEL_PATH)

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering identique à train.py (format Deriv)."""
        df['price_change'] = df['price'].diff()
        df['time_delta']   = df['timestamp'].diff()
        # On évite la division par 0 (ticks au même timestamp) comme dans train.py
        safe_delta         = df['time_delta'].replace(0, 0.001)
        df['velocity']     = df['price_change'] / safe_delta

        delta    = df['price'].diff()
        up       = delta.clip(lower=0)
        down     = -1 * delta.clip(upper=0)
        ema_up   = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs       = ema_up / ema_down.replace(0, np.nan)

        df['rsi']    = 100 - (100 / (1 + rs))
        df['ema_9']  = df['price'].ewm(span=9,  adjust=False).mean()
        df['ema_21'] = df['price'].ewm(span=21, adjust=False).mean()

        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def predict(self, recent_ticks: list) -> float:
        """
        Reçoit une liste de dicts [{price, timestamp}, ...] et retourne une probabilité [0, 1].
        """
        if not recent_ticks or len(recent_ticks) < 2:
            return 0.0

        df = pd.DataFrame(recent_ticks)
        df = self._compute_indicators(df)

        features    = ['price_change', 'time_delta', 'velocity', 'rsi', 'ema_9', 'ema_21']
        data_values = df[features].values.astype(np.float32)

        # Fenêtre glissante de taille fixe WINDOW_SIZE
        if len(data_values) > WINDOW_SIZE:
            data_values = data_values[-WINDOW_SIZE:]
        elif len(data_values) < WINDOW_SIZE:
            pad_size    = WINDOW_SIZE - len(data_values)
            padding     = np.zeros((pad_size, len(features)), dtype=np.float32)
            data_values = np.vstack((padding, data_values))

        # Standardisation avec les paramètres d'entraînement
        if self.scaler is not None:
            data_scaled = self.scaler.transform(data_values).astype(np.float32)
        else:
            data_scaled = data_values

        # Tenseur : (Batch=1, Séquence=WINDOW_SIZE, Features=6)
        x_tensor = torch.FloatTensor(np.ascontiguousarray(data_scaled)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits      = self.model(x_tensor)
            probability = torch.sigmoid(logits).item()

        # Sécurité numérique : on borne dans [0, 1]
        if probability != probability:  # NaN
            return 0.0
        return max(0.0, min(1.0, float(probability)))


# Singleton — le modèle n'est chargé en RAM qu'une seule fois au lancement du backend
predictor_service = BoomPredictor()
