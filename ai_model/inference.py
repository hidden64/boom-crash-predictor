import os
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

# FIX : le nom et l'architecture correspondent exactement à train.py (c'est un MLP, pas LSTM)
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

        # 1. Chargement du normalisateur
        try:
            self.scaler = joblib.load(SCALER_PATH)
            print("[INFO] Scaler chargé.")
        except Exception as e:
            print(f"[WARNING] Scaler non trouvé ({e}). L'inférence pourrait être faussée.")
            self.scaler = None

        # 2. Chargement du modèle PyTorch
        self.model = SpikePredictorMLP(input_size=6).to(self.device)
        try:
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            print("[INFO] [Cerveau IA] -> Connecté et prêt à prédire.")
        except Exception as e:
            print(f"[WARNING] Modèle PT non trouvé ({e}). Lance d'abord train.py.")

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering identique à train.py."""
        df['price_change'] = df['price'].diff()
        df['time_delta']   = df['timestamp'].diff()
        df['velocity']     = df['price_change'] / df['time_delta'].replace(0, np.nan)

        delta    = df['price'].diff()
        up       = delta.clip(lower=0)
        down     = -1 * delta.clip(upper=0)
        ema_up   = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs       = ema_up / ema_down.replace(0, np.nan)

        df['rsi']    = 100 - (100 / (1 + rs))
        df['ema_9']  = df['price'].ewm(span=9,  adjust=False).mean()
        df['ema_21'] = df['price'].ewm(span=21, adjust=False).mean()

        df.fillna(0, inplace=True)
        return df

    def predict(self, recent_ticks: list) -> float:
        """
        Reçoit une liste de dicts [{price, timestamp}, ...] et retourne une probabilité [0, 1].
        """
        if len(recent_ticks) < 2:
            return 0.0

        df = pd.DataFrame(recent_ticks)
        df = self._compute_indicators(df)

        features    = ['price_change', 'time_delta', 'velocity', 'rsi', 'ema_9', 'ema_21']
        data_values = df[features].values.astype(np.float32)

        # Fenêtre glissante de taille fixe 50
        if len(data_values) > WINDOW_SIZE:
            data_values = data_values[-WINDOW_SIZE:]
        elif len(data_values) < WINDOW_SIZE:
            pad_size    = WINDOW_SIZE - len(data_values)
            padding     = np.zeros((pad_size, len(features)), dtype=np.float32)
            data_values = np.vstack((padding, data_values))

        # Standardisation avec les paramètres d'entraînement
        if self.scaler:
            data_scaled = self.scaler.transform(data_values).astype(np.float32)
        else:
            data_scaled = data_values

        # Tenseur : (Batch=1, Séquence=50, Features=6)
        x_tensor = torch.FloatTensor(np.ascontiguousarray(data_scaled)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits      = self.model(x_tensor)
            probability = torch.sigmoid(logits).item()

        return probability


# Singleton — le modèle n'est chargé en RAM qu'une seule fois au lancement du backend
predictor_service = BoomPredictor()