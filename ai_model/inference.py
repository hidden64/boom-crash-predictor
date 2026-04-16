import os
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np

# ==== CONFIG PATHS ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "models_saved")
MODEL_PATH = os.path.join(SAVE_DIR, "lstm_spike_predictor.pt")
SCALER_PATH = os.path.join(SAVE_DIR, "scaler.pkl")

# On redéfinit la même architecture pour que PyTorch puisse charger les poids correctement
class SpikePredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SpikePredictorLSTM, self).__init__()
        # Le dropout est automatiquement désactivé en mode model.eval()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

class BoomPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Chargement du normalisateur (Vital pour la cohérence des Features)
        try:
            self.scaler = joblib.load(SCALER_PATH)
            print("[INFO] Scaler chargé.")
        except Exception as e:
            print(f"[WARNING] Scaler non trouvé ({e}). L'inférence pourrait être faussée.")
            self.scaler = None
            
        # 2. Chargement du Cerveau PyTorch
        self.model = SpikePredictorLSTM(input_size=3).to(self.device)
        try:
            # Map location permet de forcer en CPU si la personne qui run le bot n'a pas de GPU
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
            self.model.eval() # Mode Inférence : on fige les poids pour être ultra-rapide
            print("[INFO] [Cerveau IA] -> Connecté et prêt à prédire.")
        except Exception as e:
            print(f"[WARNING] Modèle PT non trouvé ({e}). Assure-toi que train.py a généré le fichier.")
            
    def predict(self, recent_ticks: list) -> float:
        """
        L'interface finale : reçoit une liste brute de ticks JSON et sort une probabilité.
        """
        if len(recent_ticks) < 2:
            return 0.0 # Impossible de calculer la vélocité sans 2 ticks
            
        df = pd.DataFrame(recent_ticks)
        
        # Feature Engineering à la volée (le même que data_pipeline)
        df['price_change'] = df['price'].diff()
        df['time_delta'] = df['timestamp'].diff()
        df['velocity'] = df['price_change'] / df['time_delta']
        df.fillna(0, inplace=True)
        
        data_values = df[['price_change', 'time_delta', 'velocity']].values
        
        # Le réseau s'attend IMPÉRATIVEMENT à une fenêtre de taille 50.
        if len(data_values) > 50:
            data_values = data_values[-50:] # Garde les 50 les plus frais
        elif len(data_values) < 50:
            # Padding de zéros pour compléter
            pad_size = 50 - len(data_values)
            padding = np.zeros((pad_size, 3))
            data_values = np.vstack((padding, data_values))
            
        # Standardisation avec les paramètres d'entrainement
        if self.scaler:
            data_scaled = self.scaler.transform(data_values)
        else:
            data_scaled = data_values
            
        # Formattage Tenseur : (Batch=1, Séquence=50, Features=3)
        x_tensor = torch.FloatTensor(data_scaled).unsqueeze(0).to(self.device)
        
        # Vitesse maximale (pas de calcul de backpropagation gradients)
        with torch.no_grad():
            logits = self.model(x_tensor)
            # Conversion Logits -> Probabilité % par fonction Sigmoïde (S)
            probability = torch.sigmoid(logits).item()
            
        return probability

# L'instance singleton, le modèle ne sera chargé en RAM qu'une seule fois au lancement du backend
predictor_service = BoomPredictor()
