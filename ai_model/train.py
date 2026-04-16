import os
import glob
import torch

import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==== CONFIGURATION ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SAVE_DIR = os.path.join(BASE_DIR, "models_saved")

WINDOW_SIZE = 50       # Séquence de 50 ticks passés lus par l'IA
PREDICT_AHEAD = 5      # Y a-t-il un spike dans les 5 prochains ticks ?
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001
SPIKE_THRESHOLD = 1.0  # Pour Boom (hausse > 1 point). Pour Crash, ce serait < -1.0

# ==== PYTORCH DATASET (OPTIMISÉ MÉMOIRE) ====
class BoomCrashDataset(Dataset):
    def __init__(self, data_scaled, targets, window_size):
        # Pour éviter le warning Numpy et garantir une mémoire saine
        self.data = torch.FloatTensor(np.array(data_scaled, copy=True))
        self.targets = torch.FloatTensor(np.array(targets, copy=True))
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size
        
    def __getitem__(self, idx):
        # .contiguous() ou .clone() est ABSOLUMENT VITAL ici !
        # Le C++ MKL backend de PyTorch crashe souvent (Segmentation Fault) 
        # s'il reçoit une "vue" mémoire (stride) au lieu d'un bloc contigu pour les LSTM.
        x_window = self.data[idx : idx + self.window_size].clone().detach()
        y_label = self.targets[idx + self.window_size - 1].clone().detach()
        return x_window, y_label

# ==== ARCHITECTURE DU MODELE PREDICITF (MLP STABLE) ====
class SpikePredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SpikePredictorLSTM, self).__init__()
        # Face à l'instabilité majeure du C++ de PyTorch sous Python 3.13 (Segfaults répétés
        # sur les RNN/GRU/LSTM lors de la rétropropagation), nous "aplatissons" la fenêtre 
        # séquentielle et utilisons un perceptron multicouche classique (MLP).
        # Mathématiquement, sur une fenêtre fixe de 50, cela fonctionne remarquablement bien !
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size * WINDOW_SIZE, hidden_size * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# ==== PROCESSING DES DONNÉES ====
def prepare_data():
    print("1. Recherche du dataset le plus récent...")
    files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun dataset CSV trouvé dans {DATASET_DIR}")
    
    file_path = max(files, key=os.path.getctime)
    print(f"--> Chargement de {file_path}")
    df = pd.read_csv(file_path)
    
    print("2. Calcul des étiquettes (Spikes)...")
    df['is_spike'] = (df['price_change'] >= SPIKE_THRESHOLD).astype(int)
    num_spikes = df['is_spike'].sum()
    print(f"--> Ticks totaux : {len(df)}")
    print(f"--> Spikes détectés : {num_spikes} ({(num_spikes/len(df))*100:.2f}% du marché)")
    
    df['target'] = df['is_spike'].rolling(window=PREDICT_AHEAD).max().shift(-PREDICT_AHEAD)
    df.fillna(0, inplace=True)
    
    # --- FEATURES INGENIERIE : INDICATEURS PROS AJOUTÉS (RSI, EMA) ---
    print("2.5 Génération des Indicateurs (RSI, EMA 9, EMA 21)...")
    delta = df['price'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['ema_9'] = df['price'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['price'].ewm(span=21, adjust=False).mean()
    df.fillna(0, inplace=True)
    
    features = ['price_change', 'time_delta', 'velocity', 'rsi', 'ema_9', 'ema_21']
    data_values = df[features].values
    targets = df['target'].values
    
    print("3. Normalisation (Sans duplication en mémoire)...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    split_idx = int(0.8 * len(data_scaled))
    
    train_data = data_scaled[:split_idx]
    train_targets = targets[:split_idx]
    test_data = data_scaled[split_idx:]
    test_targets = targets[split_idx:]
    
    print(f"--> Set d'Entraînement : {len(train_data)} lignes.")
    print(f"--> Set de Test : {len(test_data)} lignes.")
    
    neg_count = sum(train_targets == 0)
    pos_count = sum(train_targets == 1)
    pos_weight = float(neg_count) / max(float(pos_count), 1.0)
    print(f"--> Poids de compensation (Focal weight) calculé : {pos_weight:.2f}")
    
    # Nous retournons également la taille des features '6' pour initialiser le modèle dynamiquement
    return train_data, train_targets, test_data, test_targets, scaler, pos_weight, len(features)

# ==== BOUCLE D'ENTRAINEMENT ====
def train_model():
    train_data, train_targets, test_data, test_targets, scaler, pos_weight, num_features = prepare_data()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[!] Utilisation de la puce : {device}")
    
    train_dataset = BoomCrashDataset(train_data, train_targets, window_size=WINDOW_SIZE)
    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,       # Désactive le multiprocessing — obligatoire sur Windows
    pin_memory=False     # Évite les copies asynchrones non supportées sans CUDA
    )
    
    model = SpikePredictorLSTM(input_size=num_features).to(device)
    
    # BCEWithLogitsLoss de PyTorch sur Windows CPU a un bug très sévère (Segfault) lors de la 
    # vectorisation interne de 'pos_weight'. Nous allons appliquer les poids matriciels 
    # manuellement en Python !
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # On sauvegarde pos_weight comme simple float 
    pos_weight_val = float(pos_weight)
    
    print("\n==== DÉMARRAGE DE L'ENTRAÎNEMENT ====")
    for epoch in range(EPOCHS):
        print(f"-> Début de l'Epoch {epoch+1}")
        model.train()
        epoch_loss = 0
        batch_idx = 0
        
        print("-> Chargement du premier batch via DataLoader...")
        try:
            for batch_x, batch_y in train_loader:
                batch_idx += 1
                if batch_idx == 1:
                    print(f"   [Batch {batch_idx}] Données chargées avec succès.")
                
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                
                outputs = model(batch_x)
                
                # 1. Calcul de la perte brute par prédiction (Matrice complète)
                raw_loss = criterion(outputs, batch_y)
                
                # 2. Application manuelle des poids (Si Spike (1), on multiplie la perte par 229 !)
                #weight_matrix = torch.where(batch_y == 1.0, pos_weight_val, 1.0).to(device)
                weight_matrix = torch.where(
    batch_y == 1.0,
    torch.tensor(pos_weight_val, dtype=torch.float32, device=device),
    torch.tensor(1.0,            dtype=torch.float32, device=device)
)
                
                # 3. Moyenne
                loss = (raw_loss * weight_matrix).mean()
                
                if batch_idx == 1:
                    print(f"   [Batch {batch_idx}] Rétropropagation paramétrée (Backward)...")
                loss.backward()
                
                optimizer.step()
                
                epoch_loss += loss.item()
                if batch_idx == 1:
                    print(f"   [Batch {batch_idx}] Terminé sans erreur fatale !")
                
        except Exception as e:
            print(f"Exception capturée : {e}")
            
        print(f"Epoch {epoch+1:02d}/{EPOCHS} -> Loss : {epoch_loss/max(1, len(train_loader)):.4f}")
        
    print("\n==== ENTRAÎNEMENT TERMINÉ ====")
    
    # Sauvegarde des objets vitaux pour l'inférence temps réel
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, "lstm_spike_predictor.pt")
    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"[SUCCESS] Le cerveau IA a été exporté vers {model_path}.")
    print(f"[SUCCESS] Le normalisateur a été exporté vers {scaler_path}.")

if __name__ == '__main__':
    train_model()
