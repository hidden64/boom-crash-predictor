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

# ==== PYTORCH DATASET ====
class BoomCrashDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==== ARCHITECTURE DU MODELE PREDICITF (LSTM) ====
class SpikePredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SpikePredictorLSTM, self).__init__()
        # Le réseau de neurone LSTM est parfait pour capturer la "mémoire" des ticks
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Couche dense qui finit sur 1 seul neurone (Probabilité 0 ou 1)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # On prend la mémoire du TOUT DERNIER tick de notre fenêtre
        out = self.fc(out)
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
    
    # 2. Étiquetage (Labeling) - Identifier les spikes
    print("2. Calcul des étiquettes (Spikes)...")
    # Pour le Boom 1000, le prix saute vers le HAUT
    df['is_spike'] = (df['price_change'] >= SPIKE_THRESHOLD).astype(int)
    
    num_spikes = df['is_spike'].sum()
    print(f"--> Ticks totaux : {len(df)}")
    print(f"--> Spikes détectés : {num_spikes} ({(num_spikes/len(df))*100:.2f}% du marché)")
    
    # Cible Y : Le marché va-t-il 'spiker' dans la prochaine fraction de temps ?
    # Le rolling permet de vérifier s'il va y avoir au moins 1 spike dans le futur proche
    df['target'] = df['is_spike'].rolling(window=PREDICT_AHEAD).max().shift(-PREDICT_AHEAD)
    df.dropna(inplace=True)
    
    # 3. Features matrix (Que sait le modèle ?)
    features = ['price_change', 'time_delta', 'velocity']
    data_values = df[features].values
    targets = df['target'].values
    
    print("3. Normalisation et création des fenêtres (Sliding Windows)...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    X, y = [], []
    for i in range(len(data_scaled) - WINDOW_SIZE):
        X.append(data_scaled[i : i + WINDOW_SIZE])
        y.append(targets[i + WINDOW_SIZE - 1])
        
    X = np.array(X)
    y = np.array(y)
    
    # 4. Chronological Split (Train/Test)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"--> Set d'Entraînement : {X_train.shape}")
    print(f"--> Set de Test : {X_test.shape}")
    
    # Calcul des poids pour compenser la rareté des spikes
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    pos_weight = float(neg_count) / max(float(pos_count), 1.0)
    print(f"--> Poids de compensation (Focal weight) calculé : {pos_weight:.2f}")
    
    return X_train, X_test, y_train, y_test, scaler, pos_weight

# ==== BOUCLE D'ENTRAINEMENT ====
def train_model():
    X_train, X_test, y_train, y_test, scaler, pos_weight = prepare_data()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[!] Utilisation de la puce : {device}")
    
    train_dataset = BoomCrashDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SpikePredictorLSTM(input_size=X_train.shape[2]).to(device)
    
    # BCEWithLogitsLoss avec pos_weight est la méthode la plus performante pour déséquilibre extrême
    weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n==== DÉMARRAGE DE L'ENTRAÎNEMENT ====")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{EPOCHS} -> Loss : {epoch_loss/len(train_loader):.4f}")
        
    print("\n==== ENTRAÎNEMENT TERMINÉ ====")
    
    # Sauvegarde des objets vitaux pour l'inférence temps réel
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, "lstm_spike_predictor.pt")
    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    
    torch.save(model.state_dict(), model_path)
    joblib.save(scaler, scaler_path)
    
    print(f"[SUCCESS] Le cerveau IA a été exporté vers {model_path}.")
    print(f"[SUCCESS] Le normalisateur a été exporté vers {scaler_path}.")

if __name__ == '__main__':
    train_model()
