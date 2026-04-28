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
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SAVE_DIR    = os.path.join(BASE_DIR, "models_saved")

WINDOW_SIZE      = 50
PREDICT_AHEAD    = 5
BATCH_SIZE       = 256
EPOCHS           = 10
LEARNING_RATE    = 0.001
SPIKE_THRESHOLD  = 1.0


# ==== DATASET ====
class BoomCrashDataset(Dataset):
    def __init__(self, data_scaled, targets, window_size):
        # ascontiguousarray garantit un bloc mémoire contigu AVANT la conversion Tensor
        self.data        = torch.FloatTensor(np.ascontiguousarray(data_scaled))
        self.targets     = torch.FloatTensor(np.ascontiguousarray(targets))
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # .clone() garantit un bloc mémoire contigu — vital pour éviter les segfaults
        x_window = self.data[idx : idx + self.window_size].clone()
        y_label  = self.targets[idx + self.window_size - 1].clone()
        return x_window, y_label


# ==== ARCHITECTURE MLP ====
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


# ==== FEATURE ENGINEERING ====
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    delta    = df['price'].diff()
    up       = delta.clip(lower=0)
    down     = -1 * delta.clip(upper=0)
    ema_up   = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs       = ema_up / ema_down.replace(0, np.nan)

    df['rsi']   = 100 - (100 / (1 + rs))
    df['ema_9']  = df['price'].ewm(span=9,  adjust=False).mean()
    df['ema_21'] = df['price'].ewm(span=21, adjust=False).mean()

    return df


# ==== PRÉPARATION DES DONNÉES ====
def prepare_data():
    print("1. Recherche du dataset le plus récent...")
    files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun dataset CSV trouvé dans {DATASET_DIR}")

    file_path = max(files, key=os.path.getctime)
    print(f"   --> Chargement de {file_path}")

    # Détection automatique du séparateur (MT5 = tab, Deriv = virgule)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
    separator = '\t' if '\t' in first_line else ','

    df = pd.read_csv(file_path, sep=separator, engine='c')

    # Adaptation au format MetaTrader 5 s'il est détecté
    if '<DATE>' in df.columns:
        print("   --> Format MetaTrader détecté. Conversion des données...")
        df.columns = [c.replace('<', '').replace('>', '') for c in df.columns]

        # Sécurité RAM : Limite pour éviter un dépassement (OOM) sur les PC avec moins de 32GB
        if len(df) > 2000000:
            print(f"   --> [Alerte RAM] Dataset volumineux ({len(df)} ticks). Restriction aux 2 derniers millions.")
            df = df.tail(2000000).copy()

        # Reconstitution des colonnes nécessaires
        df['price'] = df['BID']
        df['timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S.%f')
        df['price_change'] = df['price'].diff()
        df['time_delta'] = df['timestamp'].diff().dt.total_seconds()
        df['velocity'] = df['price_change'] / df['time_delta'].replace(0, 0.001)
        df.dropna(subset=['price_change', 'time_delta'], inplace=True)
    else:
        print("   --> Format Deriv standard détecté.")

    print("2. Calcul des étiquettes (spikes)...")
    df['is_spike'] = (df['price_change'] >= SPIKE_THRESHOLD).astype(int)
    num_spikes     = df['is_spike'].sum()
    print(f"   --> Ticks totaux  : {len(df)}")
    print(f"   --> Spikes détectés : {num_spikes} ({(num_spikes / len(df)) * 100:.2f}%)")

    df['target'] = (
        df['is_spike']
        .rolling(window=PREDICT_AHEAD)
        .max()
        .shift(-PREDICT_AHEAD)
    )

    print("3. Génération des indicateurs (RSI, EMA9, EMA21)...")
    df = add_indicators(df)
    df.fillna(0, inplace=True)

    features    = ['price_change', 'time_delta', 'velocity', 'rsi', 'ema_9', 'ema_21']
    data_values = df[features].values
    targets     = df['target'].fillna(0).values.astype(np.float32)

    print("4. Normalisation...")
    scaler      = StandardScaler()
    data_scaled = scaler.fit_transform(data_values).astype(np.float32)

    split_idx     = int(0.8 * len(data_scaled))
    train_data    = data_scaled[:split_idx]
    train_targets = targets[:split_idx]
    test_data     = data_scaled[split_idx:]
    test_targets  = targets[split_idx:]

    print(f"   --> Entraînement : {len(train_data)} lignes")
    print(f"   --> Test         : {len(test_data)} lignes")

    neg_count  = float((train_targets == 0).sum())
    pos_count  = float((train_targets == 1).sum())
    # FIX : on plafonne à 100 pour éviter une explosion numérique du gradient
    pos_weight = min(neg_count / max(pos_count, 1.0), 100.0)
    print(f"   --> Poids de compensation calculé : {pos_weight:.2f}")

    return train_data, train_targets, test_data, test_targets, scaler, pos_weight, len(features)


# ==== ÉVALUATION ====
def evaluate(model, data_scaled, targets, device):
    dataset = BoomCrashDataset(data_scaled, targets, window_size=WINDOW_SIZE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=False)

    model.eval()
    correct = total = tp = fp = fn = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            preds = (torch.sigmoid(model(batch_x)) >= 0.5).float()

            correct += (preds == batch_y).sum().item()
            total   += batch_y.size(0)
            tp      += ((preds == 1) & (batch_y == 1)).sum().item()
            fp      += ((preds == 1) & (batch_y == 0)).sum().item()
            fn      += ((preds == 0) & (batch_y == 1)).sum().item()

    accuracy  = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    return accuracy, precision, recall, f1


# ==== ENTRAÎNEMENT ====
def train_model():
    train_data, train_targets, test_data, test_targets, scaler, pos_weight, num_features = prepare_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[!] Puce utilisée : {device}")

    train_dataset = BoomCrashDataset(train_data, train_targets, window_size=WINDOW_SIZE)

    # num_workers=0 est OBLIGATOIRE sur Windows pour éviter le segfault multiprocessing
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    model = SpikePredictorMLP(input_size=num_features).to(device)

    # FIX PRINCIPAL : pos_weight passé directement à BCEWithLogitsLoss
    # Cela remplace le torch.where() manuel qui causait le segfault dans le backward C++
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n==== DÉMARRAGE DE L'ENTRAÎNEMENT ====")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(batch_x)
            # Calcul direct sans torch.where — stable sur tous les backends CPU/GPU
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Loss : {avg_loss:.4f}")

    print("\n==== ENTRAÎNEMENT TERMINÉ ====")

    # Évaluation finale sur le set de test
    print("\n==== ÉVALUATION SUR LE SET DE TEST ====")
    accuracy, precision, recall, f1 = evaluate(model, test_data, test_targets, device)
    print(f"   Accuracy  : {accuracy  * 100:.2f}%")
    print(f"   Precision : {precision * 100:.2f}%")
    print(f"   Recall    : {recall    * 100:.2f}%")
    print(f"   F1-Score  : {f1        * 100:.2f}%")

    # Sauvegarde
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path  = os.path.join(SAVE_DIR, "lstm_spike_predictor.pt")
    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n[OK] Modèle exporté  → {model_path}")
    print(f"[OK] Scaler exporté  → {scaler_path}")


if __name__ == '__main__':
    train_model()