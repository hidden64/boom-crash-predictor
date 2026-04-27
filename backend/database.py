import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "alerts.db")

def init_db():
    """Initialise la base de données SQLite pour stocker les alertes."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            probability REAL NOT NULL,
            price REAL NOT NULL,
            tick_timestamp INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(symbol: str, probability: float, price: float, tick_timestamp: int):
    """Enregistre une prédiction dépassant le seuil dans la base de données."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (symbol, probability, price, tick_timestamp)
        VALUES (?, ?, ?, ?)
    ''', (symbol, probability, price, tick_timestamp))
    conn.commit()
    conn.close()
