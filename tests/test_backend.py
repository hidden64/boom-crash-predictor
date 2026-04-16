import pytest
import time
from fastapi.testclient import TestClient

# Modification du sys.path pour permettre l'importation locale depuis /backend
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app

client = TestClient(app)

def test_healthcheck():
    """Vérifie que l'API démarre et répond avec le code HTTP 200"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "ia_brain" in data

def test_predict_endpoint_success():
    """
    Test la route principale /predict avec des données factices parfaites.
    L'IA devrait pouvoir analyser et retourner une probabilité mathématique >= 0.
    """
    # Création de 50 ticks d'exemple (le minimum vital pour le Feature Engineering)
    # On simule un marché qui monte légèrement
    mock_ticks = []
    base_time = int(time.time())
    base_price = 10000.0
    for i in range(50):
        mock_ticks.append({
            "timestamp": base_time + i,
            "price": base_price + (i * 0.1)
        })

    payload = {
        "symbol": "BOOM1000",
        "ticks": mock_ticks
    }

    response = client.post("/predict", json=payload)
    
    # 1. Vérification réseau
    assert response.status_code == 200, f"Erreur API: {response.text}"
    
    data = response.json()
    
    # 2. Vérification des clés de la réponse
    assert "spike_probability" in data
    assert "alert" in data
    assert "symbol" in data
    
    # 3. Validation mathématique de l'IA (probabilité doit être [0, 100])
    assert 0.0 <= data["spike_probability"] <= 100.0
    
    # 4. L'alerte est un booleen
    assert isinstance(data["alert"], bool)

def test_predict_endpoint_insufficient_data():
    """Test le comportement si on envoie moins de 2 ticks (impossible de calculer la vélocité)"""
    payload = {
        "symbol": "BOOM1000",
        "ticks": [{"timestamp": int(time.time()), "price": 10000.0}]
    }

    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    # Le comportement défini par inference.py pour array < 2 est de renvoyer prob = 0.0
    assert data["spike_probability"] == 0.0
    assert data["alert"] is False
