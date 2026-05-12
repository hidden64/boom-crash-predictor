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
    assert data["target_market"] == "CRASH500"
    assert "ia_brain" in data


def test_predict_endpoint_success():
    """
    Test la route principale /predict avec des données factices.
    L'IA doit pouvoir analyser et retourner une probabilité de crash dans [0, 100].
    """
    # 50 ticks simulant un CRASH 500 qui dérive lentement vers le haut.
    mock_ticks = []
    base_time = int(time.time())
    base_price = 1000.0
    for i in range(50):
        mock_ticks.append({
            "timestamp": base_time + i,
            "price": base_price + (i * 0.05)
        })

    payload = {"symbol": "CRASH500", "ticks": mock_ticks}
    response = client.post("/predict", json=payload)

    # 1. Vérification réseau
    assert response.status_code == 200, f"Erreur API: {response.text}"
    data = response.json()

    # 2. Vérification des clés de la réponse
    assert "crash_probability" in data
    assert "alert" in data
    assert "symbol" in data
    assert "signal_level" in data
    assert "recommended_action" in data

    # 3. Validation mathématique de l'IA (probabilité doit être [0, 100])
    assert 0.0 <= data["crash_probability"] <= 100.0

    # 4. L'alerte est un booléen et le niveau correspond aux seuils
    assert isinstance(data["alert"], bool)
    assert data["signal_level"] in {"idle", "warning", "alert"}
    assert data["recommended_action"] in {"SELL", "WAIT"}


def test_predict_endpoint_insufficient_data():
    """Test le comportement si on envoie moins de 2 ticks (impossible de calculer la vélocité)"""
    payload = {
        "symbol": "CRASH500",
        "ticks": [{"timestamp": int(time.time()), "price": 1000.0}]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    # Le comportement défini par inference.py pour array < 2 est de renvoyer prob = 0.0
    assert data["crash_probability"] == 0.0
    assert data["alert"] is False
    assert data["recommended_action"] == "WAIT"
