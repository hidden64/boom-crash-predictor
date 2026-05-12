import os
import sys
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

# Backend dir + project root sur le sys.path : permet l'import depuis tests, racine ou backend/
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
for _path in (_BACKEND_DIR, _PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from ai_model.inference import predictor_service
except Exception as e:
    print(f"Erreur d'import critique du modèle : {e}")
    predictor_service = None

from database import init_db, save_prediction
from trading_engine import CRASH_ALERT_THRESHOLD, classify_signal

# Création du chef d'orchestre via FastAPI
app = FastAPI(title="CRASH 500 IA Predictor & Trading Engine", version="2.0")

# Initialisation de la base de données
init_db()

# Middleware vital pour autoriser notre futur Front-End (Next.js) à faire des appels AJAX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # On ouvre toutes les portes pour le développement local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === SCHEMAS DE DONNEES (Pydantic) ===
class TickData(BaseModel):
    timestamp: int
    price: float


class PredictRequest(BaseModel):
    symbol: str
    ticks: List[TickData]


# === ROUTES (ENDPOINTS) ===
@app.get("/")
def read_root():
    """Vérification de la pulsation du système"""
    return {
        "status": "online",
        "target_market": "CRASH500",
        "ia_brain": "connected" if predictor_service and predictor_service.model_loaded else "fallback_random",
        "message": "Le chef d'orchestre FastAPI écoute les événements CRASH 500."
    }


@app.post("/predict")
def predict_crash(request: PredictRequest):
    """
    Cœur du réacteur : reçoit une fenêtre de ticks, lance l'inférence,
    et retourne la probabilité d'un crash imminent (vente recommandée).
    """
    if not predictor_service:
        raise HTTPException(status_code=500, detail="Service AI injoignable ou non chargé.")

    start_time = time.time()

    # Conversion du modèle Pydantic strict en liste de dicts pour le data scientist
    ticks_dict = [{"timestamp": t.timestamp, "price": t.price} for t in request.ticks]

    try:
        prob_raw = predictor_service.predict(ticks_dict)
        signal = classify_signal(prob_raw)

        # Si la proba franchit la barre fatidique des 80%, c'est un signal "SELL" potentiel (Crash imminent)
        trigger_alert = bool(prob_raw >= CRASH_ALERT_THRESHOLD)

        # Sauvegarde en base de données si l'alerte est déclenchée
        if trigger_alert and request.ticks:
            latest_tick = request.ticks[-1]
            save_prediction(
                symbol=request.symbol,
                probability=round(prob_raw * 100, 2),
                price=latest_tick.price,
                tick_timestamp=latest_tick.timestamp
            )

        process_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "symbol": request.symbol,
            "crash_probability": round(prob_raw * 100, 2),  # `0.8711` → `87.11`
            "alert": trigger_alert,
            "signal_level": signal.level,
            "recommended_action": "SELL" if signal.should_sell else "WAIT",
            "inference_time_ms": process_time_ms
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur fatale lors de l'inférence : {str(e)}")


if __name__ == "__main__":
    # Pour le démarrer localement : cd backend && python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
