import os
import sys
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

# Inclusion manuelle du module parent pour pouvoir appeler `ai_model` depuis `backend`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ai_model.inference import predictor_service
except Exception as e:
    print(f"Erreur d'import critique du modèle : {e}")
    predictor_service = None

# Import du module de base de données
from database import init_db, save_prediction

# Création du chef d'orchestre via FastAPI
app = FastAPI(title="Boom/Crash IA Predictor & Trading Engine", version="1.0")

# Initialisation de la base de données
init_db()

# Middleware vital pour autoriser notre futur Front-End (Next.js) à faire des appels AJAX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # On ouvre toutes les portes pour le dévelopement local
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
        "ia_brain": "connected" if predictor_service else "failed_to_load",
        "message": "Le chef d'orchestre FastAPI écoute les événements."
    }

@app.post("/predict")
def predict_spike(request: PredictRequest):
    """
    Coeur du réacteur. Reçoit un flux asynchrone, lance l'inférence Deep Learning, 
    et calcule si la position d'Achat/Vente numérique doit se fermer .
    """
    if not predictor_service:
         raise HTTPException(status_code=500, detail="Service AI injoignable ou non chargé.")
         
    start_time = time.time()
    
    # Conversion du model métier Pydantic strict vers tableau de données pour le data sciencist
    ticks_dict = [{"timestamp": t.timestamp, "price": t.price} for t in request.ticks]
    
    try:
        # APPEL MAGIQUE - On questionne le modèle 
        prob_raw = predictor_service.predict(ticks_dict)
        
        # Logic Métier : Alerte
        # Si la proba franchit la barre fatidique des 80%, c'est un signal "BUY" potentiel (Boom)
        trigger_alert = bool(prob_raw >= 0.80)
        
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
            "spike_probability": round(prob_raw * 100, 2), # Transforme `0.8711` en `87.11`
            "alert": trigger_alert,
            "inference_time_ms": process_time_ms
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur fatale lors de l'inférence : {str(e)}")

if __name__ == "__main__":
    # Pour le démarrer localement : cd backend && python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
