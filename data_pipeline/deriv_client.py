import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Fix for Windows asyncio loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    import pandas as pd
    import websockets
except ImportError:
    logging.error("Veuillez installer les dépendances avec: pip install -r backend/requirements.txt")
    sys.exit(1)

from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DerivDataPipeline")

# Charger les variables d'environnement
load_dotenv(Path(__file__).parent.parent / ".env")
APP_ID = os.getenv("DERIV_APP_ID", "1089") # 1089 est l'ID public par défaut

async def fetch_historical_ticks(symbol: str = "BOOM1000EZ", total_ticks: int = 100000):
    """
    Se connecte à l'API WebSocket de Deriv pour récupérer massivement des ticks historiques.
    """
    url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
    
    all_prices = []
    all_times = []
    
    # Limite max diminuée pour éviter l'erreur côté serveur Deriv
    chunk_size = 1000 
    end_time = "latest"
    
    logger.info(f"Connexion à Deriv (URL: wss://ws.derivws.com) pour le symbole {symbol}...")
    
    try:
        async with websockets.connect(url) as ws:
            while len(all_prices) < total_ticks:
                request = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": chunk_size,
                    "end": end_time,
                    "start": 1,
                    "style": "ticks"
                }
                
                await ws.send(json.dumps(request))
                response = json.loads(await ws.recv())
                
                if "error" in response:
                    logger.error(f"Erreur renvoyée par l'API: {response['error']['message']}")
                    break
                    
                history = response.get("history", {})
                prices = history.get("prices", [])
                times = history.get("times", [])
                
                if not prices or not times:
                    logger.warning("L'API n'a pas retourné plus de données historiques pour ce symbole.")
                    break
                
                # Deriv API renvoie du plus ancien au plus récent (prix[0] = tick le plus vieux du chunk).
                # Pour paginer en reculant dans le temps, la prochaine époque doit être AVANT le tick le plus vieux reçu.
                oldest_timestamp_in_chunk = times[0]
                end_time = oldest_timestamp_in_chunk - 1
                
                # Ajout en préfixe car on récupère les données à l'envers de l'historique
                all_prices = prices + all_prices
                all_times = times + all_times
                
                logger.info(f"Récupéré un bloc de {len(prices)} ticks. Progression: {len(all_prices)} / {total_ticks} ticks.")
                
                # Temporisation pour ne pas se faire ban IP / Rate limit (Deriv a des limites strictes)
                await asyncio.sleep(0.5)

            # --- Formattage et Sauvegarde ---
            if not all_prices:
                logger.error(f"Aucune donnée récoltée. Vérifie le nom exact du symbole (ex: BOOM1000EZ ou 1HZ100V).")
                return
                
            logger.info("Processus terminé. Formatage des données dans Pandas...")
            
            df = pd.DataFrame({
                "timestamp": all_times,
                "price": all_prices
            })
            
            # Si on a récupéré un peu trop de ticks, on coupe les plus vieux
            if len(df) > total_ticks:
                df = df.tail(total_ticks).reset_index(drop=True)
                
            # Conversion en date lisible
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # --- FEATURE ENGINEERING RAPIDE (Vélocité) ---
            logger.info("Calcul préliminaire de la vélocité et du delta prix...")
            df['price_change'] = df['price'].diff()
            df['time_delta'] = df['timestamp'].diff()
            df['velocity'] = df['price_change'] / df['time_delta']
            
            # Remplacement des NaN sur la première ligne par 0
            df.fillna(0, inplace=True)
                
            # Préparation du dossier de sauvegarde
            save_dir = Path(__file__).parent.parent / "ai_model" / "dataset"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = save_dir / f"{symbol}_history_{len(df)}_ticks.csv"
            
            df.to_csv(save_path, index=False)
            
            logger.info("=" * 60)
            logger.info(f"SUCCÈS : {len(df)} ticks stockés avec vélocité.")
            logger.info(f"Fichier sauvegardé dans : {save_path}")
            logger.info(f"Extrait (5 derniers ticks) :\n{df.tail(5)}")
            logger.info("=" * 60)

    except websockets.exceptions.ConnectionClosedError:
        logger.error("La connexion WebSocket avec Deriv s'est fermée inopinément.")
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")

if __name__ == "__main__":
    logger.info("Démarrage du Data Pipeline asynchrone pour l'historique...")
    # NOTE: Sur Deriv, le symbole officiel pour Boom 1000 est très souvent "BOOM1000EZ".
    # Si cela échoue, tu pourras tester avec "BOOM1000" ou chercher le bon code dans l'API.
    
    try:
        # Exécute la boucle asynchrone principal
        asyncio.run(fetch_historical_ticks(symbol="BOOM1000", total_ticks=100000))
    except KeyboardInterrupt:
        logger.warning("Interrompu par l'utilisateur.")
