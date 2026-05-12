# 🚨 CRASH 500 Predictor (Deriv) & La Tour de Contrôle

Architecture complète d'Intelligence Artificielle de Deep Learning couplée à une plateforme web (Tour de Contrôle) pour scruter, analyser et anticiper en temps réel les **Crashs** de l'indice synthétique **CRASH 500** de [Deriv](https://deriv.com).

> Sur CRASH 500, le marché dérive lentement vers le **haut**… puis chute violemment d'un coup. L'IA cherche à détecter ce point de rupture juste avant qu'il ne se produise pour permettre de **vendre / shorter** au bon moment.

## 🧠 Architecture du Système

Le projet est divisé en 4 pôles :

```mermaid
graph TD
    A[Data Pipeline (WebSockets) - Historique CRASH500] --> B(Modèle IA PyTorch)
    B --> C[Backend FastAPI - Moteur d'Inférence]
    C <--> D(Tour de Contrôle - Next.js & Tailwind)
    E(Flux WS Temps Réel Deriv) --> D
```

### 1. Data Pipeline (`/data_pipeline`)
Le collecteur massif de données.
- Connecté en WebSocket à l'API publique Deriv (`app_id=1089`).
- Pagine intelligemment pour drainer l'historique massif des ticks `CRASH500`.
- Calcule à la volée la **vélocité** (négative durant un crash) et les **deltas** de temps / prix.

### 2. Le Cerveau IA (`/ai_model`)
Le cœur du réacteur, basé sur **PyTorch**.
- `train.py` ingurgite les datasets historiques générés par le pipeline.
- MLP profond avec une fenêtre glissante de 50 ticks pour prédire le crash dans les **5 ticks** suivants.
- Exporte le scaler `scikit-learn` et les poids PyTorch (`crash_predictor.pt`).

### 3. Le Backend Inférence API (`/backend`)
L'interface programmatique de l'IA construite avec **FastAPI**.
- Charge le modèle en RAM une seule fois (`singleton`).
- Route ultra-rapide `POST /predict`.
- Retourne `crash_probability`, `signal_level` (`idle`/`warning`/`alert`) et `recommended_action` (`SELL`/`WAIT`).
- Persiste les alertes >80 % dans SQLite (`backend/alerts.db`).

### 4. La Tour de Contrôle (`/frontend`)
Le tableau de bord en **Next.js & TailwindCSS v4**.
- UI dark / glassmorphism, palette rouge alignée sur le contexte CRASH.
- WebSocket Deriv via Web Worker (pas de bridage en onglet inactif).
- Affiche en temps réel : jauge probabilité, graphique tick-par-tick, historique des alertes (>80 %) et des prémices (60-79 %).

---

## 🛠 Pré-requis

- **Python 3.10+**
- **Node.js 18+**
- *Optionnel* : GPU Nvidia pour réentraîner massivement. L'inférence tourne très bien sur CPU.

## 🚀 Guide de démarrage express

### 1. Installation (une seule fois)

```bash
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..
```

### 2. Entraînement *(optionnel si vous avez déjà `ai_model/models_saved/`)*

```bash
python data_pipeline/deriv_client.py
python ai_model/train.py
```

### 3. Lancement complet (Windows)

```bash
start.bat
```

Alternative : `docker-compose up -d`.

### 4. Visualisation

→ **http://localhost:3000**

---

## 🔌 API Backend

`POST /predict`

```json
{
  "symbol": "CRASH500",
  "ticks": [
    { "timestamp": 1715520000, "price": 980.12 },
    { "timestamp": 1715520001, "price": 980.18 }
  ]
}
```

Réponse :

```json
{
  "symbol": "CRASH500",
  "crash_probability": 87.4,
  "alert": true,
  "signal_level": "alert",
  "recommended_action": "SELL",
  "inference_time_ms": 4.21
}
```

---

## 🚨 Avertissement
L'utilisation de cette architecture pour trader avec des fonds réels reste à vos risques et périls. Les marchés synthétiques Deriv possèdent une base RNG fortement corrélée dans le temps, mais le risque zéro n'existe pas (crash prématuré, slippage). **Toujours tester en compte démo d'abord.**
