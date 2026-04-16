# 🚀 Boom-Crash Predictor (Deriv) & La Tour de Contrôle

Bienvenue dans le dépôt du projet **Boom-Crash Predictor**, une architecture complète d'Intelligence Artificielle de Deep Learning couplée à une plateforme web (Tour de Contrôle) pour scruter, analyser et anticiper en temps réel les Spikes des indices synthétiques Boom et Crash de [Deriv](https://deriv.com).

## 🧠 Architecture du Système

Ce projet Full-Stack a été conçu de manière modulaire autour de 4 pôles d'expertise majeurs :

```mermaid
graph TD
    A[Data Pipeline (WebSockets) - Historique] --> B(Modèle IA PyTorch - LSTM)
    B --> C[Backend FastAPI - Moteur d'Inférence]
    C <--> D(Tour de Contrôle - Next.js & Tailwind)
    E(Flux WS Temps Réel Deriv) --> D
```

### 1. Data Pipeline (`/data_pipeline`)
Le collecteur massif de données.
- Connecté en WebSocket à l'API publique Deriv (`app_id=1089`).
- Pagine intelligemment pour drainer l'historique massif des Ticks de marchés (ex: `BOOM1000EZ`).
- Intègre un feature engineering immédiat : calcul de la **Vélocité** et des **Deltas de temps et de prix** (le secret pour deviner un Spike algorithmique).

### 2. Le Cerveau IA (`/ai_model`)
Le cœur du réacteur, basé sur **PyTorch**.
- `train.py` ingurgite les datasets historiques générés par le Pipeline.
- Utilise une architecture **LSTM** (Réseau de Neurones Récurrents) pour analyser les séquences de marché (mémoire court-terme) et prédire le point de rupture.
- Exporte un scaler `scikit-learn` et les poids PT `lstm_spike_predictor.pt` ultra-allégés pour l'inférence.

### 3. Le Backend Inférence API (`/backend`)
L'interface programmatique de l'IA construite avec **FastAPI**.
- Ne charge le modèle lourd en RAM qu'au premier lancement (`singleton`).
- Ouvre la route ultra-rapide `POST /predict`.
- Prend une fenêtre de Ticks brut, recrée la vélocité à la volée, scale les tenseurs et retourne une probabilité de Spike "Tick-par-Tick".

### 4. La Tour de Contrôle (`/frontend`)
Le tableau de bord visuel en **Next.js & TailwindCSS v4**.
- UI "Dark Mode" / Glassmorphism, animée et fluide.
- Ouvre de son côté un second WebSocket à Deriv pour le flux tick par tick asynchrone réel.
- Discute en croisé avec le Backend local pour afficher en temps réel, grâce à la jauge et au graph Recharts, le % de risque de Spike et journalise chaque événement critique (>80%).

---

## 🛠 Pré-requis

- **Python 3.10+** (pour le backend et la modélisation)
- **Node.js 18+** (pour l'interface Next.js)
- *Optionnel mais conseillé* : Carte Graphique Nvidia si vous souhaitez ré-entrainer le modèle massivement. L'inférence marche très bien sur CPU.

## 🚀 Guide de démarrage express

### 1. Installation des dépendances (Une seule fois)

**Pour l'IA / Backend :**
```bash
pip install -r backend/requirements.txt
```
*(Si vous avez des erreurs GPU avec Torch sous windows, cherchez la commande PyTorch dédiée sur leur site officiel).*

**Pour le Frontend Next.js :**
```bash
cd frontend
npm install
cd ..
```

### 2. Entrainement *(Optionnel)*
Si vous ne possédez pas de fichiers dans `ai_model/models_saved/`, commencez par récupérer la Data historique et entraîner le modèle :
```bash
python data_pipeline/deriv_client.py 
python ai_model/train.py
```

### 3. Lancement de la Machine Complète
Pour tout voir prendre vie, utilisez le script d'automatisation Windows placé à la racine :

```bash
# Double cliquez sur ce fichier, ou lancez-le dans le terminal.
start.bat
```
*(Une alternative `docker-compose up -d` est aussi paramétrée mais fortement déconseillée si la gestion du volume PyTorch vous fait peur !)*

### 4. Visualisation
Rendez-vous une fois l'application démarrée sur :
**[http://localhost:3000](http://localhost:3000)**

---

## 🚨 Avertissement
L'utilisation de cette architecture (Bots / IA non validées) pour trader avec des fonds réels est à vos risques et périls. Les marchés Deriv possèdent une base génératrice RNG fortement corrélée dans le temps, mais le risque nul *(Crash prématuré, Slippage)* n'existe pas. Utilisez un `demo_account` dans un premier temps.
