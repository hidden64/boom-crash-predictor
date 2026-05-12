# Guide du Trader Algorithmique — CRASH 500 Predictor 🧠📉

Ce document est votre bible pour maîtriser la machine construite de bout en bout. Objectif : un **système de trading manuel assisté par IA** capable de prédire les célèbres **crashs** (chutes brutales) de l'indice synthétique **CRASH 500** de Deriv.

---

## 🏗️ Rappel du Concept

Pour qu'une IA devine le futur, elle doit d'abord apprendre le passé. L'application est divisée en 3 grandes phases, à suivre dans l'ordre si vous partez de zéro :

1. **La Collecte (Data Pipeline)** : on aspire des dizaines de milliers de ticks CRASH 500 pour constituer une base d'entraînement.
2. **L'Entraînement (AI Model)** : on enferme l'IA dans une salle pour qu'elle apprenne à reconnaître l'empreinte précédant une chute.
3. **Le Direct (Backend & Tour de Contrôle)** : on branche l'IA entraînée sur le flux temps réel et on observe via une interface premium Next.js.

---

## ⚙️ Phase 1 : Collecter des données historiques

```bash
python data_pipeline/deriv_client.py
```

Ce script ouvre une connexion WebSocket avec Deriv et aspire en rafale jusqu'à **1 000 000 ticks** de `CRASH500` en reculant dans le temps. Il calcule la vélocité (négative durant les crashs) et enregistre le tout en CSV dans `ai_model/dataset/`.

> Astuce : si le marché change de comportement, relancez la collecte pour réentraîner sur du frais.

---

## 🧠 Phase 2 : Forger et entraîner le cerveau

```bash
python ai_model/train.py
```

Ce que fait ce script :
- Lit le CSV le plus récent.
- Détecte automatiquement le format (Deriv vs MT5).
- Labellise les **crashs** : `is_crash = (price_change <= -CRASH_THRESHOLD)` (seuil = 1.0 par défaut).
- Calcule indicateurs : RSI, EMA9, EMA21, vélocité, deltas.
- Normalise (StandardScaler).
- Entraîne un MLP profond pour 10 epochs avec `BCEWithLogitsLoss` + `pos_weight` (équilibrage classes).
- Évalue : Accuracy, Precision, Recall, F1.
- Exporte `ai_model/models_saved/crash_predictor.pt` et `scaler.pkl`.

---

## 🚀 Phase 3 : Mode Live et Tour de Contrôle

```bash
start.bat
```

Deux fenêtres s'ouvrent :
- **Backend FastAPI** (port 8000) : charge le modèle, écoute `/predict`.
- **Frontend Next.js** (port 3000) : flux Deriv temps réel + jauge IA.

→ ouvrir **http://localhost:3000**.

Le graphique s'anime, et toutes les ~1 s la fenêtre des 50 derniers ticks est envoyée au backend, qui renvoie la probabilité de crash.

---

## 💰 Phase 4 : Comment "trader" avec l'alerte

1. Ouvrez votre **plateforme Deriv** (toujours en compte démo dans un premier temps !) sur le marché `Crash 500`.
2. Surveillez la **Tour de Contrôle** :
   - **35 % vert** : marché serein, courbe qui grimpote lentement → ne touchez à rien.
   - **60 %+ jaune** : la vélocité haussière s'épuise, l'IA flaire l'épuisement → doigt sur le bouton **SELL**.
   - **87 % rouge clignotant** : ALERTE MAXIMALE.
3. Cliquez **SELL** sur Deriv instantanément.
4. La chute attendue (gros tick rouge / spike baissier) tombe en général dans les **2 à 5 ticks** suivants.
5. Une fois la grosse bougie rouge passée, fermez la position (Take Profit).
6. Sur la Tour de Contrôle la probabilité retombe à ~15 %, l'alerte est archivée en bas à droite.

---

## 🛠️ Aller plus loin

- **Changer le seuil de crash** : `CRASH_THRESHOLD` dans `ai_model/train.py`.
- **Changer la fenêtre temporelle** : `WINDOW_SIZE` dans `train.py` ET `ai_model/inference.py` (doivent rester identiques).
- **Changer l'horizon de prédiction** : `PREDICT_AHEAD` (5 ticks par défaut).
- **Trader sur un autre symbole Deriv** : adaptez `symbol=` dans `data_pipeline/deriv_client.py`, la constante `SYMBOL` dans `frontend/public/derivWorker.js` et `frontend/src/components/Dashboard.jsx`.
- **Ajuster la sensibilité de l'alerte** : `Dashboard.jsx` → `isHighRisk = probability >= 80` (passez à 85 si trop de faux positifs).

---

## 🔍 Debug rapide

| Symptôme | Cause probable | Solution |
|---|---|---|
| `ia_brain: fallback_random` | Modèle PT introuvable | Lancer `python ai_model/train.py` |
| `Deriv API : déconnecté` | App ID Deriv invalide / réseau coupé | Vérifier `.env` (`DERIV_APP_ID`) |
| Probabilité bloquée à 0 % | Moins de 2 ticks reçus | Attendre quelques secondes |
| Aucune alerte > 80 % | Modèle pas assez entraîné / seuil trop strict | Plus d'epochs / augmenter le dataset |

---

## 🚨 Rappel légal

L'utilisation de cette architecture pour trader avec des fonds réels reste à vos risques. **Compte démo obligatoire** avant tout test live.
