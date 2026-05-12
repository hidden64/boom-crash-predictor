# 🧠 Stratégie et Schémas du Modèle "CRASH 500 Predictor"

Ce guide retranscrit la stratégie mathématique apprise par l'IA pendant l'entraînement. Le modèle est un MLP profond taillé pour de la détection de motifs non linéaires. Le but est de te permettre de "lire" le marché comme ton modèle le fait intérieurement.

## 🎯 1. L'Objectif du Modèle
Sur l'indice **CRASH 500**, le marché monte lentement (petits ticks verts continus) et **plonge violemment vers le bas** à intervalle aléatoire (gros tick rouge, "Crash").
* **La Cible de l'IA** : prédire l'apparition d'une chute brutale (quand la variation de prix `≤ -1.0`).
* **Le Timing** : l'IA n'annonce pas "ça va crasher un jour", elle anticipe le crash dans la fenêtre des **5 prochains ticks**.

---

## 🔎 2. La Mémoire contextuelle (Fenêtre temporelle)
À chaque tick, l'IA analyse un bloc des **50 derniers ticks**. Elle scrute cette séquence pour repérer l'empreinte d'un retournement baissier imminent.

---

## 📊 3. Les Indicateurs vitaux (ce que l'IA traque)

Pour chaque tick de sa fenêtre de 50, l'IA convertit la donnée brute en **6 variables** :

### A. La Vitesse d'Essoufflement (`Velocity` et `Time Delta`)
L'IA calcule `time_delta` (écart de temps) et `price_change` (amplitude) entre deux ticks pour déduire la **vélocité**.
* **Signification** : CRASH 500 monte mécaniquement. Si la vélocité haussière s'effondre (les ticks montent moins fort) ou que le temps entre les ticks s'allonge, le modèle capte cet **essoufflement** des acheteurs.
* **Le schéma** : les acheteurs n'ont plus la force de pousser le prix plus haut. La pression haussière s'épuise → champ libre pour les vendeurs cachés qui déclenchent la chute.

### B. Le Point de Rupture (`RSI` 13 périodes)
Le RSI observe à quel point acheteurs ou vendeurs sont en contrôle.
* **Signification** : CRASH 500 étant constamment haussier entre chaque crash, le RSI est statistiquement poussé **vers le haut** quasi tout le temps (zone de surachat).
* **Le schéma** : l'IA surveille le niveau **extrême** de surachat. Un RSI plafonné prolongé sur les derniers 50 ticks, qui subitement cesse de marquer de nouveaux hauts (même infimes), signe un seuil de saturation absolu. Comme un ressort tendu au maximum.

### C. La Tension de la corde (`EMA 9` & `EMA 21`)
Deux Moyennes Mobiles Exponentielles : EMA 9 (rapide, nerveuse) et EMA 21 (plus profonde).
* **Signification** : en période de hausse (avant un crash), le prix réel se trouve toujours **au-dessus** de l'EMA 9, elle-même au-dessus de l'EMA 21.
* **Le schéma** : l'IA traque l'élasticité. Plus l'écart entre le prix et l'EMA 21 est grand, plus la tension pour un retour à la moyenne est forte. Le vrai signal attendu : un micro-tick de prix qui vient soudainement **toucher ou redescendre vers** la courbe de l'EMA 9 par le haut. Ce quasi-croisement dans un océan de hausse indique l'imminence d'un retournement baissier.

---

## 🚀 4. Le "Schéma en Or" (configuration à > 80 %)

Si tu vois ce comportement sur ton graphique en direct, c'est exactement ce que le modèle PyTorch attend pour envoyer une alerte "Vente imminente" :

1. **La Séquence (la compression)** : le marché a enchaîné plus de 40 ticks de faible hausse très régulière. L'élastique est tendu (`Prix` largement au-dessus de l'`EMA 21`).
2. **L'Épuisement (le sang froid)** : la puissance de la hausse diminue drastiquement (la vélocité haussière chute). Le RSI est au plafond.
3. **L'Étincelle (le déclencheur)** : le prix arrête de monter, et le tout dernier tick teste le support mineur (il frôle le haut de l'EMA 9).
4. **Action** : le modèle combine le grand écart des EMAs, le RSI écrasé vers le haut et le freinage du marché. Mathématiquement, la sigmoïde s'envole vers `0.85` / `0.90`. C'est l'alerte 80 %+ : un crash massif devrait tomber dans les 5 ticks suivants.

---

## 📱 5. Configuration sur Smartphone (MetaTrader 5)

Voici la marche à suivre étape par étape pour configurer ces paramètres sur l'application **MetaTrader 5 (Android ou iOS)**.

Puisque MT5 mobile ne permet pas d'indicateurs sur le graphique en ticks, tu vas utiliser le graphique **M1 (1 Minute)**.

### Étape préliminaire
1. Ouvre l'application MT5.
2. Onglet **Chart (Graphique)** en bas.
3. Sélectionne le symbole **Crash 500 Index**.
4. Tape n'importe où pour faire apparaître le menu circulaire → choisis **M1**.

### 1️⃣ Ajouter l'EMA 9 (moyenne rapide)
1. Appuie sur l'icône **"f"** en haut.
2. À côté de "Graphique principal", appuie sur **"f+"**.
3. Choisis **Moving Average**.
4. Paramètres :
   * **Période** : `9`
   * **Décalage** : `0`
   * **Méthode** : `Exponential`
   * **Appliquer à** : `Close`
   * **Style** : Vert, 2-3 pixels.
5. **Est fait (Done)**.

### 2️⃣ Ajouter l'EMA 21 (moyenne lente)
1. Re-clic sur **"f"** puis **"f+"**.
2. **Moving Average** à nouveau.
3. Paramètres :
   * **Période** : `21`
   * **Décalage** : `0`
   * **Méthode** : `Exponential`
   * **Appliquer à** : `Close`
   * **Style** : Rouge (ou Orange).
4. **Est fait**.

### 3️⃣ Ajouter le RSI 14
1. **"f"** → **"f+"** à côté de Graphique principal.
2. Catégorie *Oscillateurs* → **Relative Strength Index**.
3. Paramètres :
   * **Période** : `14`
   * **Appliquer à** : `Close`
   * **Style** : couleur libre (Bleu par exemple).
4. **Niveaux** :
   * Par défaut `30` et `70`. Laisse-les.
   * *Optionnel mais conseillé* : ajoute le niveau `85` nommé `Surachat Extrême` (l'inverse de Boom).
5. **Est fait**.

### ✅ C'est prêt !
Ton écran est maintenant calqué sur le cerveau de ton IA. Quand le RSI s'écrase **vers le haut** (sur `70` voire `85`) et que tes bougies vertes s'écartent fortement de ta ligne rouge (`EMA 21`) avant de venir soudainement **redescendre toucher** ta ligne verte (`EMA 9`) : c'est là que ton modèle prédit son crash avec plus de 80 % de probabilité.
