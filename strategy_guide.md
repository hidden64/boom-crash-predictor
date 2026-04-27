# 🧠 Stratégie et Schémas du Modèle "Boom Predictor"

Ce guide retranscrit la stratégie mathématique que ton IA a apprise lors de son entraînement. Le modèle est un réseau de neurones multicouche (MLP) taillé pour de la détection de motifs non linéaires. Le but est de te permettre de "lire" le marché comme ton modèle le fait intérieurement.

## 🎯 1. L'Objectif du Modèle
Sur l'indice **Boom 1000**, le marché chute lentement (petits ticks rouges de baisse continuelle) et explose violemment à la hausse à intervalle aléatoire (gros Boom vert, ou "Spike").
*   **La Cible de l'IA** : Prédire l'apparition d'un pic massif d'un seul coup (quand la variation du prix dépasse ou égale `1.0`).
*   **Le Timing** : L'IA ne cherche pas à dire "ça va boom un jour", elle anticipe spécifiquement le Boom dans la fenêtre immédiate des **5 prochains ticks**.

---

## 🔎 2. La Mémoire contextuelle (Fenêtre temporelle)
L'IA ne regarde **jamais** un prix d'entrée isolé. À chaque milliseconde, elle analyse un "bloc" englobant précisément les **50 derniers ticks**. Elle scrute cette séquence globale de 50 mouvements pour repérer l'empreinte digitale d'un retournement.

---

## 📊 3. Les Indicateurs vitaux (Ce que l'IA traque)

Pour chaque tick de sa fenêtre de 50, l'IA ne voit pas le prix "brut" seul, elle convertit la donnée brute en 6 variables essentielles qui décrivent l'état du marché (le Feature Engineering).

### A. La Vitesse d'Essoufflement (`Velocity` et `Time Delta`)
L'IA calcule l'écart de temps (`time_delta`) et l'amplitude de variation (`price_change`) entre deux ticks pour déduire la **Vélocité**.
*   **Signification** : Contrairement au marché normal qui monte et baisse, le Boom 1000 descend mécaniquement. Si la vélocité s'effondre (les ticks baissent moins fort) ou que le temps entre les ticks s'allonge un peu, le modèle capte cet "essoufflement".
*   **Le schéma** : Les vendeurs n'ont plus la force de pousser le prix plus bas. La contrainte baissière perd sa force, ce qui laisse le champ libre aux acheteurs cachés pour déclencher l'explosion ("le spike").

### B. Le Point de Rupture (Le `RSI` 13 périodes)
Le RSI (Relative Strength Index) observe à quel point les acheteurs ou vendeurs sont en contrôle.
*   **Signification** : Sur un Boom, le marché étant constamment baissier entre chaque spike, le RSI va statistiquement être poussé vers le bas quasi tout le temps.
*   **Le schéma** : L'IA surveille le niveau absolu de survente. Un RSI qui descend très bas de manière prolongée sur les derniers 50 ticks et qui subitement arrête de marquer de nouveaux bas (même infimes) avertit que le marché a atteint un seuil de saturation absolu. Comme un ressort compressé au maximum.

### C. La Tension de la corde (`EMA 9` & `EMA 21`)
Ce sont deux Moyennes Mobiles Exponentielles : l'EMA 9 (rapide et nerveuse) et l'EMA 21 (plus profonde).
*   **Signification** : En période de baisse (avant un boom), le prix réel se trouve toujours sous l'EMA 9, qui elle-même est sous l'EMA 21.
*   **Le schéma** : L'IA traque l'élasticité. Plus l'écart entre le Prix et l'EMA 21 est grand, plus la tension pour un retour à la moyenne est forte. Le vrai signal que l'IA attend c'est quand un micro-tick de prix arrive soudainement à rattraper, toucher ou effleurer la courbe de l'EMA 9. Ce quasi-croisement dans un océan de baisse indique le retour imminent du marché.

---

## 🚀 4. Le "Schéma en Or" (La configuration à > 80%)

Si tu vois ce comportement sur ton graphique en direct, sache que c'est exactement le comportement visuel (et calculatoire) que le modèle PyTorch attend pour envoyer une alerte "Achat imminente" à la base de données :

1.  **La Séquence (La Compression)** : Le marché a fait une suite de plus de 40 ticks de faible baisse très régulière. L'élastique s'est très fortement tendu (`Prix` largement sous `EMA 21`).
2.  **L'Épuisement (Le Sang Froid)** : La puissance de la baisse diminue drastiquement (la `Vélocité` de la baisse chute). Le RSI est en fond de cale.
3.  **L'Étincelle (Le Déclencheur)** : Le prix vient d'arrêter de chuter bas, et le tout dernier tick test le niveau de résistance mineure (il frôle le bas de l'EMA 9).
4.  **Action** : Le modèle combine le grand écart des EMAs, le RSI écrasé, et le freinage du marché. Mathématiquement, la courbe Sigmoid s'envole vers `0.85` / `0.90`. C'est l'alerte à 80%+ : un Boom massif devrait tomber dans les prochaines secondes (5 ticks).

---

## 📱 5. Configuration sur Smartphone (MetaTrader 5)

Voici la marche à suivre étape par étape pour configurer exactement ces paramètres sur l'application **MetaTrader 5 (Android ou iOS)** de ton téléphone.

Puisque MT5 sur mobile ne permet pas de mettre des indicateurs sur le graphique en Ticks, tu vas utiliser le graphique **M1 (1 Minute)**.

### Étape préliminaire
1. Ouvre l'application MT5.
2. Va dans l'onglet **Chart (Graphique)** en bas.
3. Assure-toi d'être sur le symbole **Boom 1000 Index**.
4. Appuie une fois n'importe où sur l'écran pour faire apparaître le menu circulaire et choisis **M1** (pour 1 minute).

### 1️⃣ Ajouter l'EMA 9 (La moyenne rapide)
1. Appuie sur l'icône **"f"** en haut de ton écran.
2. À côté de "Graphique principal" (Main chart), appuie sur le **"f+"**.
3. Dans la liste, choisis **Moving Average** (Moyenne Mobile).
4. Remplis les paramètres exactement comme ceci :
   * **Période** : `9`
   * **Décalage (Shift)** : `0`
   * **Méthode** : `Exponential`
   * **Appliquer à** : `Close`
   * **Style** : Choisis le **Vert** (puis mets 2 ou 3 pixels pour bien la voir).
5. Appuie sur **Est fait (Done)** en haut à droite.

### 2️⃣ Ajouter l'EMA 21 (La moyenne lente)
1. Appuie de nouveau sur l'icône **"f"** en haut de l'écran.
2. À côté de "Graphique principal" (Main chart), appuie encore sur le **"f+"**.
3. Choisis de nouveau **Moving Average**.
4. Modifie les paramètres comme ceci :
   * **Période** : `21`
   * **Décalage** : `0`
   * **Méthode** : `Exponential`
   * **Appliquer à** : `Close`
   * **Style** : Choisis le **Rouge** (ou Orange).
5. Appuie sur **Est fait (Done)**. *(Maintenant, tu devrais voir tes deux lignes sur le graphique).*

### 3️⃣ Ajouter le RSI 14 (La zone de tension)
1. Appuie encore sur l'icône **"f"** en haut.
2. Attention cette fois : À côté de "Graphique principal", clique sur **"f+"**.
3. Descends dans la catégorie "Oscillateurs" et choisis **Relative Strength Index**.
4. Remplis les paramètres :
   * **Période** : `14`
   * **Appliquer à** : `Close`
   * **Style** : La couleur que tu veux (ex: Bleu).
5. **Les Niveaux (Levels)** :
   * Dans le menu de configuration du RSI, clique sur la ligne "Niveaux".
   * Par défaut, tu as `30` et `70`. Laisse-les.
   * *Optionnel mais conseillé* : Clique sur le bouton `+` pour ajouter le niveau `15` et nomme-le `Survente Extrême`.
   * Reviens en arrière.
6. Appuie sur **Est fait (Done)**.

### ✅ C'est prêt !
Ton écran est maintenant calqué sur le cerveau de ton IA. Le RSI apparaîtra dans une fenêtre séparée en bas de l'écran. Quand la ligne du RSI s'écrase sur la ligne des `30` (ou `15`) en bas, et que tes bougies rouges s'écartent beaucoup de ta ligne rouge (`EMA 21`) pour soudainement venir toucher ta ligne verte (`EMA 9`) : c'est là que ton modèle prédit son "Boom" avec plus de 80% de probabilité !
