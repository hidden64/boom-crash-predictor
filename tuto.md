# Guide du Trader Algorithmique - Boom-Crash Predictor 🧠📊

Ce document est votre bible pour maîtriser la machine que nous avons construite de bout en bout. Votre objectif initial était de créer un **système de trading automatisé piloté par l'IA** capable de prédire les célèbres explosions (Spikes) des indices Boom/Crash de Deriv. 

Voici exactement comment utiliser chaque brique de ce projet, de la récolte des données brutes jusqu'au clic sur le bouton "Acheter".

---

## 🏗️ Rappel du Concept (Le Plan d'Action)

Pour qu'une IA devine le futur, elle doit d'abord apprendre le passé. L'application est donc divisée en 3 grandes phases, que vous devez suivre dans cet ordre si vous partez de zéro :

1. **La Collecte (Data Pipeline) :** On aspire des dizaines de milliers de ticks du marché pour constituer une base d'entraînement.
2. **L'Entraînement (AI Model) :** On enferme l'IA dans une salle pour qu'elle étudie la "vélocité" avant un Spike, grâce à un réseau de neurones (LSTM).
3. **Le Direct (Backend & Tour de Contrôle) :** On branche l'IA entraînée sur le flux en temps réel et on observe via une interface premium Next.js.

---

## ⚙️ Phase 1 : Collecter des données historiques

Avant d'avoir une IA intelligente, c'est une coquille vide. Il lui faut la "Data".

1. **Ouvrez un terminal** à la racine de votre dossier `boom-crash-predictor`.
2. Lancez le script de collecte :
   ```bash
   python data_pipeline/deriv_client.py
   ```
3. **Que se passe-t-il ?** Ce script ouvre une connexion WebSocket avec les serveurs de Deriv. Il va aspirer en rafale 100 000 ticks de l'indice `BOOM1000EZ` en reculant dans le temps. Il va calculer lui-même la vélocité et enregistrer tout cela dans un gros fichier `.csv` situé dans le dossier `ai_model/dataset/`.
4. *Astuce : Si le marché a un comportement très différent le mois suivant, relancez ce script pour capter les nouvelles tendances de Deriv, afin de ré-entraîner votre modèle.*

---

## 🧠 Phase 2 : Forger et Entraîner le Cerveau (Deep Learning)

Une fois le fichier `.csv` créé rempli de milliers de lignes de prix, on passe à PyTorch.

1. Laissez le terminal ouvert ou ouvrez-en un nouveau, puis tapez :
   ```bash
   python ai_model/train.py
   ```
2. **Que se passe-t-il ?** Ce script (le scientifique de données) va :
   - Lire le `.csv` généré à l'Étape 1.
   - Normaliser les données (les mettre à la même échelle).
   - "Entraîner" le modèle LSTM. L'écran va afficher des "Epochs" (cycles d'apprentissage). L'erreur (Loss) diminuera progressivement.
   - Sauvegarder le cerveau final sous forme d'un objet PyTorch : `lstm_spike_predictor.pt` (ainsi que la configuration `scaler.pkl`) dans le dossier `models_saved`.
3. **C'est fini !** Votre IA a maintenant son diplôme de marché. La phase d'entraînement n'est à refaire que si le marché change énormément sur le long terme.

---

## 🚀 Phase 3 : Mode Live et "Tour de Contrôle"

C'est ici que la magie opère et que l'interface graphique intervient. 
Votre modèle est entraîné, on veut maintenant lui donner le pouls du marché à la seconde près.

1. **Le Lancement Ultime :**
   Double-cliquez simplement sur le fichier **`start.bat`** (ou tapez `.\start.bat` dans un terminal standard).
2. **Que se passe-t-il ?** Deux fenêtres noires vont s'ouvrir.
   - **La 1ère :** Allume le "Backend FastAPI". Elle charge instantanément en mémoire votre modèle IA (le `.pt`) et écoute sur un port réseau sans broncher.
   - **La 2nde :** Allume la Tour de Contrôle "Next.js" (le site web).
3. Ouvrez votre navigateur sur **`http://localhost:3000`**. L'interface sombre apparaît.
4. Immédiatement, le graphique s'anime. L'interface vient de se brancher "en direct" chez Deriv. À chaque fois que la courbe bouge, elle envoie sournoisement les 50 derniers points au Backend, qui renvoie la probabilité calculée.

---

## 💰 Phase 4 : Comment "Trader" avec l'Alerte

Maintenant que vous avez les yeux rivés sur le moniteur, voici comment le projet a été pensé pour vous aider en trading manuel assisté :

1. Ouvrez votre **Plateforme de Trading Deriv** sur votre PC ou téléphone (en mode Démo, toujours, dans un premier temps !), sur le marché `Boom 1000`.
2. Regardez la **Tour de Contrôle**.
   - Le score est à **35%**, vert. C'est le plat pays, la courbe s'effrite lentement. Ne touchez à rien.
   - Le score monte lentement... **60%**, jaune. Le marché "ralentit" étonnamment, la vélocité change. L'IA le voit. Mettez le doigt sur le bouton "BUY / ACHETER" de Deriv.
   - D'un coup, le pourcentage clignote rouge : **87% !** ALERTE MAXIMALE.
3. Cliquez sur **BUY (Acheter)** sur Deriv instantanément.
4. L'explosion haussière (Spike) attendue va souvent se déclencher dans les 2 à 5 "Ticks" qui suivent. Une fois l'énorme bougie verte apparue sur votre graphique Deriv, cliquez immédiatement sur **Fermer la sélection (Take Profit)** pour encaisser.
5. Sur la Tour de Contrôle, l'effet de pression est libéré : la probabilité retombe à 15% instantanément. L'alerte >80% est stockée dans la liste Historique en bas à droite pour vos archives.

---

## 🛠️ Aller Plus Loin (Bricolages Avancés)

Puisque ce système est le vôtre, vous pouvez tout dompter :
* **Déclenchement trop strict/trop laxiste ?**
Si vous trouvez que l'IA sonne l'alerte à tort (Faux Positif) trop souvent, ouvrez `frontend/src/components/Dashboard.jsx` vers la ligne 126, et montez l'exigence : 
   `const isHighRisk = probability >= 85;` (au lieu de 80).
* **Vous voulez trader le CRASH au lieu du BOOM ?**
C'est le processus inverse :
  1. Modifiez `deriv_client.py` ligne 139 avec `symbol="CRASH1000"`. Récupérez les données (Phase 1).
  2. Réentraînez le modèle (Phase 2).
  3. Modifiez `Dashboard.jsx` (Front-end) dans la fonction `useEffect` du WebSocket pour s'abonner à `"ticks": "CRASH1000"`.
  4. L'indicateur d'Alerte de la Tour de Contrôle (80%) signalera alors qu'il faut cliquer sur **SELL (Vendre)** et non plus Buy sur Deriv !

**Disclaimer :** Ce logiciel (votre création) est un outil d'aide à la décision par Machine Learning, et non la garantie d'une richesse absolue. Comprendre le marché Deriv reste indispensable !
