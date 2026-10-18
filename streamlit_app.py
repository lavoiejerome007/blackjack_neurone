import streamlit as st
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ==============================================================================
# ⚙️ CONFIGURATION GLOBALE DU PROJET (MODIFIABLE À VOLONTÉ)
# C'est ici que vous définissez votre projet pour n'importe quel cas d'usage !
# ==============================================================================
PROJECT_CONFIG = {
    "project_name": "Blackjack AI - Template Modulaire",
    
    # 1. Définition des ENTRÉES (Capteurs, états, variables du problème)
    "input_names": [
        "Score du joueur", 
        "Carte visible du croupier", 
        "As utilisable (1=Oui, 0=Non)"
    ],
    
    # 2. Définition des SORTIES (Actions possibles)
    "output_names": [
        "Stand (Rester)", 
        "Hit (Tirer)"
    ],
    
    # 3. Architecture du Réseau de Neurones (Profondeur et Largeur)
    # Chaque nombre dans cette liste crée une couche cachée avec son nombre de neurones.
    # Ex: [64, 64] = 2 couches cachées de 64 neurones.
    # Ex: [128, 64, 32] = 3 couches cachées (128, puis 64, puis 32).
    # Ex: [32] = 1 seule couche cachée de 32 neurones.
    "hidden_layers": [64, 64],
    
    # 4. Hyperparamètres d'apprentissage
    "learning_rate": 0.001,
    "gamma": 0.95,              # Facteur d'actualisation (importance du futur)
    "epsilon_start": 1.0,       # Exploration initiale (100% aléatoire au début)
    "epsilon_decay": 0.9995,    # Vitesse de diminution de l'exploration
    "epsilon_min": 0.05         # Exploration minimale résiduelle
}

# Calcul automatique des dimensions d'après la configuration ci-dessus
INPUT_SIZE = len(PROJECT_CONFIG["input_names"])
OUTPUT_SIZE = len(PROJECT_CONFIG["output_names"])
HIDDEN_LAYERS = PROJECT_CONFIG["hidden_layers"]


# ==============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================================
st.set_page_config(page_title=PROJECT_CONFIG["project_name"], layout="wide")

st.title(f"🤖 {PROJECT_CONFIG['project_name']}")
st.markdown("""
Ce tableau de bord exécute une architecture de **Réseau de Neurones Profond (DQN)** entièrement dynamique. 
Modifiez les paramètres dans le code pour l'adapter à un autre projet !
""")

# Affichage de la configuration active dans l'interface
with st.expander("🔍 Voir la configuration active du modèle"):
    st.write(f"**Nombre d'entrées ({INPUT_SIZE}) :**", PROJECT_CONFIG["input_names"])
    st.write(f"**Nombre de sorties ({OUTPUT_SIZE}) :**", PROJECT_CONFIG["output_names"])
    st.write(f"**Structure des couches cachées :** {HIDDEN_LAYERS} ({len(HIDDEN_LAYERS)} couches)")


# ==============================================================================
# 1. ENVIRONNEMENT DU JEU DE BLACKJACK
# ==============================================================================
class BlackjackEnv:
    def __init__(self):
        self.reset()

    def _tirer_carte(self):
        carte = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
        return 11 if carte == 1 else carte

    def _calculer_main(self, main):
        somme = sum(main)
        as_utilisables = main.count(11)
        while somme > 21 and as_utilisables > 0:
            somme -= 10
            as_utilisables -= 1
        contient_as_utilisable = 1.0 if (11 in main and sum(main) <= 21) else 0.0
        return somme, contient_as_utilisable

    def reset(self):
        self.main_joueur = [self._tirer_carte(), self._tirer_carte()]
        self.main_croupier = [self._tirer_carte(), self._tirer_carte()]
        return self._obtenir_etat()

    def _obtenir_etat(self):
        somme_j, as_j = self._calculer_main(self.main_joueur)
        carte_croupier = self.main_croupier[0]
        # Doit correspondre exactement à la taille et l'ordre des 'input_names'
        return np.array([somme_j, carte_croupier, as_j], dtype=np.float32)

    def step(self, action):
        if action == 1:  # HIT
            self.main_joueur.append(self._tirer_carte())
            somme_j, _ = self._calculer_main(self.main_joueur)
            if somme_j > 21:
                return self._obtenir_etat(), -1.0, True
            else:
                return self._obtenir_etat(), 0.0, False
        else:  # STAND
            somme_c, _ = self._calculer_main(self.main_croupier)
            while somme_c < 17:
                self.main_croupier.append(self._tirer_carte())
                somme_c, _ = self._calculer_main(self.main_croupier)

            somme_j, _ = self._calculer_main(self.main_joueur)

            if somme_c > 21 or somme_j > somme_c:
                reward = 1.0
            elif somme_j < somme_c:
                reward = -1.0
            else:
                reward = 0.0

            return self._obtenir_etat(), reward, True


# ==============================================================================
# 2. ARCHITECTURE DU RÉSEAU DE NEURONES DYNAMIQUE
# ==============================================================================
class DynamicDQN(nn.Module):
    """
    Réseau de neurones générique qui s'adapte automatiquement au nombre 
    d'entrées, de sorties et de couches cachées spécifiés dans la configuration.
    """
    def __init__(self, input_size, output_size, hidden_layers):
        super(DynamicDQN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Construction dynamique de chaque couche cachée définie dans le dictionnaire
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU()) # Fonction d'activation non-linéaire
            prev_size = hidden_size
            
        # Couche de sortie finale (pas de ReLU sur la sortie pour garder les Q-values brutes)
        layers.append(nn.Linear(prev_size, output_size))
        
        # Empilement séquentiel automatique de toutes les couches
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Passe avant (Forward Pass) à travers toutes les couches construites."""
        return self.network(x)


# ==============================================================================
# 3. AGENT D'APPRENTISSAGE PAR RENFORCEMENT
# ==============================================================================
class DQNAgent:
    def __init__(self):
        self.gamma = PROJECT_CONFIG["gamma"]
        self.epsilon = PROJECT_CONFIG["epsilon_start"]
        self.epsilon_decay = PROJECT_CONFIG["epsilon_decay"]
        self.epsilon_min = PROJECT_CONFIG["epsilon_min"]
        
        self.memory = deque(maxlen=10000)
        
        # Instanciation du modèle avec les variables de configuration globales
        self.model = DynamicDQN(
            input_size=INPUT_SIZE, 
            output_size=OUTPUT_SIZE, 
            hidden_layers=HIDDEN_LAYERS
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=PROJECT_CONFIG["learning_rate"])
        self.criterion = nn.MSELoss()

    def choisir_action(self, etat):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, OUTPUT_SIZE - 1)  # Action aléatoire parmi toutes les sorties

        etat_tensor = torch.FloatTensor(etat).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(etat_tensor)
        return torch.argmax(q_values).item()

    def mémoriser(self, etat, action, recompense, nouvel_etat, termine):
        self.memory.append((etat, action, recompense, nouvel_etat, termine))

    def entrainer(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for etat, action, recompense, nouvel_etat, termine in minibatch:
            etat_t = torch.FloatTensor(etat).unsqueeze(0)
            nouvel_etat_t = torch.FloatTensor(nouvel_etat).unsqueeze(0)

            q_target = self.model(etat_t).clone()

            if termine:
                q_target[0][action] = recompense
            else:
                q_futur = torch.max(self.model(nouvel_etat_t)).item()
                q_target[0][action] = recompense + self.gamma * q_futur

            self.optimizer.zero_grad()
            q_actuelle = self.model(etat_t)
            loss = self.criterion(q_actuelle, q_target)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ==============================================================================
# 4. INTERFACE STREAMLIT
# ==============================================================================
if "agent" not in st.session_state:
    st.session_state.agent = DQNAgent()
if "env" not in st.session_state:
    st.session_state.env = BlackjackEnv()

# --- BARRE LATÉRALE ---
st.sidebar.header("⚙️ Contrôles d'Entraînement")
episodes = st.sidebar.slider("Nombre de parties d'entraînement", 100, 5000, 1000, step=100)

if st.sidebar.button("🚀 Lancer l'Entraînement"):
    # Réinitialisation de l'agent avec la configuration actuelle
    st.session_state.agent = DQNAgent()
    env = st.session_state.env
    agent = st.session_state.agent

    barre_progression = st.progress(0)
    zone_texte = st.empty()
    victoires = 0

    for ep in range(1, episodes + 1):
        etat = env.reset()
        termine = False

        while not termine:
            action = agent.choisir_action(etat)
            nouvel_etat, recompense, termine = env.step(action)
            agent.mémoriser(etat, action, recompense, nouvel_etat, termine)
            agent.entrainer(batch_size=32)
            etat = nouvel_etat

            if termine and recompense == 1.0:
                victoires += 1

        if ep % (episodes // 100 if episodes >= 100 else 1) == 0:
            barre_progression.progress(ep / episodes)
            taux_victoire = (victoires / ep) * 100
            zone_texte.text(f"Épisode {ep}/{episodes} | Taux de victoire : {taux_victoire:.1f}% | Epsilon : {agent.epsilon:.2f}")

    st.success("✨ Entraînement terminé avec succès !")

# --- SECTION TEST ---
st.header("🎮 Test de l'Agent en Direct")

if st.button("🎲 Lancer une Partie Test"):
    env = st.session_state.env
    agent = st.session_state.agent
    
    etat = env.reset()
    termine = False
    log_partie = []

    log_partie.append(f"**Début** — Entrées transmises ({PROJECT_CONFIG['input_names']}) : `{etat}`")

    tour = 1
    while not termine:
        etat_tensor = torch.FloatTensor(etat).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.model(etat_tensor)
            action = torch.argmax(q_values).item()

        action_nom = PROJECT_CONFIG["output_names"][action]
        nouvel_etat, recompense, termine = env.step(action)
        
        log_partie.append(f"**Tour {tour}** : L'IA choisit l'action de sortie `[{action}] : {action_nom}`")
        etat = nouvel_etat
        tour += 1

    if recompense == 1.0:
        log_partie.append("🎉 **Résultat : Victoire !**")
    elif recompense == -1.0:
        log_partie.append("❌ **Résultat : Défaite !**")
    else:
        log_partie.append("🤝 **Résultat : Égalité (Push) !**")

    for ligne in log_partie:
        st.markdown(ligne)