import streamlit as st
import random
import numpy as np
import torch

from neuro_evolution import GeneticPopulation, EVO_CONFIG, Individual

# ==============================================================================
# 1. ENVIRONNEMENT DU JEU DE BLACKJACK (4 PAIQUETS & COMPTAGE HI-LO)
# ==============================================================================
class BlackjackEnv:
    def __init__(self, num_decks=4):
        self.num_decks = num_decks
        self.shoe = []
        self.running_count = 0
        self._creer_shoe()
        self.reset()

    def _creer_shoe(self):
        """Crée un sabot de N paquets avec les poids Hi-Lo."""
        deck_unitaire = (
            [(2, 1)] * 4 + [(3, 1)] * 4 + [(4, 1)] * 4 + [(5, 1)] * 4 + [(6, 1)] * 4 +
            [(7, 0)] * 4 + [(8, 0)] * 4 + [(9, 0)] * 4 +
            [(10, -1)] * 16 + [(11, -1)] * 4
        )
        self.shoe = deck_unitaire * self.num_decks
        random.shuffle(self.shoe)
        self.running_count = 0

    def _verifier_et_recharger_shoe(self):
        """Reremplit le sabot s'il reste 25% (un quart) ou moins des cartes."""
        cartes_totales_initiales = 52 * self.num_decks
        if len(self.shoe) <= cartes_totales_initiales * 0.25:
            self._creer_shoe()

    def _tirer_carte(self):
        self._verifier_et_recharger_shoe()
        valeur, hi_lo = self.shoe.pop()
        self.running_count += hi_lo
        return valeur

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
        
        # True Count = Running Count divisé par le nombre de paquets restants
        paquets_restants = max(0.25, len(self.shoe) / 52.0)
        true_count = self.running_count / paquets_restants
        
        # 4 entrées envoyées au réseau de neurones
        return np.array([somme_j, carte_croupier, as_j, true_count], dtype=np.float32)

    def step(self, action):
        if action == 1:  # HIT (Tirer)
            self.main_joueur.append(self._tirer_carte())
            somme_j, _ = self._calculer_main(self.main_joueur)
            if somme_j > 21:
                return self._obtenir_etat(), -1.0, True
            else:
                return self._obtenir_etat(), 0.0, False
        else:  # STAND (Rester)
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
# 2. CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Blackjack IA — Card Counting & Elite Tracking", layout="wide")

st.title("🧬 Blackjack IA — Neuro-Évolution & Comptage de Cartes")
st.markdown("""
Tableau de bord interactif avec sabot multi-paquets, suivi de la moyenne des élites et mutation pure.
""")

if "pop_manager" not in st.session_state:
    EVO_CONFIG["input_size"] = 4
    EVO_CONFIG["output_size"] = 2
    st.session_state.pop_manager = GeneticPopulation(EVO_CONFIG)

if "env" not in st.session_state:
    st.session_state.env = BlackjackEnv(num_decks=4)


# --- BARRE LATÉRALE : CONTRÔLES ---
st.sidebar.header("⚙️ Paramètres d'Évolution")
nb_generations = st.sidebar.slider("Nombre de générations à lancer", 1, 150, 150)
parties_par_agent = st.sidebar.slider("Parties de test par individu (Fitness)", 20, 200, 150, step=10)

if st.sidebar.button("🚀 Lancer l'Évolution Génétique"):
    pop_manager = st.session_state.pop_manager
    env = st.session_state.env
    
    barre_progression = st.progress(0)
    status_text = st.empty()
    
    for gen in range(nb_generations):
        agents = pop_manager.get_population()
        
        for agent in agents:
            score_total = 0
            for _ in range(parties_par_agent):
                etat = env.reset()
                termine = False
                while not termine:
                    action = agent.predict(etat)
                    etat, recompense, termine = env.step(action)
                    score_total += recompense
            
            agent.fitness = score_total / parties_par_agent

        pop_manager.evolve()
        barre_progression.progress((gen + 1) / nb_generations)
        
        status_text.text(
            f"Génération {pop_manager.generation - 1} | "
            f"Moy. Générale : {pop_manager.last_gen_avg_fitness:.2f} | "
            f"Moy. Élites : {pop_manager.last_elite_avg_fitness:.2f} | "
            f"Record : {pop_manager.best_ever_fitness:.2f}"
        )

    st.success("✨ Évolution terminée avec succès !")


# --- SECTION DIAGNOSTIC (4 COLONNES) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Génération Actuelle", st.session_state.pop_manager.generation)
with col2:
    avg_gen = getattr(st.session_state.pop_manager, 'last_gen_avg_fitness', 0.0)
    st.metric("Moyenne Générale", f"{avg_gen:.2f}")
with col3:
    avg_elite = getattr(st.session_state.pop_manager, 'last_elite_avg_fitness', 0.0)
    st.metric("Moyenne des Élites", f"{avg_elite:.2f}")
with col4:
    st.metric("Record Absolu", f"{st.session_state.pop_manager.best_ever_fitness:.2f}")

st.divider()


# --- SECTION 1 : TEST D'UNE PARTIE UNIQUE ---
st.header("🎮 Test d'une Partie Unique (avec True Count)")

if st.button("🎲 Regarder jouer le champion sur une partie"):
    pop_manager = st.session_state.pop_manager
    env = st.session_state.env
    
    if pop_manager.best_ever_brain is None:
        st.warning("⚠️ Veuillez d'abord lancer l'évolution dans la barre latérale !")
    else:
        champion = Individual(EVO_CONFIG["input_size"], EVO_CONFIG["output_size"], EVO_CONFIG["hidden_layers"])
        champion.brain.load_state_dict(pop_manager.best_ever_brain.state_dict())
        
        etat = env.reset()
        termine = False
        log_partie = []
        
        log_partie.append(f"**Début de partie** — Main Joueur : `{env.main_joueur}` | Carte Croupier : `{env.main_croupier[0]}` | **True Count : {etat[3]:.2f}**")
        
        tour = 1
        while not termine:
            action = champion.predict(etat)
            action_nom = "HIT (Tirer)" if action == 1 else "STAND (Rester)"
            etat, recompense, termine = env.step(action)
            
            if action == 1:
                log_partie.append(f"**Tour {tour}** : Le champion choisit de **{action_nom}** -> Main : `{env.main_joueur}` (Total: {etat[0]}) | *True Count: {etat[3]:.2f}*")
            else:
                log_partie.append(f"**Tour {tour}** : Le champion choisit de **{action_nom}** -> Main Croupier : `{env.main_croupier}`")
            tour += 1
            
        if recompense == 1.0:
            log_partie.append("🎉 **Résultat : Victoire !**")
        elif recompense == -1.0:
            log_partie.append("❌ **Résultat : Défaite !**")
        else:
            log_partie.append("🤝 **Résultat : Égalité (Push).**")
            
        for ligne in log_partie:
            st.markdown(ligne)


st.divider()


# --- SECTION 2 : ÉVALUATION STATISTIQUE MASSIVE ---
st.header("📊 Évaluation Statistique Massive")
st.markdown("Testez le champion sur un grand nombre de parties d'affilée.")

nb_games_eval = st.slider("Nombre de parties pour l'évaluation", 100, 5000, 1000, step=100)

if st.button("🧪 Lancer l'évaluation sur les N parties"):
    pop_manager = st.session_state.pop_manager
    env = st.session_state.env
    
    if pop_manager.best_ever_brain is None:
        st.warning("⚠️ Veuillez d'abord lancer l'évolution dans la barre latérale !")
    else:
        champion = Individual(EVO_CONFIG["input_size"], EVO_CONFIG["output_size"], EVO_CONFIG["hidden_layers"])
        champion.brain.load_state_dict(pop_manager.best_ever_brain.state_dict())
        
        wins = 0
        losses = 0
        pushes = 0
        
        with st.spinner(f"Simulation de {nb_games_eval} parties en cours..."):
            for _ in range(nb_games_eval):
                etat = env.reset()
                termine = False
                while not termine:
                    action = champion.predict(etat)
                    etat, recompense, termine = env.step(action)
                
                if recompense == 1.0:
                    wins += 1
                elif recompense == -1.0:
                    losses += 1
                else:
                    pushes += 1
        
        win_rate = (wins / nb_games_eval) * 100
        loss_rate = (losses / nb_games_eval) * 100
        push_rate = (pushes / nb_games_eval) * 100
        
        st.success(f"✨ Évaluation terminée sur {nb_games_eval} parties !")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Taux de Victoire", f"{win_rate:.1f}%", f"{wins} victoires")
        with col_b:
            st.metric("Taux de Défaite", f"{loss_rate:.1f}%", f"{losses} défaites")
        with col_c:
            st.metric("Taux d'Égalité (Push)", f"{push_rate:.1f}%", f"{pushes} égalités")