import streamlit as st
import random
import numpy as np
import torch
import copy

from neuro_evolution import GeneticPopulation, EVO_CONFIG, Individual

# ==============================================================================
# 1. ENVIRONNEMENT BLACKJACK STANDARD (SANS COMPTAGE)
# ==============================================================================
class OptimizedBlackjackEnv:
    def __init__(self, num_decks=4, initial_bankroll=1000.0, min_bet=10.0, max_bet=200.0):
        self.num_decks = num_decks
        self.initial_bankroll = initial_bankroll
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.bankroll = initial_bankroll
        self.current_bet = min_bet
        
        self.shoe = []
        self._creer_shoe()
        self.reset()

    def _creer_shoe(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        deck_unitaire = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
        self.shoe = deck_unitaire * self.num_decks
        random.shuffle(self.shoe)

    def _verifier_et_recharger_shoe(self):
        if len(self.shoe) <= (52 * self.num_decks) * 0.25:
            self._creer_shoe()

    def _tirer_carte(self):
        self._verifier_et_recharger_shoe()
        return self.shoe.pop()

    def _calculer_main(self, main):
        somme = sum(main)
        as_utilisables = main.count(11)
        while somme > 21 and as_utilisables > 0:
            somme -= 10
            as_utilisables -= 1
        contient_as_utilisable = 1.0 if (11 in main and sum(main) <= 21) else 0.0
        return somme, contient_as_utilisable

    def reset_session(self, seed=None):
        self.bankroll = self.initial_bankroll
        if seed is not None:
            self._creer_shoe(seed=seed)
        return self.reset()

    def reset(self):
        self.main_joueur = [self._tirer_carte(), self._tirer_carte()]
        self.main_croupier = [self._tirer_carte(), self._tirer_carte()]
        return self._obtenir_etat()

    def _obtenir_etat(self, main_specifique=None):
        main = main_specifique if main_specifique is not None else self.main_joueur
        somme_j, as_j = self._calculer_main(main)
        carte_croupier = self.main_croupier[0]
        bankroll_ratio = self.bankroll / self.initial_bankroll
        
        return np.array([somme_j, carte_croupier, as_j, bankroll_ratio], dtype=np.float32)

    def fixer_mise(self, montant_mise):
        self.current_bet = min(max(montant_mise, self.min_bet), min(self.max_bet, self.bankroll))

    def _evaluer_main_contre_croupier(self, main_joueur, mise):
        somme_c, _ = self._calculer_main(self.main_croupier)
        while somme_c < 17:
            self.main_croupier.append(self._tirer_carte())
            somme_c, _ = self._calculer_main(self.main_croupier)

        somme_j, _ = self._calculer_main(main_joueur)
        
        if somme_j > 21:
            return -mise
        elif somme_c > 21 or somme_j > somme_c:
            return mise
        elif somme_j < somme_c:
            return -mise
        else:
            return 0.0

    def step(self, action):
        if action == 2 and (len(self.main_joueur) != 2 or self.bankroll < self.current_bet * 2):
            action = 1

        if action == 3 and (len(self.main_joueur) != 2 or self.main_joueur[0] != self.main_joueur[1] or self.bankroll < self.current_bet * 2):
            action = 1

        if action == 1:  # HIT
            self.main_joueur.append(self._tirer_carte())
            somme_j, _ = self._calculer_main(self.main_joueur)
            if somme_j > 21:
                profit = -self.current_bet
                self.bankroll += profit
                return self._obtenir_etat(), profit, True
            else:
                return self._obtenir_etat(), 0.0, False

        elif action == 0:  # STAND
            profit = self._evaluer_main_contre_croupier(self.main_joueur, self.current_bet)
            self.bankroll += profit
            return self._obtenir_etat(), profit, True

        elif action == 2:  # DOUBLE DOWN
            self.current_bet *= 2
            self.main_joueur.append(self._tirer_carte())
            profit = self._evaluer_main_contre_croupier(self.main_joueur, self.current_bet)
            self.bankroll += profit
            return self._obtenir_etat(), profit, True

        elif action == 3:  # SPLIT
            main_1 = [self.main_joueur[0], self._tirer_carte()]
            main_2 = [self.main_joueur[1], self._tirer_carte()]
            
            profit_1 = self._evaluer_main_contre_croupier(main_1, self.current_bet)
            profit_2 = self._evaluer_main_contre_croupier(main_2, self.current_bet)
            
            profit_total = profit_1 + profit_2
            self.bankroll += profit_total
            return self._obtenir_etat(), profit_total, True


# ==============================================================================
# 2. INTERFACE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Blackjack IA — Profit & Risk Optimization", layout="wide")

st.title("🧬 Blackjack IA — Optimisation du Profit & Aversion au Risque")
st.markdown("""
Entraînement génétique sans comptage de cartes. **À la fin de l'évolution**, un tournoi miroir valide automatiquement le champion pour éliminer le hasard.
""")

if "pop_manager" not in st.session_state:
    EVO_CONFIG["input_size"] = 4
    EVO_CONFIG["output_size"] = 4
    st.session_state.pop_manager = GeneticPopulation(EVO_CONFIG)

if "env" not in st.session_state:
    st.session_state.env = OptimizedBlackjackEnv()


# --- BARRE LATÉRALE ---
st.sidebar.header("⚙️ Entraînement Génétique")
nb_generations = st.sidebar.slider("Générations", 1, 100, 20)
parties_par_agent = st.sidebar.slider("Parties par session d'entraînement", 20, 200, 50, step=10)

st.sidebar.subheader("🏆 Paramètres de Validation Finale")
nb_rondes_val = st.slider("Rondes du tournoi miroir (Dernière étape)", 3, 20, 5)
parties_par_ronde_val = st.slider("Parties par ronde de validation", 50, 500, 100, step=50)

st.sidebar.subheader("💰 Mises & Sécurité")
min_bet = st.sidebar.number_input("Mise Minimale ($)", value=10.0, step=5.0)
max_bet = st.sidebar.number_input("Mise Maximale ($)", value=200.0, step=10.0)

if st.sidebar.button("🚀 Lancer l'Évolution & la Validation Finale"):
    pop_manager = st.session_state.pop_manager
    env = st.session_state.env
    env.min_bet = min_bet
    env.max_bet = max_bet
    
    barre_progression = st.progress(0)
    status_text = st.empty()
    
    # --------------------------------------------------------------------------
    # ÉTAPE 1 : ENTRAÎNEMENT GÉNÉTIQUE CLASSIQUE
    # --------------------------------------------------------------------------
    for gen in range(nb_generations):
        agents = pop_manager.get_population()
        
        for agent in agents:
            env.reset_session()
            penalites_cumulees = 0.0
            profit_brut_total = 0.0
            
            for _ in range(parties_par_agent):
                if env.bankroll < min_bet:
                    penalites_cumulees += 5000.0
                    break
                
                etat = env._obtenir_etat()
                montant_mise = agent.predict_bet(etat, min_bet, max_bet)
                env.fixer_mise(montant_mise)
                
                etat = env.reset()
                termine = False
                while not termine:
                    action = agent.predict_action(etat)
                    etat, profit, termine = env.step(action)
                    profit_brut_total += profit
                    
                    if profit < 0:
                        perte = abs(profit)
                        penalites_cumulees += (perte ** 1.35) * 0.12
            
            agent.fitness = profit_brut_total - penalites_cumulees

        pop_manager.evolve()
        barre_progression.progress((gen + 1) / (nb_generations + 1))
        
        status_text.text(
            f"Génération {pop_manager.generation - 1} | "
            f"Moy. Générale : {pop_manager.last_gen_avg_fitness:.1f} | "
            f"Moy. Élites : {pop_manager.last_elite_avg_fitness:.1f} | "
            f"Record : {pop_manager.best_ever_fitness:.1f}"
        )

    # --------------------------------------------------------------------------
    # ÉTAPE 2 : TOURNOI MIROIR DE VALIDATION FINALE (Élimination du hasard)
    # --------------------------------------------------------------------------
    status_text.text("🏆 Étape finale : Tournoi miroir de validation des élites en cours...")
    
    population_triee = sorted(pop_manager.get_population(), key=lambda ind: ind.fitness, reverse=True)
    nb_elites = max(1, int(pop_manager.pop_size * pop_manager.config["elite_fraction"]))
    elites = population_triee[:nb_elites]
    
    points_tournoi = {i: 0 for i in range(len(elites))}
    profits_cumules = {i: 0.0 for i in range(len(elites))}
    
    for r in range(nb_rondes_val):
        graine_ronde = 5000 + r * 100
        resultats_ronde = []
        
        for idx, elite in enumerate(elites):
            env.reset_session(seed=graine_ronde)
            profit_ronde = 0.0
            
            for _ in range(parties_par_ronde_val):
                if env.bankroll < env.min_bet:
                    break
                
                etat_avant = env._obtenir_etat()
                montant_mise = elite.predict_bet(etat_avant, env.min_bet, env.max_bet)
                env.fixer_mise(montant_mise)
                
                etat = env.reset()
                termine = False
                while not termine:
                    action = elite.predict_action(etat)
                    etat, profit, termine = env.step(action)
                    profit_ronde += profit

            resultats_ronde.append((idx, profit_ronde))
            profits_cumules[idx] += profit_ronde

        resultats_ronde.sort(key=lambda x: x[1], reverse=True)
        for rang, (idx_elite, _) in enumerate(resultats_ronde):
            pts_attribues = len(elites) - rang
            points_tournoi[idx_elite] += pts_attribues

    classement_final = sorted(points_tournoi.items(), key=lambda x: x[1], reverse=True)
    idx_gagnant = classement_final[0][0]
    vrai_champion = elites[idx_gagnant]
    
    # Couronnement officiel du vrai champion validé
    pop_manager.best_ever_brain = copy.deepcopy(vrai_champion.brain)
    pop_manager.best_ever_fitness = profits_cumules[idx_gagnant]
    
    barre_progression.progress(1.0)
    status_text.text("✨ Évolution et Tournoi de Validation terminés avec succès !")
    st.success(f"👑 **Vainqueur Couronné : Élite #{idx_gagnant + 1}** avec **{classement_final[0][1]} Points de Rang** !")


# --- DIAGNOSTIC (4 COLONNES) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Génération", st.session_state.pop_manager.generation)
with col2:
    avg_gen = getattr(st.session_state.pop_manager, 'last_gen_avg_fitness', 0.0)
    st.metric("Score Moy. Général", f"{avg_gen:.1f}")
with col3:
    avg_elite = getattr(st.session_state.pop_manager, 'last_elite_avg_fitness', 0.0)
    st.metric("Score Moy. Élites", f"{avg_elite:.1f}")
with col4:
    st.metric("Record Absolu (Score)", f"{st.session_state.pop_manager.best_ever_fitness:.1f}")

st.divider()


# ==============================================================================
# SECTION 1 : TEST D'UNE PARTIE UNIQUE DU CHAMPION
# ==============================================================================
st.header("🎮 Test d'une Partie Unique du Champion Couronné")

if st.button("🎲 Regarder jouer le champion"):
    pop_manager = st.session_state.pop_manager
    env = st.session_state.env
    
    if pop_manager.best_ever_brain is None:
        st.warning("⚠️ Veuillez d'abord lancer l'évolution dans la barre latérale !")
    else:
        champion = Individual(EVO_CONFIG["input_size"], EVO_CONFIG["output_size"], EVO_CONFIG["hidden_layers"])
        champion.brain.load_state_dict(pop_manager.best_ever_brain.state_dict())
        
        etat_initial = env._obtenir_etat()
        mise_choisie = champion.predict_bet(etat_initial, env.min_bet, env.max_bet)
        env.fixer_mise(mise_choisie)
        
        etat = env.reset()
        termine = False
        log_partie = []
        
        actions_dict = {0: "STAND", 1: "HIT", 2: "DOUBLE DOWN", 3: "SPLIT"}
        
        log_partie.append(f"Bankroll : `{env.bankroll:.2f}$`")
        log_partie.append(f"👉 **Mise choisie par l'IA : `{env.current_bet:.2f}$`**")
        log_partie.append(f"Main Joueur : `{env.main_joueur}` | Carte Croupier : `{env.main_croupier[0]}`")
        
        tour = 1
        while not termine:
            action = champion.predict_action(etat)
            action_nom = actions_dict.get(action, "UNKNOWN")
            etat, profit, termine = env.step(action)
            
            log_partie.append(f"**Tour {tour}** : Action = **{action_nom}** -> Main : `{env.main_joueur}` | Main Croupier : `{env.main_croupier}`")
            tour += 1
            
        if profit > 0:
            log_partie.append(f"🎉 **Résultat : Victoire ! Profit = +{profit:.2f}$**")
        elif profit < 0:
            log_partie.append(f"❌ **Résultat : Défaite ! Perte = {profit:.2f}$**")
        else:
            log_partie.append("🤝 **Résultat : Égalité (Push). Profit = 0$**")
            
        for ligne in log_partie:
            st.markdown(ligne)


st.divider()


# ==============================================================================
# SECTION 2 : SIMULATION FINANCIÈRE DU CHAMPION (POURCENTAGE & ARGENT)
# ==============================================================================
st.header("📊 Simulation Financière du Champion Validé")
st.markdown("Testez la performance à long terme du champion couronné (taux de rendement, argent gagné et évolution du capital).")

nb_games_eval = st.slider("Parties dans la session de test", 100, 2000, 500, step=100)

if st.button("🧪 Lancer la simulation financière"):
    pop_manager = st.session_state.pop_manager
    env = st.session_state.env
    
    if pop_manager.best_ever_brain is None:
        st.warning("⚠️ Lancez d'abord l'évolution dans la barre latérale !")
    else:
        champion = Individual(EVO_CONFIG["input_size"], EVO_CONFIG["output_size"], EVO_CONFIG["hidden_layers"])
        champion.brain.load_state_dict(pop_manager.best_ever_brain.state_dict())
        
        env.reset_session()
        mises = []
        bankroll_history = [env.bankroll]
        victoires = 0
        
        with st.spinner(f"Simulation de {nb_games_eval} parties en cours..."):
            for _ in range(nb_games_eval):
                if env.bankroll < env.min_bet:
                    st.error("💥 Banqueroute pendant la simulation !")
                    break
                
                etat_avant = env._obtenir_etat()
                montant_mise = champion.predict_bet(etat_avant, env.min_bet, env.max_bet)
                env.fixer_mise(montant_mise)
                mises.append(env.current_bet)
                
                etat = env.reset()
                termine = False
                while not termine:
                    action = champion.predict_action(etat)
                    etat, profit, termine = env.step(action)
                    if profit > 0:
                        victoires += 1
                
                bankroll_history.append(env.bankroll)
        
        profit_total = env.bankroll - env.initial_bankroll
        roi = (profit_total / env.initial_bankroll) * 100
        win_rate = (victoires / nb_games_eval) * 100
        mise_moyenne = np.mean(mises) if mises else 0
        
        st.success("✨ Simulation financière terminée !")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Bankroll Finale", f"{env.bankroll:.2f} $", f"{profit_total:+.2f} $")
        with col_b:
            st.metric("Rendement (ROI)", f"{roi:.1f}%")
        with col_c:
            st.metric("Taux de Victoire", f"{win_rate:.1f}%")
        with col_d:
            st.metric("Mise Moyenne", f"{mise_moyenne:.2f} $")
            
        st.line_chart(bankroll_history)