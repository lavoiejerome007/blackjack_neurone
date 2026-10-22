import random
import numpy as np
import torch
import torch.nn as nn
import copy

# ==============================================================================
# ⚙️ CONFIGURATION GLOBALE (SANS COMPTAGE DE CARTES)
# ==============================================================================
EVO_CONFIG = {
    "population_size": 100,         # Population de 100 individus
    "input_size": 4,                # 4 entrées : Somme, Carte Croupier, As, Ratio Bankroll
    "output_size": 4,               # 4 actions : 0: Stand, 1: Hit, 2: Double, 3: Split
    "hidden_layers": [64, 64],      # Architecture MLP
    
    # Paramètres génétiques (Mutation pure et élitisme fort)
    "mutation_rate": 0.25,          # 25% de chance de mutation par poids
    "mutation_strength": 0.2,       # Bruit gaussien doux
    "crossover_rate": 0.0,          # 0% de croisement pour préserver les réseaux
    "elite_fraction": 0.20          # Top 20% des meilleurs champions conservés intacts
}


# ==============================================================================
# 1. ARCHITECTURE DU RÉSEAU (4 ENTRÉES -> 4 ACTIONS + MISE)
# ==============================================================================
class BettingNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(BettingNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Tête 1 : Actions de jeu (Stand, Hit, Double, Split)
        self.action_head = nn.Linear(prev_size, output_size)
        
        # Tête 2 : Pourcentage de mise (Entre 0.0 et 1.0)
        self.bet_head = nn.Sequential(
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        q_values = self.action_head(features)
        bet_pct = self.bet_head(features)
        return q_values, bet_pct


# ==============================================================================
# 2. L'INDIVIDU
# ==============================================================================
class Individual:
    def __init__(self, input_size, output_size, hidden_layers):
        self.brain = BettingNeuralNetwork(input_size, output_size, hidden_layers)
        self.fitness = 0.0

    def predict_action(self, inputs):
        self.brain.eval()
        with torch.no_grad():
            x = torch.FloatTensor(inputs).unsqueeze(0)
            q_values, _ = self.brain(x)
            return torch.argmax(q_values).item()

    def predict_bet(self, inputs, min_bet, max_bet):
        self.brain.eval()
        with torch.no_grad():
            x = torch.FloatTensor(inputs).unsqueeze(0)
            _, bet_pct = self.brain(x)
            pct = bet_pct.item()
            montant = min_bet + pct * (max_bet - min_bet)
            return round(montant, 2)


# ==============================================================================
# 3. MOTEUR ÉVOLUTIF
# ==============================================================================
class GeneticPopulation:
    def __init__(self, config=EVO_CONFIG):
        self.config = config
        self.pop_size = config["population_size"]
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.hidden_layers = config["hidden_layers"]
        
        self.population = [
            Individual(self.input_size, self.output_size, self.hidden_layers) 
            for _ in range(self.pop_size)
        ]
        self.generation = 1
        self.best_ever_fitness = -float('inf')
        self.best_ever_brain = None
        self.last_gen_avg_fitness = 0.0
        self.last_elite_avg_fitness = 0.0

    def get_population(self):
        return self.population

    def _selection_tournoi(self, k=5):
        candidats = random.sample(self.population, k)
        candidats.sort(key=lambda ind: ind.fitness, reverse=True)
        return candidats[0]

    def _mutate(self, individual):
        with torch.no_grad():
            for param in individual.brain.parameters():
                if random.random() < self.config["mutation_rate"]:
                    bruit = torch.randn_like(param) * self.config["mutation_strength"]
                    param.add_(bruit)

    def evolve(self):
        somme_fitness = sum(ind.fitness for ind in self.population)
        self.last_gen_avg_fitness = somme_fitness / self.pop_size

        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        nb_elites = int(self.pop_size * self.config["elite_fraction"])
        if nb_elites > 0:
            somme_elites = sum(ind.fitness for ind in self.population[:nb_elites])
            self.last_elite_avg_fitness = somme_elites / nb_elites
        else:
            self.last_elite_avg_fitness = 0.0

        if self.population[0].fitness > self.best_ever_fitness:
            self.best_ever_fitness = self.population[0].fitness
            self.best_ever_brain = copy.deepcopy(self.population[0].brain)

        print(f"--- Génération {self.generation} terminée ---")
        print(f"Moyenne Générale : {self.last_gen_avg_fitness:.2f} | Moyenne Élites : {self.last_elite_avg_fitness:.2f} | Record : {self.best_ever_fitness:.2f}")

        nouvelle_generation = []

        # Élitisme : Copie exacte
        for i in range(nb_elites):
            elite_clone = Individual(self.input_size, self.output_size, self.hidden_layers)
            elite_clone.brain.load_state_dict(self.population[i].brain.state_dict())
            elite_clone.fitness = 0.0
            nouvelle_generation.append(elite_clone)

        # Remplissage par Sélection & Mutation
        while len(nouvelle_generation) < self.pop_size:
            parent = self._selection_tournoi(k=5)
            
            enfant = Individual(self.input_size, self.output_size, self.hidden_layers)
            enfant.brain.load_state_dict(parent.brain.state_dict())
            
            self._mutate(enfant)
            enfant.fitness = 0.0
            nouvelle_generation.append(enfant)

        self.population = nouvelle_generation
        self.generation += 1