import random
import numpy as np
import torch
import torch.nn as nn
import copy

# ==============================================================================
# ⚙️ CONFIGURATION GLOBALE DE LA NEURO-ÉVOLUTION
# ==============================================================================
EVO_CONFIG = {
    "population_size": 100,         # Population de 100 individus
    "input_size": 4,                # 4 entrées : Somme, Carte Croupier, As, True Count
    "output_size": 2,               # 2 actions : Hit (0) ou Stand (1)
    "hidden_layers": [64, 64],      # Architecture des couches cachées (modifiable à volonté)
    
    # Paramètres génétiques (Focus sur la mutation pure et l'élitisme fort)
    "mutation_rate": 0.2,          # 25% de chance qu'un poids subisse une mutation
    "mutation_strength": 0.35,       # Ampleur douce des mutations pour affiner
    "crossover_rate": 0.0,          # 0% de croisement (supprimé pour éviter de casser les MLP)
    "elite_fraction": 0.20          # Top 20% des meilleurs champions conservés intacts
}


# ==============================================================================
# 1. ARCHITECTURE DU RÉSEAU DE NEURONES (MLP DYNAMIQUE)
# ==============================================================================
class NeuralNetwork(nn.Module):
    """Réseau de neurones dont la structure s'adapte automatiquement aux couches définies."""
    def __init__(self, input_size, output_size, hidden_layers):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ==============================================================================
# 2. L'INDIVIDU (UN CERVEAU DE LA POPULATION)
# ==============================================================================
class Individual:
    """Représente un agent unique avec son réseau et son score (fitness)."""
    def __init__(self, input_size, output_size, hidden_layers):
        self.brain = NeuralNetwork(input_size, output_size, hidden_layers)
        self.fitness = 0.0

    def predict(self, inputs):
        """Prend des entrées et retourne l'indice de l'action choisie."""
        self.brain.eval()
        with torch.no_grad():
            x = torch.FloatTensor(inputs).unsqueeze(0)
            q_values = self.brain(x)
            return torch.argmax(q_values).item()


# ==============================================================================
# 3. LA POPULATION ET LE MOTEUR ÉVOLUTIF
# ==============================================================================
class GeneticPopulation:
    """Gère toute la colonie : évaluation, sélection, élitisme et mutations."""
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
        """Sélection par tournoi strict (k=5 pour plus d'exigence)."""
        candidats = random.sample(self.population, k)
        candidats.sort(key=lambda ind: ind.fitness, reverse=True)
        return candidats[0]

    def _crossover(self, parent1, parent2):
        """Inutilisé si crossover_rate = 0.0, conservé par structure."""
        enfant = Individual(self.input_size, self.output_size, self.hidden_layers)
        state_dict_p1 = parent1.brain.state_dict()
        state_dict_p2 = parent2.brain.state_dict()
        state_dict_enfant = enfant.brain.state_dict()

        for key in state_dict_p1:
            if random.random() < self.config["crossover_rate"]:
                mask = torch.rand_like(state_dict_p1[key]) < 0.5
                state_dict_enfant[key] = torch.where(mask, state_dict_p1[key], state_dict_p2[key])
            else:
                state_dict_enfant[key] = state_dict_p1[key].clone()

        enfant.brain.load_state_dict(state_dict_enfant)
        return enfant

    def _mutate(self, individual):
        """Applique un bruit gaussien aléatoire sur les poids du réseau."""
        with torch.no_grad():
            for param in individual.brain.parameters():
                if random.random() < self.config["mutation_rate"]:
                    bruit = torch.randn_like(param) * self.config["mutation_strength"]
                    param.add_(bruit)

    def evolve(self):
        """Passe à la génération suivante (Moyennes, Élitisme, Sélection, Mutations)."""
        # 1. Calcul de la moyenne générale
        somme_fitness = sum(ind.fitness for ind in self.population)
        self.last_gen_avg_fitness = somme_fitness / self.pop_size

        # 2. Tri du plus fort au plus faible
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        # 3. Calcul de la moyenne des élites
        nb_elites = int(self.pop_size * self.config["elite_fraction"])
        if nb_elites > 0:
            somme_elites = sum(ind.fitness for ind in self.population[:nb_elites])
            self.last_elite_avg_fitness = somme_elites / nb_elites
        else:
            self.last_elite_avg_fitness = 0.0

        # 4. Sauvegarde du record absolu
        if self.population[0].fitness > self.best_ever_fitness:
            self.best_ever_fitness = self.population[0].fitness
            self.best_ever_brain = copy.deepcopy(self.population[0].brain)

        print(f"--- Génération {self.generation} terminée ---")
        print(f"Moyenne Générale : {self.last_gen_avg_fitness:.2f} | Moyenne Élites : {self.last_elite_avg_fitness:.2f} | Record : {self.best_ever_fitness:.2f}")

        nouvelle_generation = []

        # 5. Élitisme pur : Copie exacte des meilleurs sans modification
        for i in range(nb_elites):
            elite_clone = Individual(self.input_size, self.output_size, self.hidden_layers)
            elite_clone.brain.load_state_dict(self.population[i].brain.state_dict())
            elite_clone.fitness = 0.0
            nouvelle_generation.append(elite_clone)

        # 6. Remplissage par Sélection & Mutation (Sans croisement)
        while len(nouvelle_generation) < self.pop_size:
            parent = self._selection_tournoi(k=5)
            
            # On duplique le parent et on le fait muter (pas de second parent nécessaire)
            enfant = Individual(self.input_size, self.output_size, self.hidden_layers)
            enfant.brain.load_state_dict(parent.brain.state_dict())
            
            self._mutate(enfant)
            enfant.fitness = 0.0
            nouvelle_generation.append(enfant)

        self.population = nouvelle_generation
        self.generation += 1