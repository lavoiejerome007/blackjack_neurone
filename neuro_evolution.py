import random
import numpy as np
import torch
import torch.nn as nn
import copy

# ==============================================================================
# ⚙️ CONFIGURATION GLOBALE DE LA NEURO-ÉVOLUTION
# ==============================================================================
EVO_CONFIG = {
    "population_size": 60,
    "input_size": 6,
    "output_size": 4,
    "hidden_layers": [64, 64],
    
    "mutation_rate": 0.25,
    "mutation_strength": 0.2,
    "crossover_rate": 0.0,
    "elite_fraction": 0.20
}


class PokerNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(PokerNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        self.feature_extractor = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_size, output_size)
        self.sizing_head = nn.Sequential(
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        q_values = self.action_head(features)
        sizing_pct = self.sizing_head(features)
        return q_values, sizing_pct


class PokerIndividual:
    def __init__(self, input_size, output_size, hidden_layers):
        self.brain = PokerNeuralNetwork(input_size, output_size, hidden_layers)
        self.fitness = 0.0

    def predict_action(self, inputs):
        self.brain.eval()
        with torch.no_grad():
            x = torch.FloatTensor(inputs).unsqueeze(0)
            q_values, sizing_pct = self.brain(x)
            action = torch.argmax(q_values).item()
            pct = sizing_pct.item()
            return action, pct


class PokerGeneticPopulation:
    def __init__(self, config=EVO_CONFIG):
        self.config = config
        self.pop_size = config["population_size"]
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.hidden_layers = config["hidden_layers"]
        
        self.population = [
            PokerIndividual(self.input_size, self.output_size, self.hidden_layers) 
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

        nouvelle_generation = []

        for i in range(nb_elites):
            elite_clone = PokerIndividual(self.input_size, self.output_size, self.hidden_layers)
            elite_clone.brain.load_state_dict(self.population[i].brain.state_dict())
            elite_clone.fitness = 0.0
            nouvelle_generation.append(elite_clone)

        while len(nouvelle_generation) < self.pop_size:
            parent = self._selection_tournoi(k=5)
            enfant = PokerIndividual(self.input_size, self.output_size, self.hidden_layers)
            enfant.brain.load_state_dict(parent.brain.state_dict())
            
            self._mutate(enfant)
            enfant.fitness = 0.0
            nouvelle_generation.append(enfant)

        self.population = nouvelle_generation
        self.generation += 1