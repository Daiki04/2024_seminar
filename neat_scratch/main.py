import sys
import math
import random
from itertools import count
from random import choice
import numpy as np
import copy
import pandas as pd

from genome import DefaultGenomes
from gene import DefaultNodeGene, DefaultConnectionGene
from spices import Species, DefaultSpeciesSet
from network import FeedForwardNetwork
from mutate_crossover import DefaultReproduction
from read_config import read_default_config

config = read_default_config()

species_fitness_func_dict = {
    'max': max,
    'min': min,
}

elitism = int(config.get('elitism'))
spice_elitism = int(config.get('spice_elitism'))
survival_threshold = float(config.get('survival_threshold'))
min_species_size = int(config.get('min_species_size'))
species_fitness_func = species_fitness_func_dict[config.get('species_fitness_func')]
max_stagnation = int(config.get('max_stagnation'))
compatibility_threshold = float(config.get('compatibility_threshold'))

num_inputs = int(config.get('num_inputs'))
num_outputs = int(config.get('num_outputs'))

pop_size = int(config.get('pop_size'))
fitness_theashold = float(config.get('fitness_threshold'))

def print_config():
    print(f'elitism: {elitism}')
    print(f'spice_elitism: {spice_elitism}')
    print(f'survival_threshold: {survival_threshold}')
    print(f'min_species_size: {min_species_size}')
    print(f'species_fitness_func: {species_fitness_func}')
    print(f'max_stagnation: {max_stagnation}')
    print(f'num_inputs: {num_inputs}')
    print(f'num_outputs: {num_outputs}')
    print(f'pop_size: {pop_size}')
    print(f'fitness_theashold: {fitness_theashold}')
    print(f'copatibility_threshold: {compatibility_threshold}')

if __name__ == '__main__':
    population = DefaultGenomes.create_new(DefaultGenomes, pop_size)
    species_set = DefaultSpeciesSet(compatibility_threshold=compatibility_threshold)
    generation = 0

    best_fitness_hist = []
    fitness_hist = []

    print_config()

    while True:

        ### 1.種に分ける
        species_set.speciate(population, generation)

        ### 2.適応度の評価

        FeedForwardNetwork.eval_genomes(population)

        best_genome_id, best_genome = max(population.items(), key=lambda x: x[1].fitness)

        fitness_hist.append(best_genome.fitness)
        if generation == 0:
            best_fitness_hist.append(best_genome.fitness)
        else:
            if best_genome.fitness > best_fitness_hist[-1]:
                best_fitness_hist.append(best_genome.fitness)
            else:
                best_fitness_hist.append(best_fitness_hist[-1])

        print(f'Generation: {generation}, Number of Species: {len(species_set.species)}, Best Genome: {best_genome_id}, Fitness: {best_genome.fitness}, compatibility_threshold: {species_set.compatibility_threshold}')

        if best_genome.fitness >= fitness_theashold:
            print('Threshold reached')
            break

        ### 3.交叉 & 突然変異
        reproduction = DefaultReproduction()
        population = reproduction.reproduce(species_set, pop_size, generation)

        if not species_set.species:
            print("Create New Population")
            population = DefaultGenomes.create_new(DefaultGenomes, pop_size)

        generation += 1

    winner = best_genome
    winner_id = best_genome_id

    # print(f'Winner: {winner_id}\n{winner}')
    print(f'Winner Fitness: {winner.fitness}')
    print(f'Winner: {winner}')

    # 結果の保存
    result = pd.DataFrame({
        'best_fitness': best_fitness_hist,
        'fitness': fitness_hist,
    })

    result.to_csv('result.csv', index=False)
    print('Saved result')

    # ネットワークの保存
    winner.save('winner.pkl')
    print('Saved winner')
    