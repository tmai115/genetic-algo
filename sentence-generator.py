# A system using a genetic algorithm to generate a target sentence
import random
import matplotlib.pyplot as plt 
import numpy as np 
import string

# Fixing seed of reproducibility
seed = 123
np.random.seed(seed)
random.seed(seed)

# Class to represent genetic algo, with method to initialise population
class GeneticAlgorithm:
	def __init__(self, fitness_function, num_attributes = 2, population_size = 100,
				crossover_prob = .75, mutation_prob = .05):
		self.fitness_function = fitness_function
		self.num_attributes = num_attributes
		self.population_size = population_size
		self.crossover_prob = crossover_prob
		self.mutation_prob = mutation_prob
		self.population = None
		self.population_avg_score = 0
		self.fitness_scores = None
		self.fittest_individuals = None

	def initialise_population(self):
		"""init a population of individuals
		args:
			num_attributes: length of each individual (attributes)
			population_size: number of individuals
		returns:
			population_size lists of n length each.
		"""
		attributes = []

		for attribute in range(self.num_attributes):
			attributes.append(
				np.random.choice(
					list(string.punctuation + string.ascii_letters +
						string.whitespace),
					size = self.population_size))
		self.population = np.array(attributes).T
