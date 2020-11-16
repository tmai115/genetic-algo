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

	def compute_fitness_score(self):
		"""
		computing the fitness score of the population.
		args:
			individual: numpy array representing the chromosome of the
			parents
		returns:
			population_size lists of n length each.
		"""
		scores = np.array([
			self.fitness_function(individual) for individual in self.population
			])
		self.fitness_scores = scores
	
	def roulette_wheel_selection(self):
		"""
		Select the fittest individuals based on their fitness scores.
		Each individual is associated with its index in the input array.
		args:
			fitness_scores: numpy array of fitness scores of each
			individual
		returns:
			parents:
		"""
		sum_scores = np.sum(np.abs(self.fitness_scores))
		selection_prob = np.abs(self.fitness_scores) / sum_scores

		parents = random.choices(self.population, weights = selection_prob,
			k = 2)
		return parents
		


# Specify fitness function with fitness score being the number of characters in the target 
# sentence our GA has got correctly:

def fitness_function(individual, target_sentence = "This is a target sentence!"):
	"""
	computes the score of a given individual based on its performance
	approaching the target sentence
	"""

	assert len(target_sentence) == len(individual)
	score = np.sum([
		individual[i] == target_sentence[i]
		for i in range(len(target_sentence))
		])
	return score

