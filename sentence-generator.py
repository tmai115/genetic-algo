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

	def get_best(self):
		best_fitness_score = max(self.fitness_scores)
		for each_id in range(len(self.population)):
			if self.fitness_scores[each_id] == best_fitness_score:
				best_s = ''.join(self.population[each_id])
				return best_s

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
		
	def run(self):
		"""
		running the genetic algorithm to produce a new generation.
		"""
		def cross_over(parents):
			"""
			produces a new individual by combining the genetic information
			of both parents
			args:
				individual_1: numpy array representing the chromosomes
				of the first parent
				individual_2: numpy array representing the chromosomes of
				the second parent
			returns:
				child: newly created individual by cross over of the 2 parents.
			"""
			if np.random.uniform() <= self.crossover_prob:
				parent_1, parent_2 = parents
				crossover_point = np.random.choice(
					range(1, self.num_attributes))
				child = np.concatenate(
					(parent_1[:crossover_point],
					parent_2[crossover_point:]))
				return child
			else:
				return random.choices(parents)[0]

		def mutate(individual):
			"""
			produces a new individual by mutating the original one.
			args:
				individual: numpy array representing the chromosomes of the parent.
			returns:
			new: newly mutated individual.
			"""
			new_individual = []
			for attribute in individual:
				if np.random.uniform() <= self.mutation_prob:
					new_individual.append(random.choice(string.ascii_letters))
				else:
					new_individual.append(attribute)
			return new_individual

		new_population = []

		# reproduces the new population
		for _ in range(self.population_size):
			parents = self.roulette_wheel_selection()
			child = cross_over(parents)
			child = mutate(child)
			new_population.append(child)

		self.population = np.array(new_population)
		






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


def main():
	target_sentence = "This is a target sentence!"
	np.random.seed(123)
	MAX_GEN = 100 # termination
	MAX_SUCCESS = 3 # termination
	NUM_ATTRIBUTES = len(target_sentence)
	POPULATION_SIZE = 500
	MUTATION_PROB = .01
	CROSSOVER_PROB = .75

	GA = GeneticAlgorithm(fitness_function,
						  num_attributes=NUM_ATTRIBUTES,
						  population_size=POPULATION_SIZE,
						  mutation_prob=MUTATION_PROB,
						  crossover_prob=CROSSOVER_PROB)
	GA.initialise_population()
	scores = []
	generation_counter = 0
	success_counter = 0

	while generation_counter < MAX_GEN and success_counter < MAX_SUCCESS:
		GA.compute_fitness_score()
		scores.append(np.mean(GA.fitness_scores))
		print("Generation", generation_counter, ", avg score:",
			scores[generation_counter],
			", best:", GA.get_best())
		if GA.get_best() == target_sentence:
			success_counter += 1
		GA.run()
		generation_counter += 1

	plt.plot(scores)
	plt.xlabel("Generation")
	plt.ylabel("Average score per generation")
	plt.show()

main()


