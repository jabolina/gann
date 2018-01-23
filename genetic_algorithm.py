import random
from _operator import add
from functools import reduce

from network import Network


class GA():
    def __init__(self, network_options, retain=0.4, random_select=0.1, mutation=0.2):
        self.network_options = network_options
        self.retain = retain
        self.random_select = random_select
        self.mutation = mutation

    def create_population(self, count):
        pop = []
        for _ in range(count):
            network = Network(self.network_options)
            network.create_random()
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        return network.accuracy

    def grade(self, pop):
        summ = reduce(add, (self.fitness(network) for network in pop))
        return summ / float(len(pop))

    def breed(self, individual_a, individual_b):
        children = []
        for _ in range(2):
            child = {}
            for opt in self.network_options:
                child[opt] = random.choice([individual_a.network[opt], individual_b.network[opt]])

            network = Network(self.network_options)
            network.create_dict(child)

            if self.mutation > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, network):
        mutation = random.choice(list(self.network_options.keys()))
        network.network[mutation] = random.choice(self.network_options[mutation])

        return network

    def evolve(self, pop):
        graded = [(self.fitness(network), network) for network in pop]
        graded = [x[1] for x in sorted(graded, key=lambda k: k[0], reverse=True)]
        retain_length = int(len(graded) * self.retain)

        parents = graded[:retain_length]

        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        nb_parents = len(parents)
        length = len(pop) - nb_parents
        children = []

        while len(children) < length:
            individual_a = random.randint(0, nb_parents-1)
            individual_b = random.randint(0, nb_parents-1)

            if individual_a != individual_b:
                individual_a = parents[individual_a]
                individual_b = parents[individual_b]

                babies = self.breed(individual_a, individual_b)

                for baby in babies:
                    if len(children) < length:
                        children.append(baby)

        parents.extend(children)
        return parents




