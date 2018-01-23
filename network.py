import random

from methods import train_and_score


class Network:
    def __init__(self, options):
        self.accuracy = 0
        self.options = options
        self.network = {}

    def create_random(self):
        for key in self.options:
            self.network[key] = random.choice(self.options[key])

    def create_dict(self, network):
        self.network = network

    def train(self):
        if not self.accuracy:
            self.accuracy = train_and_score(self.network)

