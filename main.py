import logging
import os
from genetic_algorithm import GA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


def train_networks(networks):
    logging.info('[+] Training network.')
    print('[+] Training network.')
    i = 1
    for network in networks:
        logging.info('[+] Training %d out of %d networks.', i, len(networks))
        print('[+] Training ' + str(i) + ' out of ' + str(len(networks)))
        network.train()
        i += 1


def avg_accuracy(networks):
    accuracy = 0.0
    for network in networks:
        accuracy += network.accuracy

    return accuracy / float(len(networks))


def generate(generations, population, options):
    optimizer = GA(options)
    logging.info('[+] Generating populations.')
    print('[+] Generating populations.')
    networks = optimizer.create_population(population)

    for _ in range(generations):
        print('[+] Population ' + str(_ + 1) + ' out of ' + str(generations))
        logging.info('[+] Population %d out of %d', _+1, generations)
        train_networks(networks)

        if _ != generations - 1:
            networks = optimizer.evolve(networks)

    networks = sorted(networks, key=lambda k: k.accuracy, reverse=True)
    print_networks(networks[:5])


def print_networks(networks):
    for network in networks:
        print(network.accuracy * 100)


if __name__ == '__main__':
    generations = 10
    population = 20
    options = {
        'nb_neurons': [32, 64, 128, 256, 512, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']
    }
    generate(generations, population, options)
