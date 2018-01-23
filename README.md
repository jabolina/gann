# GA-NN

An neural network using a genetic algorithm to find the best parameters inside a set of differents options.

## About

Using the cifar 10 dataset, with a simple Sequential network. We create 20 populations in 10 generations, this means that
for every generation is created 20 neural networks with differents parameters picked at random, but passed in a set of options.
The options are <i>nb_neurons, nb_layers, activation</i> and <i>optimizer</i>.

