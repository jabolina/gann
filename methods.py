from keras import Sequential
from keras.datasets import cifar10
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical


def get_dataset():
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(50000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(10000, 3072)
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return nb_classes, batch_size, input_shape, x_train, y_train, x_test, y_test


def compile_network(network, nb_classes, input_shape):
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    for i in range(nb_layers):
        if not i:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))

    model.add(Dense(nb_classes, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_and_score(network):
    nb_classes, batch_size, input_shape, x_train, y_train, x_test, y_test = get_dataset()
    model = compile_network(network, nb_classes, input_shape)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_test, y_test), verbose=0)

    return model.evaluate(x_test, y_test, verbose=0)[1]
