
from __future__ import print_function
import keras, sys
from operators import activation_function_operators
from operators import training_data_operators
from operators import bias_operators
from operators import weights_operators
from operators import optimiser_operators
from operators import dropout_operators,hyperparams_operators
from operators import training_process_operators
from operators import loss_operators
from utils import mutation_utils
from utils import properties
from keras import optimizers
import numpy
from tensorflow.python.keras.layers import Activation
from operators import activation_function_operators
from operators import training_data_operators
from operators import bias_operators
from operators import weights_operators
from operators import optimiser_operators
from operators import dropout_operators, hyperparams_operators
from operators import training_process_operators
from operators import loss_operators
from utils import mutation_utils
from utils import properties
from keras import optimizers
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy
import time
import keras
import sys

def main(model_name):
    model_location = os.path.join('trained_models', model_name)
    dataset = numpy.genfromtxt("./test_models/Book1.csv", delimiter=',')
    print(dataset.shape)
    print(dataset)
    X = dataset[:, 2:4]
    Y = dataset[:, 1]
    if (not os.path.exists(model_location)):
        batch_size = 10
        epochs = 50
        model = Sequential()
        model.add(Dense(4, input_dim=2))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('relu'))
        model.summary()
        model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.SGD(learning_rate=0.01),metrics=['MSE'])
        model.fit(X, Y, batch_size=batch_size, epochs=epochs)
        model.save(os.path.join('trained_models', 'model_trained.h5'))
        score = model.evaluate(X, Y, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score
    else:
        graph1 = tf.Graph()
        with graph1.as_default():
            session1 = tf.compat.v1.Session()
            with session1.as_default():
                model = tf.keras.models.load_model(model_location)
                score = model.evaluate(X, Y, verbose=0)
                print(('score:' + str(score)))
        return score
if (__name__ == '__main__'):
    score = main('')
