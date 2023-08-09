
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
from utils import mutation_utils
from utils import mutation_utils
from utils import mutation_utils
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
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
import numpy
import numpy as np
import keras
import time

def main(model_name):
    model_location = os.path.join('trained_models', model_name)
    list1 = [1, 1]
    label1 = [0]
    list2 = [1, 0]
    label2 = [1]
    list3 = [0, 0]
    label3 = [0]
    list4 = [0, 1]
    label4 = [1]
    train_data = np.array((list1, list2, list3, list4))
    label = np.array((label1, label2, label3, label4))
    if (not os.path.exists(model_location)):
        batch_size = 4
        epochs = 1000
        model = Sequential()
        model.add(Dense(4, input_dim=2, kernel_initializer='glorot_uniform'))
        model.add(Activation('sigmoid'))
        model.add(Dense(1, input_dim=4, kernel_initializer='glorot_uniform'))
        model.add(Activation('sigmoid'))
        model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.SGD(lr=0.05, decay=1e-06, momentum=0.11, nesterov=True), metrics=['accuracy'])
        model.fit(train_data, label, batch_size=batch_size, epochs=epochs, verbose=1)
        model.save(os.path.join('trained_models', 'model_trained.h5'))
        score = model.evaluate(train_data, label, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score
    else:
        graph1 = tf.Graph()
        with graph1.as_default():
            session1 = tf.compat.v1.Session()
            with session1.as_default():
                model = tf.keras.models.load_model(model_location)
                score = model.evaluate(train_data, label, verbose=1)
                print(('score:' + str(score)))
        return score
if (__name__ == '__main__'):
    score = main('')
