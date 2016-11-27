from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Input
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

discriminator_model = Sequential()
discriminator_model.add(Dense(320,input_dim=784))
discriminator_model.add(Activation('tanh'))
discriminator_model.add(Dense(2))
discriminator_model.add(Activation('softmax'))
discriminator_model.compile(loss='categorical_crossentropy',optimizer='sgd')
discriminator_model.summary()

generative_model = Sequential()
generative_model.add(Dense(320,input_dim=100))
generative_model.add(Activation('relu'))
generative_model.add(Dense(784))
generative_model.add(Activation('sigmoid'))
generative_model.compile(loss='binary_crossentropy',optimizer='sgd')
generative_model.summary()

gan_input = Input(shape=[100])
H = generative_model(gan_input)
gan_v = discriminator_model(H)
gan_model = Model(gan_input,gan_v)
gan_model.compile(loss='categorical_crossentropy',optimizer='sgd')
gan_model.summary()