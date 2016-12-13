from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Input
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist
import random
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
    	l.trainable = val

def plot_loss(losses):
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.show()

def plot_10_by_10_images(images):
   '''Plot 100 MNIST images, real or generated in a 10x10 grid'''
   fig = plt.figure()
   images = images.reshape(images.shape[0],28,28)
   for x in range(10):
       for y in range(10):
           ax = fig.add_subplot(10, 10, 10*y+x+1)
           plt.imshow(images[10*y+x], cmap = matplotlib.cm.binary)
           plt.xticks(np.array([]))
           plt.yticks(np.array([]))
   plt.show()

def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):

    for e in tqdm(range(nb_epoch)):  
        
        # Make generative images
        image_batch = X_train[np.random.choice(range(X_train.shape[0]),BATCH_SIZE,replace=False)]
        noise_gen = np.random.normal(size=(BATCH_SIZE,100))
        generated_images = generative_model.predict(noise_gen)
        
        # Train discriminator on generated images
        image_batch = image_batch.reshape(image_batch.shape[0],image_batch.shape[1]*image_batch.shape[2])
        X = np.vstack([image_batch, generated_images])
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        make_trainable(discriminator_model,True)
        d_loss  = discriminator_model.train_on_batch(X,y)
        losses["d"].append(d_loss)

        # train Generator-discriminator_model stack on input noise to non-generated output class
        noise_tr = np.random.normal(size=(BATCH_SIZE,100))
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        make_trainable(discriminator_model,False)
        g_loss = gan_model.train_on_batch(noise_tr, y2)
        losses["g"].append(g_loss)

        if (e+1)%plt_frq==0:
          pass
        	# plot_10_by_10_images(generative_model.predict(np.random.normal(size=(100,100))))

sgd = SGD(lr=0.005, momentum=0.5, decay=0.0, nesterov=True)

discriminator_model = Sequential()
discriminator_model.add(Dense(320,input_dim=784))
discriminator_model.add(Activation('tanh'))
discriminator_model.add(Dense(2))
discriminator_model.add(Activation('softmax'))
discriminator_model.compile(loss='categorical_crossentropy',optimizer='sgd')

generative_model = Sequential()
generative_model.add(Dense(320,input_dim=100))
generative_model.add(Activation('relu'))
generative_model.add(Dense(784))
generative_model.add(Activation('sigmoid'))
generative_model.compile(loss='binary_crossentropy',optimizer='sgd')

gan_input = Input(shape=[100])
gen = generative_model(gan_input)
dis = discriminator_model(gen)
gan_model = Model(gan_input,dis)
gan_model.compile(loss='categorical_crossentropy',optimizer='sgd')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train/255.0

# Pre=training the discriminator
ntrain = X_train.shape[0]/10
XT = X_train[np.random.choice(range(X_train.shape[0]),ntrain,replace=False)]
XT = XT.reshape(XT.shape[0],XT.shape[1]*XT.shape[2])
noise_gen = np.random.normal(size=(ntrain,100))
generated_images = generative_model.predict(noise_gen)
X = np.vstack([XT, generated_images])
n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1

make_trainable(discriminator_model,True)
discriminator_model.fit(X,y, nb_epoch=1, batch_size=256)


losses = {"d":[], "g":[]}

train_for_n(nb_epoch=50000, plt_frq=1000,BATCH_SIZE=256)

plot_loss(losses)

plot_10_by_10_images(generative_model.predict(np.random.normal(size=(100,100))))