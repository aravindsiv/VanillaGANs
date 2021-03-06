import os,random
import os.path
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.models import Sequential
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys
from keras.models import Model
from keras.utils import np_utils
from tqdm import tqdm

img_rows, img_cols = 32, 32

img_channels = 3

train_with = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print X_train.shape

train_label = np.random.choice(range(10))

X_train = X_train[np.where(y_train == train_with)[0]].astype('float32')/255

num_exs, img_rows, img_cols, img_channels = X_train.shape

# X_train = X_train.reshape((num_exs,img_rows,img_cols,img_channels))

def plot_image(image):
   '''Helper function to plot an MNIST image.'''
   fig = plt.figure()
   ax = fig.add_subplot(1, 1, 1)
   # image = np.reshape(image, (32,32,3), order='F')
   plt.imshow(image)
   plt.xticks(np.array([]))
   plt.yticks(np.array([]))
   plt.show()

# for i in np.random.choice(range(num_exs),10):
#    plot_image(X_train[i])

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def plot_10_by_10_images(images):
   """ Plot 100 MNIST images in a 10 by 10 table. Note that we crop
   the images so that they appear reasonably close together.  The
   image is post-processed to give the appearance of being continued."""
   fig = plt.figure()
   # images = [image[3:25, 3:25] for image in images]
   #image = np.concatenate(images, axis=1)
   for x in range(10):
       for y in range(10):
           ax = fig.add_subplot(10, 10, 10*y+x+1)
           plt.imshow(images[10*y+x])
           plt.xticks(np.array([]))
           plt.yticks(np.array([]))
   plt.show()


def plot_gen(n_ex=16,dim=(4,4), figsize=(10,10) ):
    noise = np.random.uniform(0,1,size=[n_ex,100])
    generated_images = generator.predict(noise)

    plot_10_by_10_images(generated_images)

def plot_loss(losses):
#        display.clear_output(wait=True)
#        display.display(plt.gcf())
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.show()

# Set up our main training loop
def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):

    for e in tqdm(range(nb_epoch)):  
        
        # Make generative images
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]    
        noise_gen = np.random.normal(0,1,size=[BATCH_SIZE,100])
        generated_images = generator.predict(noise_gen)
        
        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        make_trainable(discriminator,True)
        d_loss  = discriminator.train_on_batch(X,y)
        losses["d"].append(d_loss)

        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        make_trainable(discriminator,False)
        g_loss = GAN.train_on_batch(noise_tr, y2 )
        losses["g"].append(g_loss)
        
        if e%1000==0:
            GAN.save_weights("GAN_weights_new_"+str(e))

        # Updates plots
        # if e%plt_frq==plt_frq-1:
        #     plot_loss(losses)
        #     plot_gen(100,(5,5),(12,12))

shp = X_train.shape[1:]
print "Shape is:",shp
dropout_rate = 0.5
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

# Build Generative model ...
nch = 200
g_input = Input(shape=[100])
H = Dense(nch*16*16, init='glorot_normal')(g_input)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Reshape( [16, 16, nch] )(H)
H = UpSampling2D(size=(2, 2))(H)
H = Convolution2D(nch/2, 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(nch/4, 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(3, 1, 1, border_mode='same', init='glorot_uniform')(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()

print "shape is",shp
d_input = Input(shape=shp)
H = Convolution2D(32, 3, 3, border_mode = 'same', activation='relu')(d_input)
# H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(32, 3, 3, border_mode = 'same', activation='relu')(H)
# H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(32)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2,activation='softmax')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()

# Freeze weights in the discriminator for stacked training
make_trainable(discriminator, False)

# Build stacked GAN model
gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()

if(os.path.isfile("GAN_weights_9000")):
    GAN.load_weights("GAN_weights_9000")
    plot_gen(100,(5,5),(12,12))

ntrain = 256
trainidx = random.sample(range(0,X_train.shape[0]), ntrain)
XT = X_train[trainidx,:,:,:]

# Pre-train the discriminator network ...
noise_gen = np.random.uniform(0,1,size=[XT.shape[0],100])
generated_images = generator.predict(noise_gen)
X = np.concatenate((XT, generated_images))
n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1

make_trainable(discriminator,True)
# discriminator.load_weights("discriminator")
discriminator.fit(X,y, nb_epoch=1, batch_size=128)
discriminator.save_weights("discriminator")
y_hat = discriminator.predict(X)

# Measure accuracy of pre-trained discriminator network
y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_tot = y.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print "Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot)

# set up loss storage vector
losses = {"d":[], "g":[]}        

# Train for 6000 epochs at original learning rates
train_for_n(nb_epoch=10000, plt_frq=500,BATCH_SIZE=32)

GAN.save_weights("GAN_weights")

# Train for 2000 epochs at reduced learning rates
# opt.lr.set_value(1e-5)
# dopt.lr.set_value(1e-4)
# train_for_n(nb_epoch=2000, plt_frq=500,BATCH_SIZE=32)

# # Train for 2000 epochs at reduced learning rates
# opt.lr.set_value(1e-6)
# dopt.lr.set_value(1e-5)
# train_for_n(nb_epoch=2000, plt_frq=500,BATCH_SIZE=32)

# Plot the final loss curves
plot_loss(losses)

# Plot some generated images from our GAN
plot_gen(100,(5,5),(12,12))

# Plot real MNIST images for comparison
#plot_real()