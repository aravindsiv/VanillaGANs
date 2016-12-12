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

def plot_gen(n_ex=16,dim=(4,4), figsize=(10,10) ):
    noise = np.random.normal(0,1,size=[n_ex,100])
    generated_images = generative_model.predict(noise)

    plot_10_by_10_images(generated_images)

def plot_loss(losses):
#        display.clear_output(wait=True)
#        display.display(plt.gcf())
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.show()


def plot_10_by_10_images(images):
   """ Plot 100 MNIST images in a 10 by 10 table. Note that we crop
   the images so that they appear reasonably close together.  The
   image is post-processed to give the appearance of being continued."""
   fig = plt.figure()
   images = images.reshape(images.shape[0],28,28)
   # images = [image[3:25, 3:25] for image in images]
   #image = np.concatenate(images, axis=1)
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
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE)]
        noise_gen = np.random.normal(0,1,size=[BATCH_SIZE,100])
        generated_images = generative_model.predict(noise_gen)
        
        # Train discriminator on generated images
        image_batch = image_batch.reshape(image_batch.shape[0],image_batch.shape[1]*image_batch.shape[2])
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        make_trainable(discriminator_model,True)
        d_loss  = discriminator_model.train_on_batch(X,y)
        losses["d"].append(d_loss)

        # train Generator-discriminator_model stack on input noise to non-generated output class
        noise_tr = np.random.normal(0,1,size=[BATCH_SIZE,100])
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        make_trainable(discriminator_model,False)
        g_loss = gan_model.train_on_batch(noise_tr, y2 )
        losses["g"].append(g_loss)

        if e%plt_frq==0:
        	gan_model.save_weights("GAN_weights_k")

discriminator_model = Sequential()
discriminator_model.add(Dense(320,input_dim=784))
discriminator_model.add(Activation('tanh'))
discriminator_model.add(Dense(2))
discriminator_model.add(Activation('softmax'))
discriminator_model.compile(loss='categorical_crossentropy',optimizer='adam')
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
gan_model.compile(loss='categorical_crossentropy',optimizer='adam')
gan_model.summary()

# gan_model.load_weights("GAN_weights_k")
# plot_gen(100,(5,5),(12,12))
# sys.exit()
#load mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train[np.where(y_train == train_with)[0]].astype('float32')/255

print "X_train shape",X_train.shape

ntrain = 256
trainidx = random.sample(range(0,X_train.shape[0]), ntrain)

#shape of XT is (256, 28, 28)
XT = X_train[trainidx]
XT = XT.reshape(XT.shape[0],XT.shape[1]*XT.shape[2])
print "XT shape",XT.shape
noise_gen = np.random.normal(0,1,size=[XT.shape[0],100])
generated_images = generative_model.predict(noise_gen)
print "shape of generated images is",generated_images.shape
X = np.concatenate((XT, generated_images))
n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1
print "shape of y is",y.shape
print "shape of X is",X.shape

make_trainable(discriminator_model,True)
# discriminator.load_weights("discriminator")
# discriminator_model.fit(X,y, nb_epoch=1, batch_size=128)
# discriminator_model.save_weights("discriminator_model")
# y_hat = discriminator_model.predict(X)

# y_hat_idx = np.argmax(y_hat,axis=1)
# y_idx = np.argmax(y,axis=1)
# diff = y_idx-y_hat_idx
# n_tot = y.shape[0]
# n_rig = (diff==0).sum()
# acc = n_rig*100.0/n_tot
# print "Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot)


losses = {"d":[], "g":[]}

train_for_n(nb_epoch=100000, plt_frq=500,BATCH_SIZE=32)

plot_loss(losses)

plot_gen(100,(5,5),(12,12))