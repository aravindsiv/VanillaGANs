import numpy as np
from DiscriminativeModel import DiscriminativeModel
from GenerativeModel import GenerativeModel
import cPickle as pkl
import matplotlib
import matplotlib.pyplot as plt 
import seaborn

images_fname = "train_data.pkl"

with open(images_fname,'r') as f:
	imgs, labels = pkl.load(f)

print "Images loaded!"

train_with = 9

images = imgs/255.0
# images = imgs[np.where(labels==train_with)[0]]/255.0 # A little bit of housekeeping to keep the input values between 0 and 1.

N = images.shape[0] # No. of training examples
k = 1 # How many times per training iteration should we update the discriminator?
m = 256 # What is the size of the minibatch of samples?
num_iters = 5000
nn_input_dim_dis = nn_output_dim_gen = images.shape[1]*images.shape[2] 
nn_input_dim_gen = 100 
nn_hdim_gen = nn_hdim_dis = 320

def plot_mnist_image(image):
    '''Helper function to plot an MNIST image.'''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def plot_10_by_10_images(images):
    """ Plot 100 MNIST images in a 10 by 10 table. Note that we crop
    the images so that they appear reasonably close together.  The
    image is post-processed to give the appearance of being continued."""
    fig = plt.figure()
    images = [image[3:25, 3:25] for image in images]
    #image = np.concatenate(images, axis=1)
    for x in range(10):
        for y in range(10):
            ax = fig.add_subplot(10, 10, 10*y+x+1)
            ax.matshow(images[10*y+x], cmap = matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

def get_minibatch(images,m):
	'''Helper function to generate a random minibatch of m examples'''
	indices = np.random.choice(range(images.shape[0]),size=m,replace=False)
	return images[indices].reshape((m,nn_input_dim_dis))

Discriminator = DiscriminativeModel(nn_input_dim_dis,nn_hdim_dis)
Generator = GenerativeModel(nn_input_dim_gen,nn_hdim_gen,nn_output_dim_gen)

loss_discriminator = []
loss_generator = []

learning_rate = 5e-4
momentum = 0.5
momentum_training = True

# # Pre-train the discriminator
# num_epochs = 100
# for j in range(num_epochs):
# 	# Sample minibatch of m noise samples from noise prior
# 	prior_z = np.random.normal(size=(m,nn_input_dim_gen))
# 	z = Generator.forward_pass(prior_z)
# 	# Sample minibatch of m examples from data generating distribution
# 	x = get_minibatch(images,m)
# 	# Update the discriminator by ascending its stochastic gradient
# 	Discriminator.backward_pass(learning_rate,momentum, momentum_training, np.vstack([x,z]))

# print "Pre-training of discriminator finished!"
prior_z = np.random.normal(size=(m,nn_input_dim_gen))
z = Generator.forward_pass(prior_z)
x = get_minibatch(images,m)
print "The discriminator achieved a loss of ", Discriminator.calculate_loss(x,z)

for i in range(num_iters):
	for j in range(k):
		# Sample minibatch of m noise samples from noise prior
		prior_z = np.random.normal(size=(m,nn_input_dim_gen))
		z = Generator.forward_pass(prior_z)
		# Sample minibatch of m examples from data generating distribution
		x = get_minibatch(images,m)
		# Update the discriminator by ascending its stochastic gradient
		Discriminator.backward_pass(learning_rate,momentum, momentum_training, np.vstack([x,z]))
		# Calculate loss for the discriminator
		loss_discriminator.append(Discriminator.calculate_loss(x,z))
	# Sample minibatch of m noise samples from noise prior
	prior_z = np.random.normal(size=(m,nn_input_dim_gen))
	z = Generator.forward_pass(prior_z)
	# Update the generator by descending its stochastic gradient
	Generator.backward_pass(learning_rate,momentum, momentum_training, z,Discriminator.backward_pass_for_generator(z),prior_z)
	# Calculate loss for the generator
	prior_z = np.random.normal(size=(m,nn_input_dim_gen))
	z = Generator.forward_pass(prior_z)
	discriminator_outputs = Discriminator.forward_pass(z)
	loss_generator.append(Generator.calculate_loss(discriminator_outputs))
	if ((i+1) % 1000 == 0):
		print "Iteration " + str(i+1)
		print "Generator loss is ",loss_generator[i]
		print "Discriminator loss is ",loss_discriminator[i]
		prior_z = np.random.normal(size=(100,nn_input_dim_gen))
		rand_output = Generator.forward_pass(prior_z)
		plot_10_by_10_images(255.0*rand_output.reshape(100,images.shape[1],images.shape[2]))

prior_z = np.random.normal(size=(100,nn_input_dim_gen))
rand_output = Generator.forward_pass(prior_z)
plot_10_by_10_images(255.0*rand_output.reshape(100,images.shape[1],images.shape[2]))

plt.plot(loss_discriminator)
plt.show()
plt.plot(loss_generator)
plt.show()