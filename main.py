import numpy as np
from DiscriminativeModel import DiscriminativeModel
from GenerativeModel import GenerativeModel
import cPickle as pkl
import matplotlib
import matplotlib.pyplot as plt 

images_fname = "test_data.pkl"

with open(images_fname,'r') as f:
	images = pkl.load(f)

images = images/255.0 # A little bit of housekeeping to keep the input values between 0 and 1.

N = images.shape[0] # No. of training examples
k = 1 # How many times per training iteration should we update the discriminator?
m = 16 # What is the size of the minibatch of samples?
num_iters = 1
nn_input_dim_dis = nn_output_dim_gen = images.shape[1]*images.shape[2] 
nn_input_dim_gen = 10 
nn_hdim_gen = nn_hdim_dis = 100

def plot_mnist_image(image):
    '''Helper function to plot an MNIST image.'''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def get_minibatch(images,m):
	'''Helper function to generate a random minibatch of m examples'''
	indices = np.random.choice(range(images.shape[0]),size=m,replace=False)
	return images[indices].reshape((m,nn_input_dim_dis))

Discriminator = DiscriminativeModel(nn_input_dim_dis,nn_hdim_dis)
Generator = GenerativeModel(nn_input_dim_gen,nn_hdim_gen,nn_output_dim_gen)

learning_rate = 0.05

for i in range(num_iters):
	for j in range(k):
		# Sample minibatch of m noise samples from noise prior
		prior_z = np.random.normal(size=(m,nn_input_dim_gen))
		z = Generator.forward_pass(prior_z)
		# Sample minibatch of m examples from data generating distribution
		x = get_minibatch(images,m)
		# Update the discriminator by ascending its stochastic gradient
		Discriminator.backward_pass(learning_rate,np.vstack([x,z]))
		# print "Discriminator loss is " + str(Discriminator.calculate_loss(x,z))
	# Sample minibatch of m noise samples from noise prior
	prior_z = np.random.normal(size=(m,nn_input_dim_gen))
	z = Generator.forward_pass(prior_z)
	# Update the generator by descending its stochastic gradient
	generator_outputs = Generator.forward_pass(prior_z)
	Generator.backward_pass(learning_rate,generator_outputs,Discriminator.backward_pass_for_generator(generator_outputs),prior_z)
	# print "Generator loss is " + str(Generator.calculate_loss(generator_outputs))
