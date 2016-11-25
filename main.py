import numpy as np
from DiscriminativeModel import DiscriminativeModel
from GenerativeModel import GenerativeModel
import cPickle as pkl
import matplotlib
import matplotlib.pyplot as plt 

images_fname = "train_data.pkl"

with open(images_fname,'r') as f:
	imgs, labels = pkl.load(f)

print "Images loaded!"

train_with = 4

images = imgs[np.where(labels==train_with)[0]]

images = images/255.0 # A little bit of housekeeping to keep the input values between 0 and 1.

N = images.shape[0] # No. of training examples
k = 1 # How many times per training iteration should we update the discriminator?
m = 16 # What is the size of the minibatch of samples?
num_iters = 10000
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

def get_minibatch(images,m):
	'''Helper function to generate a random minibatch of m examples'''
	indices = np.random.choice(range(images.shape[0]),size=m,replace=False)
	return images[indices].reshape((m,nn_input_dim_dis))

Discriminator = DiscriminativeModel(nn_input_dim_dis,nn_hdim_dis)
Generator = GenerativeModel(nn_input_dim_gen,nn_hdim_gen,nn_output_dim_gen)

loss_discriminator = []
loss_generator = []

learning_rate = 5e-4

for i in range(num_iters):
	print "Iteration " + str(i+1)
	for j in range(k):
		# Sample minibatch of m noise samples from noise prior
		prior_z = np.random.normal(size=(m,nn_input_dim_gen))
		z = Generator.forward_pass(prior_z)
		# Sample minibatch of m examples from data generating distribution
		x = get_minibatch(images,m)
		# Update the discriminator by ascending its stochastic gradient
		Discriminator.backward_pass(learning_rate,np.vstack([x,z]))
		loss_discriminator.append(Discriminator.calculate_loss(x,z))
	# Sample minibatch of m noise samples from noise prior
	prior_z = np.random.normal(size=(m,nn_input_dim_gen))
	z = Generator.forward_pass(prior_z)
	# Update the generator by descending its stochastic gradient
	generator_outputs = Generator.forward_pass(prior_z)
	Generator.backward_pass(learning_rate,generator_outputs,Discriminator.backward_pass_for_generator(generator_outputs),prior_z)
	discriminator_outputs = Discriminator.forward_pass(generator_outputs)
	loss_generator.append(Generator.calculate_loss(discriminator_outputs))
	# if (i % 100 == 0):
	# 	prior_z = np.random.normal(size=(1,nn_input_dim_gen))
	# 	rand_output = Generator.forward_pass(prior_z)
	# 	plot_mnist_image(255.0*rand_output.reshape(images.shape[1],images.shape[2]))

plt.plot(loss_discriminator)
plt.show()
plt.plot(loss_generator)
plt.show()