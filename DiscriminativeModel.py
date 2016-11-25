import numpy as np

class DiscriminativeModel:
	def __init__(self,nn_input_dim,nn_hdim,nn_output_dim=2):
		# Initialize the parameters to random values. We need to learn these.
		np.random.seed(42)
		self.W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
		self.b1 = np.zeros((1,nn_hdim))
		self.W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
		self.b2 = np.zeros((1,nn_output_dim))
		self.z1 = self.a1 = self.z2 = self.probs = self.dX =  None

	def forward_pass(self,X):
		# Forward propagation
		self.z1 = np.dot(X, self.W1) + self.b1
		self.a1 = np.tanh(self.z1)
		self.z2 = np.dot(self.a1,self.W2) + self.b2
		exp_scores = np.exp(self.z2)
		self.probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)
		return self.probs

	def backward_pass(self, learning_rate, X):
		# Backpropagation
		probs = self.forward_pass(X)
		delta3 = probs
		loss = np.vstack([np.ones((X.shape[0]/2,1)),np.zeros((X.shape[0]/2,1))])
		delta3 -= loss
		dW2 = np.dot(self.a1.T,delta3)
		db2 = np.sum(delta3,axis=0)
		delta2 = np.dot(delta3,self.W2.T) * (1-np.power(self.a1,2))
		dW1 = np.dot(X.T,delta2)
		db1 = np.sum(delta2,axis=0)

		self.W1 -= learning_rate * dW1
		self.b1 -= learning_rate * db1
		self.W2 -= learning_rate * dW2
		self.b2 -= learning_rate * db2

	def backward_pass_for_generator(self,X):
		# Helper function to backpropagate the loss through the discriminator network without updating its weights
		# Returns the loss at the output layer of the generator
		probs = self.forward_pass(X)
		delta3 = probs
		loss = np.ones((X.shape[0],1)) 
		delta3 -= loss
		dW2 = np.dot(self.a1.T,delta3)
		db2 = np.sum(delta3,axis=0)
		delta2 = np.dot(delta3,self.W2.T) * (1-np.power(self.a1,2))
		dW1 = np.dot(X.T,delta2)
		db1 = np.sum(delta2,axis=0)

		dX = np.dot(delta2,self.W1.T)

		return dX

	def calculate_loss(self,x,x_fake):
		num_examples = x.shape[0]
		probs = self.forward_pass(x)
		probs_fake = self.forward_pass(x_fake)
		correct_logprobs = np.log(probs[:,1])+np.log(probs_fake[:,1])
		data_loss = np.sum(correct_logprobs)
		return 1./num_examples * data_loss

