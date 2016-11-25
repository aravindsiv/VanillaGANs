import numpy as np

class DiscriminativeModel:
	def __init__(self,nn_input_dim,nn_hdim,nn_output_dim=2):
		# Initialize the parameters to random values. We need to learn these.
		np.random.seed(42)
		self.W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
		self.b1 = np.zeros((1,nn_hdim))
		self.W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
		self.b2 = np.zeros((1,nn_output_dim))

	def forward_pass(self,X):
		# Forward propogation
		self.z1 = np.dot(X, self.W1) + self.b1
		self.a1 = np.tanh(z1)
		self.z2 = np.dot(a1,self.W2) + self.b2
		exp_scores = np.exp(z2)
		self.probs = exp_scores / np.sum(exp_scores,axis=1)

	def predict(self,probs):
		# Predict an output:
		# - 0 if the data was drawn from the generative model
		# - 1 if the data was drawn from the data distribution
		return np.argmax(probs,axis=1)

	def backward_pass(self, learning_rate, x, x_fake probs, probs_fake):
		# Backpropagation
		delta3 = np.vstack([probs,probs_fake])
		loss = np.vstack([np.ones_like(probs),np.zeros_like(probs_fake)])
		delta3 -= loss
		dW2 = np.dot(a1.T,delta3)
		db2 = np.sum(delta3,axis=0)
		delta2 = np.dot(delta3,W2.T) * (1-np.power(self.a1,2))
		dW1 = np.dot(np.vstack([x,x_fake]),delta2)
		db1 = np.sum(delta2,axis=0)

		self.W1 += learning_rate * dW1
		self.b1 += learning_rate * db1
		self.W2 += learning_rate * dW2
		self.b2 += learning_rate * db2

	def calculate_loss(self,x,x_fake):
		num_examples = len(x)
		probs = self.predict(x)
		probs_fake = self.predict(x_fake)
		correct_logprobs = np.log(probs[:,1])+np.log(probs_fake[:,1])
		data_loss = np.sum(correct_logprobs)
		return 1./num_examples * data_loss

