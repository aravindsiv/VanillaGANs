import numpy as np
import math

class GenerativeModel:
	def __init__(self,nn_input_dim,nn_hdim,nn_output_dim=2):
		# Initialize the parameters to random values. We need to learn these.
		np.random.seed(42)
		self.W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
		self.b1 = np.zeros((1,nn_hdim))
		self.W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
		self.b2 = np.zeros((1,nn_output_dim))
		self.a1 = None

	def forward_pass(self,X):
		f=np.vectorize(sigmoid)
		z1 = np.dot(X, self.W1) + self.b1
		a1 = np.maximum(z1,0)
		z2 = np.dot(a1,self.W2) + self.b2
		a2 = f(z2)
		return a2

	def predict(self,probs):
		# Predict an output:
		# - 0 if the data was drawn from the generative model
		# - 1 if the data was drawn from the data distribution
		return np.argmax(probs,axis=1)


	def backward_pass(self, learning_rate, reg_lambda, output, loss, X):
		f = np.vectorize(sigmoid)
		z1 = np.dot(X, self.W1) + self.b1
		a1 = f(z1)
		delta3=loss*output*(1-output)
		dW2=(a1.T).dot(delta3)
		db2=np.sum(delta3,axis=0,keepdims=true)

		delta2= delta3.dot(np.where(z1>1,w2,0))

		dW1=np.dot(X.T,delta2)
		db1=np.sum(delta2,axis=0)

	def calculate_loss(self,x,x_fake):
		num_examples = len(x)
		probs = self.predict(x)
		probs_fake = self.predict(x_fake)
		correct_logprobs = np.log(probs[:,1])+np.log(probs_fake[:,1])
		data_loss = np.sum(correct_logprobs)
		return 1./num_examples * data_loss

	def sigmoid(x):
		return 1 / (1 + math.exp(-x))