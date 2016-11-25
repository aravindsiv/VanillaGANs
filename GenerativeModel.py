import numpy as np
import math

class GenerativeModel:
	def __init__(self,nn_input_dim,nn_hdim,nn_output_dim):
		# Initialize the parameters to random values. We need to learn these.
		np.random.seed(42)
		self.W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
		self.b1 = np.zeros((1,nn_hdim))
		self.W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
		self.b2 = np.zeros((1,nn_output_dim))
		self.z1 = self.a1 = self.z2 = None

	def forward_pass(self,X):
		self.z1 = np.dot(X, self.W1) + self.b1
		self.a1 = np.maximum(self.z1,0)
		self.z2 = np.dot(self.a1,self.W2) + self.b2
		out = 1/(1+np.exp(-self.z2))
		return out

	def backward_pass(self, learning_rate, reg_lambda, output, loss, X):
		delta3=loss*output*(1-output)
		dW2=(self.a1.T).dot(delta3)
		db2=np.sum(delta3,axis=0,keepdims=true)

		delta2= delta3.dot(np.where(self.z1>1,self.W2,0))

		dW1=np.dot(X.T,delta2)
		db1=np.sum(delta2,axis=0)

	def calculate_loss(self,x,x_fake):
		pass