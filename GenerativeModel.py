import numpy as np

class GenerativeModel:
	def __init__(self,nn_input_dim,nn_hdim,nn_output_dim):
		# Initialize the parameters to random values. We need to learn these.
		self.W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
		self.b1 = np.zeros((1,nn_hdim))
		self.W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
		self.b2 = np.zeros((1,nn_output_dim))
		self.z1 = self.a1 = self.z2 = None
		self.v_previous_W1 = 0
		self.v_previous_b1 = 0
		self.v_previous_W2 = 0
		self.v_previous_b2 = 0
		self.v_W1 = 0
		self.v_b1 = 0
		self.v_W2 = 0
		self.v_b2 = 0

	def forward_pass(self,X):
		# Forward propagation
		self.z1 = np.dot(X, self.W1) + self.b1
		self.a1 = np.maximum(self.z1,0)
		self.z2 = np.dot(self.a1,self.W2) + self.b2
		self.out = 1/(1+np.exp(-self.z2))
		return self.out

	def backward_pass(self, learning_rate, momentum, momentum_training, out, loss, X):
		# Backpropagation
		delta3=loss*out*(1-out)
		dW2=(self.a1.T).dot(delta3)
		db2=np.sum(delta3,axis=0,keepdims=True)
		delta2= np.where(self.z1>0,np.dot(delta3,self.W2.T),np.zeros_like(self.z1))
		dW1=np.dot(X.T,delta2)
		db1=np.sum(delta2,axis=0)

		if(momentum_training==True):
			self.v_previous_W1 = self.v_W1
			self.v_previous_b1 = self.v_b1
			self.v_previous_W2 = self.v_W2
			self.v_previous_b2 = self.v_b2

			self.v_W1 = momentum * self.v_W1 - learning_rate * dW1
			self.v_b1 = momentum * self.v_b1 - learning_rate * db1
			self.v_W2 = momentum * self.v_W2 - learning_rate * dW2
			self.v_b2 = momentum * self.v_b2 - learning_rate * db2

			self.W1 += -momentum * self.v_previous_W1 + (1 + momentum) * self.v_W1
			self.b1 += -momentum * self.v_previous_b1 + (1 + momentum) * self.v_b1
			self.W2 += -momentum * self.v_previous_W2 + (1 + momentum) * self.v_W2
			self.b2 += -momentum * self.v_previous_b2 + (1 + momentum) * self.v_b2

		else:
			self.W1 -= learning_rate * dW1
			self.b1 -= learning_rate * db1
			self.W2 -= learning_rate * dW2
			self.b2 -= learning_rate * db2

	def calculate_loss(self,discriminator_outputs):
		num_examples = discriminator_outputs.shape[0]
		correct_logprobs = np.log(discriminator_outputs[:,1])
		data_loss = np.sum(correct_logprobs)
		return 1./num_examples * data_loss