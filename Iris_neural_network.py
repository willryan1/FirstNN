import numpy as np
import os
from numpy import random

#This seeds the data which ensures the data is randomized but is the same every time it runs
#np.random.seed(0)

#This sets the learning rate which determines how much the weights and biases change each iteration in the training loop
learning_rate = .5

#this initializes a numpy array to store the data we will be using for testing and training
data = np.array([[5.1,3.5,1.4,0.2],
				[4.9,3.0,1.4,0.2],
				[4.7,3.2,1.3,0.2],
				[4.6,3.1,1.5,0.2],
				[5.0,3.6,1.4,0.2],
				[5.4,3.9,1.7,0.4],
				[4.6,3.4,1.4,0.3],
				[5.0,3.4,1.5,0.2],
				[4.4,2.9,1.4,0.2],
				[4.9,3.1,1.5,0.1],
				[5.4,3.7,1.5,0.2],
				[4.8,3.4,1.6,0.2],
				[4.8,3.0,1.4,0.1],
				[4.3,3.0,1.1,0.1],
				[5.8,4.0,1.2,0.2],
				[5.7,4.4,1.5,0.4],
				[5.4,3.9,1.3,0.4],
				[5.1,3.5,1.4,0.3],
				[5.7,3.8,1.7,0.3],
				[5.1,3.8,1.5,0.3],
				[5.4,3.4,1.7,0.2],
				[5.1,3.7,1.5,0.4],
				[4.6,3.6,1.0,0.2],
				[5.1,3.3,1.7,0.5],
				[4.8,3.4,1.9,0.2],
				[5.0,3.0,1.6,0.2],
				[5.0,3.4,1.6,0.4],
				[5.2,3.5,1.5,0.2],
				[5.2,3.4,1.4,0.2],
				[4.7,3.2,1.6,0.2],
				[4.8,3.1,1.6,0.2],
				[5.4,3.4,1.5,0.4],
				[5.2,4.1,1.5,0.1],
				[5.5,4.2,1.4,0.2],
				[4.9,3.1,1.5,0.1],
				[5.0,3.2,1.2,0.2],
				[5.5,3.5,1.3,0.2],
				[4.9,3.1,1.5,0.1],
				[4.4,3.0,1.3,0.2],
				[5.1,3.4,1.5,0.2],
				[5.0,3.5,1.3,0.3],
				[4.5,2.3,1.3,0.3],
				[4.4,3.2,1.3,0.2],
				[5.0,3.5,1.6,0.6],
				[5.1,3.8,1.9,0.4],
				[4.8,3.0,1.4,0.3],
				[5.1,3.8,1.6,0.2],
				[4.6,3.2,1.4,0.2],
				[5.3,3.7,1.5,0.2],
				[5.0,3.3,1.4,0.2],
				[7.0,3.2,4.7,1.4],
				[6.4,3.2,4.5,1.5],
				[6.9,3.1,4.9,1.5],
				[5.5,2.3,4.0,1.3],
				[6.5,2.8,4.6,1.5],
				[5.7,2.8,4.5,1.3],
				[6.3,3.3,4.7,1.6],
				[4.9,2.4,3.3,1.0],
				[6.6,2.9,4.6,1.3],
				[5.2,2.7,3.9,1.4],
				[5.0,2.0,3.5,1.0],
				[5.9,3.0,4.2,1.5],
				[6.0,2.2,4.0,1.0],
				[6.1,2.9,4.7,1.4],
				[5.6,2.9,3.6,1.3],
				[6.7,3.1,4.4,1.4],
				[5.6,3.0,4.5,1.5],
				[5.8,2.7,4.1,1.0],
				[6.2,2.2,4.5,1.5],
				[5.6,2.5,3.9,1.1],
				[5.9,3.2,4.8,1.8],
				[6.1,2.8,4.0,1.3],
				[6.3,2.5,4.9,1.5],
				[6.1,2.8,4.7,1.2],
				[6.4,2.9,4.3,1.3],
				[6.6,3.0,4.4,1.4],
				[6.8,2.8,4.8,1.4],
				[6.7,3.0,5.0,1.7],
				[6.0,2.9,4.5,1.5],
				[5.7,2.6,3.5,1.0],
				[5.5,2.4,3.8,1.1],
				[5.5,2.4,3.7,1.0],
				[5.8,2.7,3.9,1.2],
				[6.0,2.7,5.1,1.6],
				[5.4,3.0,4.5,1.5],
				[6.0,3.4,4.5,1.6],
				[6.7,3.1,4.7,1.5],
				[6.3,2.3,4.4,1.3],
				[5.6,3.0,4.1,1.3],
				[5.5,2.5,4.0,1.3],
				[5.5,2.6,4.4,1.2],
				[6.1,3.0,4.6,1.4],
				[5.8,2.6,4.0,1.2],
				[5.0,2.3,3.3,1.0],
				[5.6,2.7,4.2,1.3],
				[5.7,3.0,4.2,1.2],
				[5.7,2.9,4.2,1.3],
				[6.2,2.9,4.3,1.3],
				[5.1,2.5,3.0,1.1],
				[5.7,2.8,4.1,1.3]])

#this is a separate array that holds the information on the type of flower
output = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

'''
The neural object has 6 parameters
nueral(input_nodes = the amount of input nodes the network will be recieving, output_nodes = the amount of output nodes the data will be produce, hidden_nodes = the amount of hidden nodes stored in the singular hidden layer, data = the training data stored in a numpy array, learning_rate = the learning rate(a float value), output = the numpy array of the result of the training data)

'''
class neural():
	def __init__(self, input_nodes, hidden_nodes, output_nodes, data, learning_rate, output):
		#initializes the fields
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes
		self.data = data
		self.learning_rate = learning_rate
		self.output = output

		#initializes the weight matricies and the bias matrix for each layer
		self.weightsl1 = np.random.randn(input_nodes, hidden_nodes)
		self.weightsl2 = np.random.randn(hidden_nodes, output_nodes)
		self.biasl1 = np.random.randn(hidden_nodes, 1)
		self.biasl2 = np.random.randn(output_nodes, 1)

	def tanh(self, x):
		'''this is the activation function I am using, hyperbolic tangent. I use it rather than the sigmoid function because the math.exp() function always overloads haha'''
		return np.tanh(x)

	def tanh_deriv(self, x):
		'''The derivative for the hyperbolic tangent function which is to be used to find the delta'''
		return 1.0 - np.tanh(x)**2

	
	def feedforeward(self, value):
		'''
		Feedforeward takes an array of inputs, takes the dot product of the inputs with the first weight matrix, adds the transpose of the bias.
		It then multiplies that value applied to the activation function which makes the value smaller, to the dot product of the previous value by the weights and added to the transpose of the bias.
		'''
		hidden = self.tanh(np.dot(value, self.weightsl1) + self.biasl1.T)
		z = self.tanh(np.dot(hidden, self.weightsl2) + self.biasl2.T)
		return z

	'''
	The train method does most of the work.
	It takes a random input then is uses those values and applies them to the feedforeward method to get an estimate of the value.
	Then using the target value it calculates the loss with respect to each layer. Then calculates the delta based on that loss
	Using the delta and hidden layer delta values we shift the values of the weights and bias accordingly.
	By multiplying it by the derivative of the hyperbolic tangent it changes less as the value becomes more decisive.
	'''
	def train(self):
		ri = np.random.randint(len(self.data))
		point = np.array([self.data[ri]])
		z = self.feedforeward(point)
		hidden = self.tanh(np.dot(point, self.weightsl1) + self.biasl1.T)
		target = self.output[ri]

		d_cost = 2 * (z - target)
		delta = d_cost * self.tanh_deriv(z)
		hidden_layer_delta = (self.weightsl2.T * delta) * self.tanh_deriv(hidden)
		weight_grad_l2 = delta * hidden * self.learning_rate
		bias_grad_l2 = delta * self.learning_rate
		weight_grad_l1 = hidden_layer_delta.T * point * self.learning_rate
		bias_grad_l1 = hidden_layer_delta * self.learning_rate
		self.weightsl2 = self.weightsl2 - weight_grad_l2.T
		self.weightsl1 = self.weightsl1 - weight_grad_l1.T

		self.biasl2 = self.biasl2 - bias_grad_l2.T
		self.biasl1 = self.biasl1 - bias_grad_l1.T

	'''
	This method checks the output printing out the data it is testing it upon and printing the result.
	At the bottom it prints out the percentage that it was correct
	'''
	def check_output(self):
		#initialize a number to count the amount of correct values
		num = 0
		#For loop to iterate through the data but since we need the index value I use a the range function
		for i in range(len(data)):
			#prints out the value of the data then the value of the data when put through the feedforeward method which will estimate which type of flower it is
			point = data[i]
			print(point)
			z = self.feedforeward(point)
			#printing the estimation
			print("Pred {}".format(z))
			#Check to see if the data is correct
			cornum = output[i]
			onum = 0
			if(z > 0):
				onum = 1
			else:
				onum = -1
			if(onum == cornum):
				num += 1
		#print out the percentage this neural network was correct
		print(num / len(output) * 100)


n = neural(4, 5, 1, data, learning_rate, output)

for j in range(10000):
	n.train()

n.check_output()
