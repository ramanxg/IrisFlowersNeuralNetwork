from matrix import Matrix
from math import exp

class NeuralNetwork:
	def __init__(self, input_nodes, hidden_nodes, output_nodes):
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes
		#create random weights matrix 
		self.weights_ih = Matrix.zeros(self.hidden_nodes, self.input_nodes)
		self.weights_ho = Matrix.zeros(self.output_nodes, self.hidden_nodes)
		self.weights_ho.randomize()
		self.weights_ih.randomize()
		#create random bias matrix
		self.bias_h = Matrix.zeros(self.hidden_nodes, 1)
		self.bias_o = Matrix.zeros(self.output_nodes, 1)
		self.bias_h.randomize()
		self.bias_o.randomize()
		self.learning_rate = 0.1

	def feedforward(self, input_list):
		#convert input list into a vector
		inputs = Matrix.vector(input_list)
		#input to hidden
		hidden = self.weights_ih * inputs
		hidden = hidden + self.bias_h
		#activation function
		hidden = hidden.elementmap(NeuralNetwork.sigmoid)
		#hidden to output
		output = self.weights_ho * hidden
		output = output + self.bias_o
		#activation function
		output = output.elementmap(NeuralNetwork.sigmoid)

		return output

	@staticmethod
	def sigmoid(x):
		return 1 / (1 + exp(-x))

	@staticmethod
	def dsigmoid(y):
		#return sigmoid(x) * (1-sigmoid(x))
		return y * (1 - y)

	def train(self, input_list, target_array):

		### FEEDFORWARD ###

		#convert input list into a vector
		inputs = Matrix.vector(input_list)
		#input to hidden
		hidden = self.weights_ih * inputs
		hidden = hidden + self.bias_h
		#activation function
		hidden = hidden.elementmap(NeuralNetwork.sigmoid)
		#hidden to output
		outputs = self.weights_ho * hidden
		outputs = outputs + self.bias_o
		#activation function
		outputs = outputs.elementmap(NeuralNetwork.sigmoid)
		
		#convert targets list into a vector
		targets = Matrix.vector(target_array)


		### BACKPROPAGATION ###

		#Calculate error: error = expected - predicted
		output_error = targets - outputs

		#gradient = output * (1 - output)
		gradients = outputs.elementmap(NeuralNetwork.dsigmoid)
		gradients = gradients.elementmul(output_error)
		gradients = gradients * self.learning_rate
		

		#calculate deltas
		hidden_t = Matrix.transpose(hidden)
		weight_ho_deltas = gradients * hidden_t 

		#Adjust weights by deltas
		self.weights_ho = self.weights_ho + weight_ho_deltas
		#adjust bias by deltas
		self.bias_o = self.bias_o + gradients

		#with more layers, use loop to take this through each layer
		weights_ho = Matrix.transpose(self.weights_ho)
		hidden_errors = weights_ho * output_error

		#calculate hidden gradient
		hidden_grad = hidden.elementmap(NeuralNetwork.dsigmoid)
		hidden_grad = hidden_grad.elementmul(hidden_errors)
		hidden_grad = hidden_grad * self.learning_rate

		#calculate input-to-hidden deltas
		inputs_t = Matrix.transpose(inputs)
		weight_ih_deltas = hidden_grad * inputs_t

		self.weights_ih = self.weights_ih + weight_ih_deltas
		self.bias_h = self.bias_h + hidden_grad\

		