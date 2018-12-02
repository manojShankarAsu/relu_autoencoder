import random
import numpy as np
import sys
from load_mnist import mnist
import os
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z)) 

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
	return np.maximum(z, 0)

def reluD(x):
	x[x<=0] = 0
	x[x>0] = 1
	return x


def activation_method(x,layer):
	if layer == 1:
		return relu(x)
	elif layer == 2:
		return sigmoid(x)

def activation_derivative(x,layer):
	if layer == 1:
		return reluD(x)
	elif layer == 2:
		return sigmoid_prime(x)

def get_no_of_samples_perclass(features_100,y_train,number):
	counts = [0,0,0,0,0,0,0,0,0,0]
	count = 0
	ret  = []
	labels = []
	i = 0
	j = 0
	while count < number*10:
		curr_class = int(y_train[i])
		if counts[curr_class-1] < number:
			counts[curr_class-1] +=1
			count += 1
			ret.append( features_100[i])
			j +=1 
			labels.append(curr_class)
		i += 1
	return ret,labels


def square(z):
	return z ** 2

def one_hot(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

class NeuralNetwork(object):

	def __init__(self,sizes,activation_functions=[]):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]] 
		# between two layers , the no of bias = no of neurons in the next layer
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		# ignore the last layer. from layer 1 to n-1 , 
		# weight matrix dimension = no of next neurons (rows) * no of current layer neurons (columns)
		# size[:-1] ignores last layer. sizes[1:] starts from second layer.
		self.activation_functions = activation_functions

	def get_hidden_layer_activation(self,a):
		activation = activation_method(np.dot(self.weights[0],a)+self.biases[0] , 1)
		return activation

	def feedforward(self, a):
		layer = 1
		for b,w in zip(self.biases,self.weights):
			a = activation_method(np.dot(w,a) + b , layer)
			layer = layer + 1
		return a

	def BatchGD(self,training_data,epochs,learning_rate,test_data=None):
		if test_data:
			n_test = len(test_data)
		n = len(training_data)
		for j in xrange(epochs):
			random.shuffle(training_data)
			self.update_batch(training_data,learning_rate)
			print "Epoch {0} complete".format(j)

	def SGD(self,training_data,epochs,mini_batch_size,learning_rate,test_data=None):
		if test_data:
			n_test = len(test_data)
		n = len(training_data)
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_batch(mini_batch,learning_rate)
			if j % 3 == 0:
				cost = self.cost(training_data)
				print 'cost {0}'.format(cost)
			if test_data:
				print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
			else:
				print "Epoch {0} complete".format(j)
	
	def cost(self,training_data):
		costt = 0
		for x,y in training_data:
			prediction = self.feedforward(x)
			diff = prediction - y
			diff = square(diff)
			costt += np.sum(diff)
		return costt / len(training_data)

	def update_batch(self,mini_batch,learning_rate):
		total_gradient_b = [np.zeros(b.shape) for b in self.biases]
		total_gradient_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_b, delta_w = self.backprop(x,y)
			total_gradient_b = [ t_g_b + db for t_g_b,db in zip(total_gradient_b,delta_b)]
			total_gradient_w = [ t_g_w + dw for t_g_w, dw in zip(total_gradient_w,delta_w)]
		self.weights = [w-(learning_rate/len(mini_batch))*gw for w, gw in zip(self.weights,total_gradient_w)]
		self.biases = [b- (learning_rate/len(mini_batch))*gb for b, gb in zip(self.biases,total_gradient_b)]

	def backprop(self,x,y):
		gradient_b = [np.zeros(b.shape) for b in self.biases]
		gradient_w = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]
		zs = []

		# forward propogation
		layer = 1
		for b,w in zip(self.biases,self.weights):
			z = np.dot(w,activation) + b
			zs.append(z)
			activation = activation_method(z,layer)
			activations.append(activation)
			layer += 1

		# backward propogation
		del_l = self.derivative_cost_wrt_aL(activations[-1],y) * sigmoid_prime(zs[-1])
		gradient_b[-1] = del_l
		gradient_w[-1] = np.dot(del_l,np.array(activations[-2]).transpose())

		for l in xrange(2,self.num_layers):
			z_l = zs[-l]
			del_l = np.dot(np.array(self.weights[-l+1]).T , del_l) * reluD(z_l)
			gradient_b[-l]  = del_l
			gradient_w[-l] = np.dot(del_l, np.array(activations[-l-1]).T)

		return (gradient_b , gradient_w)



	def derivative_cost_wrt_aL(self,last_layer_activation,y):
		return (last_layer_activation - y)

	def evaluate(self,test_data):
		test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
		return sum(int(x == y) for (x,y) in test_results)


def main():

	# Stacked Auto encoder 1
	stacked1 = NeuralNetwork([784,500,784]) # map(int,input.strip('[]').split(','))
	data_dir = os.getcwd()
	fashion_data = os.path.join(data_dir,'fashion_mnist')
	no_training = 100
	train_data, train_label, test_data, test_label = mnist(noTrSamples=no_training,noTsSamples=100,\
            digit_range=[0,1,2,3,4,5,6,7,8,9],\
            noTrPerClass=no_training/10, noTsPerClass=10)
	no_of_images = train_data.shape[1]
	train_label_backup = train_label
	
	
	train_images = [  np.reshape(train_data[:,i] , (784,1)) for i in xrange(no_of_images)]
	train_label = [  np.reshape(train_data[:,i] , (784,1)) for i in xrange(no_of_images)]
	new_train_data = zip(train_images,train_label)
	
	# train_images[0] - (784,1)
	# Train Stacked Auto encoder 1
	stacked1.SGD(new_train_data,500,10,0.3,test_data=None)

	# Freeze and get the 500 features
	features_500 = [stacked1.get_hidden_layer_activation(tr_data[0]) for tr_data in new_train_data]
	print(len(features_500))
	print(features_500[0].shape)
	# list of  500,1 elements

	# Train Stacked Auto encoder 2
	stacked2 = NeuralNetwork([500,200,500])
	train_data_2 = zip(features_500,features_500)
	stacked2.SGD(train_data_2,500,10,0.3,test_data=None)

	# Freeze and get the 200 features
	features_200 = [stacked2.get_hidden_layer_activation(a[0]) for a in train_data_2]
	# list of 200,1

	# Train Stacked Auto encoder 3
	stacked3 = NeuralNetwork([200,100,200])
	train_data_3 = zip(features_200,features_200)
	stacked3.SGD(train_data_3,500,10,0.3,test_data=None)

	# Freeze and get the 100 features
	features_100 = [stacked3.get_hidden_layer_activation(a[0]) for a in train_data_3]
	#list of  100,1

	# # classify
	# Freeze the Stacked auto encoder layers and train the classifier
	one_perclass,labels = get_no_of_samples_perclass(features_100,train_label_backup[0],1)
	classifier1_onesample_perclass = NeuralNetwork([100,10],activation_functions=['softmax'])
	label_one_hots = [ one_hot(l) for l in labels]
	train_data_4 = zip(one_perclass,label_one_hots)
	classifier1_onesample_perclass.SGD(train_data_4,500,10,0.5,test_data=None)

	ten_perclass,labels = get_no_of_samples_perclass(features_100,train_label_backup[0],10)
	classifier2_tensample_perclass = NeuralNetwork([100,10],activation_functions=['softmax'])
	label_one_hots = [ one_hot(l) for l in labels]
	train_data_4 = zip(one_perclass,label_one_hots)
	classifier2_tensample_perclass.SGD(train_data_4,500,10,0.5,test_data=None)

	


if __name__ == '__main__':
	main()