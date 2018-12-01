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


def square(z):
	return z ** 2

class NeuralNetwork(object):

	def __init__(self,sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]] 
		# between two layers , the no of bias = no of neurons in the next layer
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		# ignore the last layer. from layer 1 to n-1 , 
		# weight matrix dimension = no of next neurons (rows) * no of current layer neurons (columns)
		# size[:-1] ignores last layer. sizes[1:] starts from second layer.

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
	dimension_string = sys.argv[1]
	dimensions = map(int,dimension_string.strip('[]').split(','))
	net = NeuralNetwork(dimensions) # map(int,input.strip('[]').split(','))
	data_dir = os.getcwd()
	fashion_data = os.path.join(data_dir,'fashion_mnist')
	no_training = 2000
	train_data, train_label, test_data, test_label = mnist(noTrSamples=no_training,noTsSamples=1000,\
            digit_range=[0,1,2,3,4,5,6,7,8,9],\
            noTrPerClass=no_training/10, noTsPerClass=100)
	no_of_images = train_data.shape[1]
	mu, sigma = 0.1, 0.2
	# row1=train_data[:,0]
	# row1_mat=row1.reshape(28,28)
	# plt.matshow(row1_mat)
	# plt.show()
	noise = np.random.normal(mu, sigma, [784, no_training])
	noised_train_data=train_data+noise
	# noiserow1=noised_train_data[:,0]
	# noiserow1=noiserow1.reshape(28,28)
	# plt.matshow(noiserow1)
	# plt.show()
	train_images = [  np.reshape(noised_train_data[:,i] , (784,1)) for i in xrange(no_of_images)]
	train_label = [  np.reshape(train_data[:,i] , (784,1)) for i in xrange(no_of_images)]
	new_train_data = zip(train_images,train_label)
	# print 'Train Noisy image'
	# noiserow1=new_train_data[0][0]
	# noiserow1=noiserow1.reshape(28,28)
	# plt.matshow(noiserow1)
	# plt.show()
	# print 'Original image'
	# noiserow1=new_train_data[0][1]
	# noiserow1=noiserow1.reshape(28,28)
	# plt.matshow(noiserow1)
	# plt.show()

	net.SGD(new_train_data,30,10,0.05,test_data=None)
	print 'Predicted Data'
	pred = net.feedforward(new_train_data[0][0])
	pred=pred.reshape(28,28)
	plt.matshow(pred)
	plt.show()

	print 'Original data'
	noiserow1=new_train_data[0][1]
	noiserow1=noiserow1.reshape(28,28)
	plt.matshow(noiserow1)
	plt.show()
	weight_filename = 'weight_{0}'
	bias_filename = 'bias_{0}'
	i = 1
	for weight in net.weights:
		np.savetxt(weight_filename.format(i),weight)
		i+=1
	i = 1
	for bias in net.biases:
		np.savetxt(bias_filename.format(i),bias)
		i+=1






if __name__ == '__main__':
	main()