import matplotlib.pyplot as pyplot
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from kfkd import load2d, plot_loss
import sys

sys.setrecursionlimit(100000)

net2 = NeuralNet(
		layers=[
			('input', layers.InputLayer),
			('conv1', layers.Conv2DLayer),
			('pool1', layers.MaxPool2DLayer),
			('conv2', layers.Conv2DLayer),
			('pool2', layers.MaxPool2DLayer),
			('conv3', layers.Conv2DLayer),
			('pool3', layers.MaxPool2DLayer),
			('hidden4', layers.DenseLayer),
			('hidden5', layers.DenseLayer),
			('output', layers.DenseLayer)
			],
		input_shape=(None, 1, 96, 96),
		conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
		conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
		conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
		hidden4_num_units=500,
		hidden5_num_units=500,
		output_num_units=30,
		output_nonlinearity=None,

		update_learning_rate=0.01,
		update_momentum=0.9,

		regression=True,
		max_epochs=1000,
		verbose=1
		)

X, y = load2d() # load 2D data
net2.fit(X, y)

# Training for 1000 epochs will take a while. We'll pickle the
# trained model so that we can load it back later:

from pickle import dump
with open("net2_{0}.pickle".format(sys.argv[1]), 'wb') as f:
	dump(net2, f, -1)
