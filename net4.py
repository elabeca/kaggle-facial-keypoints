import matplotlib.pyplot as pyplot
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from kfkd import load2d, plot_loss, float32
# from flip_batch_iterator import FlipBatchIterator
from adjust_variable import AdjustVariable
import theano
import sys

sys.setrecursionlimit(100000)

net4 = NeuralNet(
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

  	update_learning_rate=theano.shared(float32(0.03)),
  	update_momentum=theano.shared(float32(0.9)),

  	regression=True,
  	#batch_iterator_train=FlipBatchIterator(batch_size=128),
  	on_epoch_finished=[
      AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
      AdjustVariable('update_momentum', start=0.9, stop=0.999)
      ],
  	max_epochs=3000,
  	verbose=1
  	)

X, y = load2d() # load 2D data
net4.fit(X, y)

# Training for 3000 epochs will take a while. We'll pickle the
# trained model so that we can load it back later:

from pickle import dump
with open("net4_{0}.pickle".format(sys.argv[1]), 'wb') as f:f:
	dump(net4, f, -1)