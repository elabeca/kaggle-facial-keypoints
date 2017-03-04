import matplotlib.pyplot as pyplot
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from kfkd import load2d, plot_loss, float32
from flip_batch_iterator import FlipBatchIterator
from adjust_variable import AdjustVariable
from early_stopping import EarlyStopping
import theano
import sys

sys.setrecursionlimit(100000)

net8 = NeuralNet(
    layers=[
      ('input', layers.InputLayer),
      ('conv1', layers.Conv2DLayer),
      ('pool1', layers.MaxPool2DLayer),
      ('dropout1', layers.DropoutLayer),
      ('conv2', layers.Conv2DLayer),
      ('pool2', layers.MaxPool2DLayer),
      ('dropout2', layers.DropoutLayer),
      ('conv3', layers.Conv2DLayer),
      ('pool3', layers.MaxPool2DLayer),
      ('dropout3', layers.DropoutLayer),
      ('hidden4', layers.DenseLayer),
      ('dropout4', layers.DropoutLayer),
      ('hidden5', layers.DenseLayer),
      ('output', layers.DenseLayer)
      ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,
    hidden4_num_units=1000,
    dropout4_p=0.5,
    hidden5_num_units=1000,
    output_num_units=30,
    output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
      AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
      AdjustVariable('update_momentum', start=0.9, stop=0.999),
      EarlyStopping(patience=200)
      ],
    max_epochs=10000,
    verbose=1
    )

X, y = load2d() # load 2D data
net8.fit(X, y)

# Training for 10000 epochs will take a while. We'll pickle the
# trained model so that we can load it back later:

from pickle import dump
with open("net8_{0}.pickle".format(sys.argv[1]), 'wb') as f:
  dump(net8, f, -1)