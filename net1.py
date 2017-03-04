import matplotlib.pyplot as pyplot
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from kfkd import load, plot_loss, plot_sample

net1 = NeuralNet(
	layers=[	# three layers: one hidden layer
		('input', layers.InputLayer),
		('hidden', layers.DenseLayer),
		('output', layers.DenseLayer)
		],
	# layer parameters:
	input_shape=(None, 9216),	# 96x96 input pixels per batch
	hidden_num_units=100,		# number of units in hidden layer
	output_nonlinearity=None,	# output layer uses identity function
	output_num_units=30,		# 30 target values

	# optimization method:
	update=nesterov_momentum,
	update_learning_rate=0.01,
	update_momentum=0.9,

	regression=True,	# flag indicate we're dealing with a regression problem
	max_epochs=400,
	verbose=1
	)

def show_sample():
	X, _ = load(test=True)
	y_pred = net1.predict(X)

	fig = pyplot.figure(figsize=(6, 6))
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in range(16):
		ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
		plot_sample(X[i], y_pred[i], ax)

	pyplot.show()


X, y = load()
net1.fit(X, y)
#plot_loss(net1)
#show_sample()


