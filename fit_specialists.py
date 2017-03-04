from collections import OrderedDict
from sklearn.base import clone
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

SPECIALIST_SETTINGS = [
		dict(
      columns=(
        'left_eye_center_x', 'left_eye_center_y',
        'right_eye_center_x', 'right_eye_center_y'
        ),
      flip_indices=((0, 2), (1, 3))
      ),

    dict(
      columns=(
        'nose_tip_x', 'nose_tip_y'
        ),
      flip_indices=()
      ),

    dict(
      columns=(
        'mouth_left_corner_x', 'mouth_left_corner_y',
        'mouth_right_corner_x', 'mouth_right_corner_y',
        'mouth_center_top_lip_x', 'mouth_center_top_lip_y'
        ),
      flip_indices=((0, 2), (1, 3))
      ),

    dict(
      columns=(
        'mouth_center_bottom_lip_x',
        'mouth_center_bottom_lip_y'
        ),
      flip_indices=()
      ),

    dict(
      columns=(
        'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
        'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
        'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
        'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        ),
      flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
      ),

    dict(
      columns=(
        'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
        'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
        'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
        'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        ),
      flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
      )
    ]

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

def fit_specialists(fname_pretrain=None):
  if fname_pretrain:
    with open(fname_pretrain, 'rb') as f:
      net_pretrain = pickle.load(f)
  else:
    net_pretrain = None

  specialists = OrderedDict()

  for setting in SPECIALIST_SETTINGS:
    cols = setting['columns']
    X, y = load2d(cols=cols)

    model = clone(net8)
    model.output_num_units = y.shape[1]
    model.batch_iterator_train.flip_indices = setting['flip_indices']
    # set number of epochs relative to number of training examples:
    model.max_epochs = int(1e7 / y.shape[0])
    
    if 'kwargs' in setting:
      # an option 'kwargs' in the settings list may be used to
      # set any other parameter of the net:
      vars(model).update(setting['kwargs'])

    if net_pretrain is not None:
      # if a pretrain model was given, use it to initialize the
      # weights of our new specialist model:
      model.load_params_from(net_pretrain)

    print("Training model for columns {} for {} epochs".format(cols,
    model.max_epochs))
    model.fit(X, y)
    specialists[cols] = model

  from pickle import dump
  with open("fit-specialists_{0}.pickle".format(sys.argv[1]), 'wb') as f:
    # this time we're persisting a dictionary with all models:
    dump(specialists, f, -1)

fit_specialists()
