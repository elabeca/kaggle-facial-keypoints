import matplotlib.pyplot as pyplot
from kfkd import load2d, plot_sample

X, y = load2d()
X_flipped = X[:, :, :, ::-1] # simple slice to flip all images

# plot two images
fig = pyplot.figure(figsize=(6, 3))

ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plot_sample(X[1], y[1], ax)

ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plot_sample(X_flipped[1], y[1], ax)

pyplot.show()