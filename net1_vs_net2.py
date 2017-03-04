import matplotlib.pyplot as pyplot
from kfkd import load, load2d, plot_loss, plot_sample
from net1 import net1
from net2 import net2

# X1, y1 = load()
# net1.fit(X1, y1)

# X2, y2 = load2d() # load 2D data
# net2.fit(X2, y2)

sample1 = load(test=True)[0][6:7]
sample2 = load2d(test=True)[0][6:7]
y_pred1 = net1.predict(sample1)[0]
y_pred2 = net2.predict(sample2)[0]

fig = pyplot.figure(figsize=(6, 3))

ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plot_sample(sample1[0], y_pred1, ax)

ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plot_sample(sample1[0], y_pred2, ax)

pyplot.show()