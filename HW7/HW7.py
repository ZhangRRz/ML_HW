import tensorflow.keras as keras  # use tensorflow 2.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn import datasets


def lin(x):
    return x**3+2*x**2-3*x-1


x_train = np.random.uniform(-15, 15, size=80000)
y_train = np.array([lin(x) for x in x_train])
x_test = sorted(np.random.uniform(-15, 15, size=20))
y_test = np.array([lin(x) for x in x_test])

model1 = tf.keras.models.Sequential()
model1.add(Dense(100, input_shape=(1,)))
model1.add(Activation('relu'))
model1.add(Dense(1))

model1.compile(loss='mse', optimizer="adam")
model1.fit(x_train, y_train, epochs=5)
y_pred1 = model1.predict(x_test)

model2 = tf.keras.models.Sequential()
model2.add(Dense(50, input_shape=(1,)))
model2.add(Activation('relu'))
model2.add(Dense(100))
model2.add(Activation('relu'))
model2.add(Dense(200))
model2.add(Activation('relu'))
model2.add(Dense(150))
model2.add(Activation('relu'))
model2.add(Dense(50))
model2.add(Activation('relu'))
model2.add(Dense(1))

model2.compile(loss='mse', optimizer="adam")
model2.fit(x_train, y_train, epochs=10)
y_pred2 = model2.predict(sorted(x_test))

plt.plot(x_test, y_pred1, marker='o')
plt.plot(x_test, y_pred2, marker='^')
plt.plot(x_test, y_test, marker='.')
plt.show()
# ====================================================================

circles_data, circles_data_labels = datasets.make_circles(
    n_samples=50, factor=0.1, noise=0.1)
