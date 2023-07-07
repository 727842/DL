from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
models = keras.Sequential()
models.add(Dense(4, input_dim=2, activation='relu'))
models.add(Dense(4, activation='relu'))
models.add(Dense(1, activation='sigmoid'))
models.compile(loss='binary_crossentropy', optimizer='adam')
models.fit(X, Y, epochs=500)
import numpy as np

Xnew, Yreal = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)
Yclass = np.argmax(models.predict(Xnew), axis=-1)
Ynew = models.predict(Xnew)
for i in range(len(Xnew)): print("X=%s,Predicted_probability=%s,Predicted_class=%s" % (Xnew[i], Ynew[i], Yclass[i]))
