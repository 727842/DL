from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
dataset = loadtxt('diabetes.csv',delimiter= ',',skiprows = 1)
X = dataset[:,0:8]
Y = dataset[:,8]
print(X)
print(Y)
model =Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,input_dim=8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,Y,batch_size=12,epochs=500)
Accuracy=model.evaluate(X,Y)
print('Accuracy of model is ',(Accuracy*100))
prediction=model.predict(X)
exec('for i in range(5):print(X[i].tolist(),prediction[i],Y[i])')
