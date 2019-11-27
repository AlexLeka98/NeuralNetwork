import numpy as np
from mnist import MNIST
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import math

path = "{}".format(os.getcwd())

mndata = MNIST(path)

Xtrain, ytrain = mndata.load_training()
Xtest, ytest = mndata.load_testing()

Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)
Xtest = np.array(Xtest)
ytest = np.array(ytest)

print(len(Xtrain[0]),len(Xtrain[1]))


Xtrain, ytrain = Xtrain[:2000], ytrain[:2000]
Xtest, ytest = Xtest[:1000], ytest[:1000]

ytrain = keras.utils.to_categorical(ytrain, num_classes=10)
ytest = keras.utils.to_categorical(ytest, num_classes=10)

model = Sequential()

model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
adam = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(Xtrain, ytrain, epochs=50, batch_size=256, verbose=2)

print("Train accuracy: ", model.evaluate(Xtrain, ytrain, batch_size=128))
print("Test accuracy: ", model.evaluate(Xtest, ytest, batch_size=128))
