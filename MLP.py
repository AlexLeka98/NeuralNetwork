import numpy as np
from mnist import MNIST
import os
import math
import random


class Perceptron():
    def __init__(self,activation,input_dim):
        self.activation = activation
        self.input_dim = input_dim
        self.bias = random.randrange(0,100) / 100
    def actFun(self,u):
        if (self.activation == "relu"):
            if (u<0):
                return 0
            return u
        elif(self.activation == "sigmoid"):
            return  1  /  ( 1+math.exp(-u) )
        elif(self.activation == "tahn"):
            return 2* (  1/(1+math.exp(-u))  ) -1

    def getStimuli(self,sample):
        sum = -bias
        for i in range(0,input_dim):
            sum = sum + sample[i]
        return actFun(sum)
    def getInput_dim(self):
        return self.input_dim
    def getBias(self):
        return self.bias



#kai  37 plhntirio
class Layer():
    def __init__(self,numberOfPerceptorns,activation,input_dim):
        self.numPercep = numberOfPerceptorns    #Number of perceptrons in this layer
        self.activation = activation    #Activation function used from the perceptrons
        """self.input_dim = input_dim      #how many outputs does the previous layer have"""
        self.layer = list()             #List with all the perceptrons in this layer
        self.weightVector = np.random.rand(input_dim,1)     #All the weights in this layer
        for i in range(0,numberOfPerceptorns):                      #Initializing each perceptron
            self.layer.append(Perceptron(activation,input_dim))

    def stimuli(self,sample):
        newX = np.empty(self.numPercep)                     #Output of each Perceptron will be stored here, so that it can be used as an input for the next layer
        u = np.dot(sample,self.weightVector)                #Calculating the dot product of the sample vector and the weight vector
        for i in range(0,len(self.layer)):
            newX[i] = self.layer[i].actFun(u - self.layer[i].getBias())     #Substructing the bias from each perceptron and passing that value from the activation function. The result is stored at newX[]
            #print("i)  ",newX[i],"    u: ",u,"     bias: ",self.layer[i].getBias())
        return newX

    def getNumPercep(self):
        return self.numPercep
    """def getInputDim(self):
        return self.input_dim"""
    def getWeightVector(self):
        return self.weightVector


class NeuralNetwork():
    def __init__(self):
        self.networkLayers = list()

    def addLayer(self,numberOfPerceptrons,activation, input_dim = 0 ):
        if input_dim == 0:
            input_dim = self.networkLayers[len(self.networkLayers)-1].getNumPercep()
        self.networkLayers.append(Layer(numberOfPerceptrons,activation,input_dim))

    def train(self,trainingSamples,trainingLabels,epochs=0,batch_size=0,verbose=0):
        vector = trainingSamples
        for i in range (0,len(trainingSamples)):
            x = trainingSamples[i]/255.0
            y = np.zeros(10)
            y[trainingLabels[i]] = 1.0
            for e in self.networkLayers:                                                        #inputDimentions = weightVector    ,    OutputDimentrion = Number of Perceptrons
                x = e.stimuli(x)
            print("this is the final x :  ",x,"\n")
            #print("And this is the y",y)


    def evaluate(self,Xtrain,ytrain,batch_size):
        pass

    def getWholeNetwork(self):
        return self.networkLayers


if __name__ == "__main__" :
    path = "{}".format(os.getcwd())
    mndata = MNIST(path)

    Xtrain, ytrain = mndata.load_training()
    Xtest, ytest = mndata.load_testing()

    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)

    Xtrain, ytrain = Xtrain[0:100], ytrain[0:100]
    #Xtest, ytest = Xtest[:50], ytest[:50]
    myNN = NeuralNetwork()
    myNN.addLayer(10,'relu',784)
    myNN.addLayer(60,'sigmoid')
    myNN.addLayer(50,'sigmoid')
    myNN.addLayer(10,'sigmoid')
    myNN.train(Xtrain,ytrain)
