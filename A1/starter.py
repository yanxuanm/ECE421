import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    pass

def MSE(W, b, x, y, reg):
    pass


def gradMSE(W, b, x, y, reg):
   pass

def crossEntropyLoss(W, b, x, y, reg):
    pass

def gradCE(W, b, x, y, reg):
    pass

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    pass

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    pass

if __name__ == '__main__':
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
    validData = validData.reshape((-1,validData.shape[1]*validData.shape[2]))
    testData = testData.reshape((-1,testData.shape[1]*testData.shape[2]))
    W = np.random.randn((trainData.shape[1]),1)
    b = np.random.randn(1,1)

    W2 = np.random.randn((trainData.shape[1]), 1)
    b2 = np.random.randn(1, 1)

    W3 = np.random.randn((trainData.shape[1]), 1)
    b3 = np.random.randn(1, 1)
    
    alpha = 0.005
    alpha2 = 0.001
    alpha3 = 0.0001
    iterations = 5000
    reg = 0
    EPS = 1e-7
    W, b, cost = grad_descent(W, b, trainData, trainTarget, alpha, iterations, reg, EPS,"MSE")
    W2, b2, cost2 = grad_descent(W2, b2, trainData, trainTarget, alpha2, iterations, reg, EPS, "MSE")
    W3, b3, cost3 = grad_descent(W3, b3, trainData, trainTarget, alpha3, iterations, reg, EPS, "MSE")

    plt.plot(range(len(cost)), cost, c = "r", label = "cost_history_005")
    plt.plot(range(len(cost2)), cost2, c="g", label="cost_history_001")
    plt.plot(range(len(cost3)), cost3, c="b", label="cost_history_0001")
    plt.legend(loc = "best")
    plt.show()

