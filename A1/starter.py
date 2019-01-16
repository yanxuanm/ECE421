import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    # Your implementation here
    transpose_W = np.transpose(W)
    error = np.matmul(transpose_W, x) + b - y
    mse = (np.sum(np.square(error))/(2*np.shape(y)[0])) + reg*np.matmul(W, W)
    return mse


def gradMSE(W, b, x, y, reg):
    # Your implementation here
    transpose_W = np.transpose(W)
    error = np.matmul(transpose_W, x) + b - y
    grad_mse_W = (np.sum(np.transpose(error)* x))/(np.shape(y)[0]) + 2 * reg * W
    grad_mse_d = (np.sum(np.transpose(error)))/(np.shape(y)[0]) 
    return grad_mse_W, grad_mse_d

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here

def gradCE(W, b, x, y, reg):
    # Your implementation here

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here

