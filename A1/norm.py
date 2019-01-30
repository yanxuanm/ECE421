import numpy as np
import matplotlib.pyplot as plt
import time


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
    # transpose_W = np.transpose(W)
    error = np.matmul(x,W) + b - y
    mse = (np.sum(error*error))/((2*np.shape(y)[0])) + reg/2*np.sum(W*W)
    # print(mse)
    return mse


def norm_equation(x, y):
    x_b = np.ones((np.shape(x)[0],1))
    xx = np.hstack((x,x_b))
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(xx), xx)), np.transpose(xx)), y)
    print(W.shape)
    return W[:-1,:],W[-1][0]


def MSE_norm(W, b, x, y):
    # Your implementation here
    # transpose_W = np.transpose(W)
    error = np.matmul(x,W) + b - y
    mse = (np.sum(error*error))/((2*np.shape(y)[0]))
    # print(mse)
    return mse

if __name__ == '__main__':
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
    validData = validData.reshape((-1,validData.shape[1]*validData.shape[2])) 
    testData = testData.reshape((-1,testData.shape[1]*testData.shape[2]))
    time_start = time.clock()

    W = np.random.randn(trainData.shape[1],1)
    alpha = 0.005
    iterations = 5000
    reg = 0
    EPS = 1e-7
    # print(trainData.shape,trainTarget.shape,W.shape,testData.shape,validData.shape)
    b = np.random.randn(1, 1)
    W, b = norm_equation(trainData, trainTarget)
    # plt.imshow(W.reshape((28,28)))
    
    # plt.show()
    out = np.matmul(trainData,W)+b
    # for (x,y) in zip(out[:20],trainTarget[:20]):
    # 	print(x,y)
    print(np.sum((out>=0.5)==trainTarget))
    print("Training data accuracy: ", np.sum((out>=0.5)==trainTarget)/(trainData.shape[0]))

    out_valid = np.matmul(validData,W)+b
    print(np.sum((out_valid>=0.5)==validTarget))
    print("Valid data accuracy: ", np.sum((out_valid>=0.5)==validTarget)/(validData.shape[0]))

    out_test = np.matmul(testData,W)+b
    print(np.sum((out_test>=0.5)==testTarget))
    print("Test data accuracy: ", np.sum((out_test>=0.5)==testTarget)/(testData.shape[0]))

    time_elapsed = (time.clock() - time_start)

    print(time_elapsed)

    print(MSE(W, b, trainData, trainTarget, reg))
