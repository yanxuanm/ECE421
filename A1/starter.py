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
    # transpose_W = np.transpose(W)
    error = np.matmul(x,W) + b - y
    mse = (np.sum(error*error)/(2*np.shape(y)[0])) + reg/2*np.sum(W*W)
    return mse


def gradMSE(W, b, x, y, reg):
    # Your implementation here
    # transpose_W = np.transpose(W)
    error = np.matmul(x,W) + b - y
    grad_mse_W = np.matmul(np.transpose(x),error)/(np.shape(y)[0]) + 2*reg*W
    # grad_mse_W = (np.sum(np.transpose(error)* x))/(np.shape(y)[0])# + 2 * reg * W
    # print(grad_mse_W.shape)
    grad_mse_b = (np.sum(error))/(np.shape(y)[0])
    return grad_mse_W, grad_mse_b

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    return 

def gradCE(W, b, x, y, reg):
    # Your implementation here
    return 

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS,
                 validData, testData, validTarget, testTarget):
    # Your implementation here
    train_loss = [MSE(W,b,trainingData,trainingLabels,reg)]
    valid_loss = [MSE(W, b, validData, validTarget, reg)]
    test_loss = [MSE(W, b, testData, testTarget, reg)]
    for i in range(iterations):
        grad_mse_W, grad_mse_b = gradMSE(W, b, trainingData, trainingLabels, reg)
        new_W = W - alpha * grad_mse_W
        new_b = b - alpha * grad_mse_b
        train_loss.append(MSE(new_W,new_b,trainingData,trainingLabels,reg))
        valid_loss.append(MSE(new_W,new_b,validData,validTarget,reg))
        test_loss.append(MSE(new_W,new_b,testData,testTarget,reg))
        mag = np.linalg.norm(new_W-W)
        if mag<EPS:
            return new_W,new_b, train_loss, valid_loss, test_loss
        else:
            W = new_W
            b = new_b
    return W,b,train_loss, valid_loss, test_loss


# def iter_cases(validData, testData, validTarget, testTarget, W, b, reg, iterations):
#     valid_loss = [MSE(W, b, validData, validTarget, reg)]
#     test_loss = [MES(W, b, testData, testTarget, reg)]
#     for i in range(iterations):
#         valid_loss.append(MSE(new_W,new_b,trainingData,trainingLabels,reg))




def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    return 

if __name__ == '__main__':
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
    validData = validData.reshape((-1,validData.shape[1]*validData.shape[2])) 
    testData = testData.reshape((-1,testData.shape[1]*testData.shape[2]))
    W = np.random.random_sample(trainData.shape[1],1)
    print(trainData.shape,trainTarget.shape,W.shape,testData.shape,validData.shape)
    b = np.random(size=None)
    alpha = 0.0001
    iterations = 5000
    reg = 0
    EPS = 1e-4
    W, b, train_loss, valid_loss, test_loss = grad_descent(W, b, trainData, trainTarget, 
                    alpha, iterations, reg, EPS, validData, testData, validTarget, testTarget)

    out = np.matmul(trainData,W)+b
    print(np.sum((out>=0.5)==trainTarget))
    print("Training data accuracy: ", np.sum((out>=0.5)==trainTarget)/(trainData.shape[0]))

    out_valid = np.matmul(validData,W)+b
    print(np.sum((out_valid>=0.5)==validTarget))
    print("Valid data accuracy: ", np.sum((out_valid>=0.5)==validTarget)/(validData.shape[0]))

    out_test = np.matmul(testData,W)+b
    print(np.sum((out_test>=0.5)==testTarget))
    print("Test data accuracy: ", np.sum((out_test>=0.5)==testTarget)/(testData.shape[0]))

    iterations = range(len(train_loss))
    plt.plot(iterations,train_loss)
    plt.plot(iterations,valid_loss)
    plt.plot(iterations,test_loss)
    plt.suptitle('Alpha = 0.005, lambda = 0.001', fontsize=16)
    plt.legend(['train_loss', 'valid_loss', 'test_loss'], loc='upper right')
    plt.show()
