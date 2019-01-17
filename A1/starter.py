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
    mse = (np.sum(error*error))/((2*np.shape(y)[0])) + reg/2*np.sum(W*W)
    print(mse)
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
    y_hat = 1.0/(1.0+np.exp(-(np.matmul(x,W)+b)))

    cross_entropy_loss = (np.sum(-(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))))/(np.shape(y)[0]) + reg/2*np.sum(W*W)
    print(cross_entropy_loss)
    return cross_entropy_loss

def gradCE(W, b, x, y, reg):
    # Your implementation here
    y_hat = 1.0/(1.0+np.exp(-(np.matmul(x,W)+b)))
    # der_y_hat = (-y)/(y_hat) + (1-y)/(1-y_hat)
    # der_z = der_y_hat*y_hat*(1-y_hat)
    # der_w =  np.matmul(np.transpose(x), der_z)/(np.shape(y)[0]) + 2*reg*W
    der_w =  np.matmul(np.transpose(x), (y_hat - y))/(np.shape(y)[0]) + 2*reg*W
    return der_w, np.sum((y_hat - y))/(np.shape(y)[0])

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS,
                 validData, testData, validTarget, testTarget, lossType = "None"):
    # Your implementation here
    if lossType == "None":
        train_loss = [MSE(W,b,trainingData,trainingLabels,reg)]
        valid_loss = [MSE(W, b, validData, validTarget, reg)]
        test_loss = [MSE(W, b, testData, testTarget, reg)]
        out_train = np.matmul(trainData,W)+b
        train_accur = [np.sum((out_train>=0.5)==trainTarget)/(trainData.shape[0])]

        out_valid = np.matmul(validData,W)+b
        valid_accur = [np.sum((out_valid>=0.5)==validTarget)/(validTarget.shape[0])]

        out_test = np.matmul(testData,W)+b
        test_accur = [np.sum((out_test>=0.5)==testTarget)/(testData.shape[0])]
        for i in range(iterations):
            grad_mse_W, grad_mse_b = gradMSE(W, b, trainingData, trainingLabels, reg)
            new_W = W - alpha * grad_mse_W
            new_b = b - alpha * grad_mse_b
            train_loss.append(MSE(new_W,new_b,trainingData,trainingLabels,reg))
            valid_loss.append(MSE(new_W,new_b,validData,validTarget,reg))
            test_loss.append(MSE(new_W,new_b,testData,testTarget,reg))
            out_train = np.matmul(trainData,new_W)+new_b
            train_accur.append(np.sum((out_train>=0.5)==trainTarget)/(trainData.shape[0]))

            out_valid = np.matmul(validData,new_W)+new_b
            valid_accur.append(np.sum((out_valid>=0.5)==validTarget)/(validTarget.shape[0]))

            out_test = np.matmul(testData,new_W)+new_b
            test_accur.append(np.sum((out_test>=0.5)==testTarget)/(testData.shape[0]))
            mag = np.linalg.norm(new_W-W)
            if mag<EPS:
                return new_W,new_b, train_loss, valid_loss, test_loss, train_accur, valid_accur, test_accur
            else:
                W = new_W
                b = new_b
        return W,b,train_loss, valid_loss, test_loss, train_accur, valid_accur, test_accur
    else:
        train_loss = [crossEntropyLoss(W,b,trainingData,trainingLabels,reg)]
        valid_loss = [crossEntropyLoss(W, b, validData, validTarget, reg)]
        test_loss = [crossEntropyLoss(W, b, testData, testTarget, reg)]
        out_train = np.matmul(trainData,W)+b
        train_accur = [np.sum((out_train>=0.5)==trainTarget)/(trainData.shape[0])]

        out_valid = np.matmul(validData,W)+b
        valid_accur = [np.sum((out_valid>=0.5)==validTarget)/(validTarget.shape[0])]

        out_test = np.matmul(testData,W)+b
        test_accur = [np.sum((out_test>=0.5)==testTarget)/(testData.shape[0])]
        for i in range(iterations):
            grad_mse_W, grad_mse_b = gradCE(W, b, trainingData, trainingLabels, reg)
            new_W = W - alpha * grad_mse_W
            new_b = b - alpha * grad_mse_b
            train_loss.append(crossEntropyLoss(new_W,new_b,trainingData,trainingLabels,reg))
            valid_loss.append(crossEntropyLoss(new_W,new_b,validData,validTarget,reg))
            test_loss.append(crossEntropyLoss(new_W,new_b,testData,testTarget,reg))
            out_train = np.matmul(trainData,new_W)+new_b
            train_accur.append(np.sum((out_train>=0.5)==trainTarget)/(trainData.shape[0]))

            out_valid = np.matmul(validData,new_W)+new_b
            valid_accur.append(np.sum((out_valid>=0.5)==validTarget)/(validTarget.shape[0]))

            out_test = np.matmul(testData,new_W)+new_b
            test_accur.append(np.sum((out_test>=0.5)==testTarget)/(testData.shape[0]))
            mag = np.linalg.norm(new_W-W)
            if mag<EPS:
                return new_W,new_b, train_loss, valid_loss, test_loss, train_accur, valid_accur, test_accur
            else:
                W = new_W
                b = new_b
        return W,b,train_loss, valid_loss, test_loss, train_accur, valid_accur, test_accur

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
    W = np.zeros((trainData.shape[1],1))

    # print(trainData.shape,trainTarget.shape,W.shape,testData.shape,validData.shape)
    b = 0

    alpha = 0.005
    iterations = 5000
    reg = 0.1
    EPS = 1e-7
    W, b, train_loss, valid_loss, test_loss, train_accur, valid_accur, test_accur = grad_descent(W, 
            b, trainData, trainTarget, alpha, iterations, reg, EPS, validData, testData, validTarget, testTarget, lossType = "None")
    plt.imshow(W.reshape((28,28)))
    plt.show()
    out = np.matmul(trainData,W)+b
    print(np.sum((out>=0.5)==trainTarget))
    print("Training data accuracy: ", np.sum((out>=0.5)==trainTarget)/(trainData.shape[0]))

    out_valid = np.matmul(validData,W)+b
    print(np.sum((out_valid>=0.5)==validTarget))
    print("Valid data accuracy: ", np.sum((out_valid>=0.5)==validTarget)/(validData.shape[0]))

    out_test = np.matmul(testData,W)+b
    print(np.sum((out_test>=0.5)==testTarget))
    print("Test data accuracy: ", np.sum((out_test>=0.5)==testTarget)/(testData.shape[0]))

    # print(W)
    iterations = range(len(train_loss))
    plt.plot(iterations,train_loss)
    plt.plot(iterations,valid_loss)
    plt.plot(iterations,test_loss)
    # plt.plot(iterations,train_accur)
    # plt.plot(iterations,valid_accur)
    # plt.plot(iterations,test_accur)
    plt.suptitle('MSE loss Alpha = 0.005, lambda = 0.1', fontsize=16)
    plt.legend(['train loss', 'valid loss', 'test loss'], loc='upper right')
    plt.show()
