import tensorflow as tf
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


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    beta = 0
    graph = tf.Graph()
    batch_size = 500
    n_epochs = 700
    with graph.as_default():
        # Initialize weight and bias tensors
        W = tf.Variable(tf.truncated_normal(shape=(784, 1), mean=0.0, stddev=0.5, dtype=tf.float32, seed =None, name=None))
        b = tf.Variable(tf.zeros(1))

        x = tf.placeholder(tf.float32, shape=(batch_size, 784))
        y = tf.placeholder(tf.float32, shape=(batch_size, 1))
        # reg = tf.placeholder(tf.float32, shape = (1))

        valid_data = tf.placeholder(tf.float32, shape=(100, 784))
        valid_label = tf.placeholder(tf.int8, shape=(100, 1))

        test_data = tf.placeholder(tf.float32, shape=(145, 784))
        test_label = tf.placeholder(tf.int8, shape=(145, 1))

        tf.set_random_seed(421)
        if lossType == "MSE":
            # predictions = tf.matmul(x,W)+b
            # loss = tf.losses.mean_squared_error(y, predictions)
            # regularizer = tf.nn.l2_loss(W)
            # loss = loss + beta/2.0 * regularizer
            train_prediction = tf.matmul(x,W)+b
            loss = tf.losses.mean_squared_error(y, train_prediction)
            regularizer = tf.nn.l2_loss(W)
            loss = loss + beta/2.0 * regularizer

            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-04).minimize(loss)
            # Predictions for the training, validation, and test data.
            

            valid_prediction = tf.matmul(valid_data,W)+b
            valid_loss = tf.losses.mean_squared_error(valid_label, valid_prediction)
            regularizer = tf.nn.l2_loss(W)
            valid_loss = valid_loss + beta/2.0 * regularizer

            test_prediction = tf.matmul(test_data,W)+b
            test_loss = tf.losses.mean_squared_error(test_label, test_prediction)
            regularizer = tf.nn.l2_loss(W)
            test_loss = test_loss + beta/2.0 * regularizer

        elif lossType == "CE":
            # logits = tf.matmul(x, W) + b 
            # # Original loss function
            # loss = tf.losses.sigmoid_cross_entropy(y, logits)
            # logits = tf.sigmoid(logits)
            # # Loss function using L2 Regularization
            # regularizer = tf.nn.l2_loss(W)
            # loss = loss + beta/2.0 * regularizer

            logits = tf.matmul(x, W) + b 
            train_prediction = tf.sigmoid(logits)
            loss = tf.losses.sigmoid_cross_entropy(y, train_prediction)
            # Loss function using L2 Regularization
            regularizer = tf.nn.l2_loss(W)
            loss = loss + beta/2.0 * regularizer
            
            # Optimizer.
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.99).minimize(loss)
          
            # Predictions for the training, validation, and test data.

            logits = tf.matmul(valid_data,W) + b 
            valid_prediction = tf.sigmoid(tf.matmul(valid_data, W) + b)
            valid_loss = tf.losses.sigmoid_cross_entropy(valid_label, valid_prediction)
            # Loss function using L2 Regularization
            regularizer = tf.nn.l2_loss(W)
            valid_loss = valid_loss + beta/2.0 * regularizer

            logits = tf.matmul(test_data,W) + b 
            test_prediction = tf.sigmoid(tf.matmul(test_data, W) + b)
            test_loss = tf.losses.sigmoid_cross_entropy(test_label, test_prediction)
            # Loss function using L2 Regularization
            regularizer = tf.nn.l2_loss(W)
            test_loss = test_loss + beta/2.0 * regularizer
        
        def accuracy(predictions, labels):
            return (np.sum((predictions>=0.5)==labels) / np.shape(predictions)[0])

        with tf.Session(graph=graph) as session:
            # This is a one-time operation which ensures the parameters get initialized as
            # we described in the graph: random weights for the matrix, zeros for the
            # biases. 

            n_batches = int(3500/batch_size)
            tf.global_variables_initializer().run()
            print('Initialized')
            training_loss = []
            validating_loss = []
            testing_loss = []
            train_accur = []
            valid_accur = []
            test_accur = []
            for i in range(n_epochs): 
                trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
                trainData = trainData.reshape((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
                validData = validData.reshape((-1,validData.shape[1]*validData.shape[2])) 
                testData = testData.reshape((-1,testData.shape[1]*testData.shape[2]))
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
                total_loss = 0

                for j in range(n_batches):
                    X_batch = trainData[j*batch_size:(j+1)*batch_size,]
                    Y_batch = trainTarget[j*batch_size:(j+1)*batch_size,]
                    _, trained_W, trained_b, l, predictions, v_loss, v_prediction, t_loss, t_prediction = session.run(
                        [optimizer, W, b, loss, train_prediction, valid_loss, valid_prediction, test_loss, test_prediction], 
                        {x: X_batch, 
                        y: Y_batch,
                        valid_data: validData,
                        valid_label: validTarget,
                        test_data: testData,
                        test_label: testTarget})
                    # print(b.eval())
                if (i % 1 == 0):
                    training_loss.append(l)
                    validating_loss.append(v_loss)
                    testing_loss.append(t_loss)
                    train_accur.append(accuracy(predictions, Y_batch))
                    valid_accur.append(accuracy(v_prediction, validTarget))
                    test_accur.append(accuracy(t_prediction, testTarget))
                    print('Loss at step {}: {}'.format(i, l))
                    print('Training accuracy: {}'.format(accuracy(predictions, Y_batch)))

                    # train_prediction
                    print('Validation accuracy: {}'.format(accuracy(v_prediction, validTarget)))
                    print('Test accuracy: {}'.format(accuracy(t_prediction, testTarget))) 
                
        # plt.subplot(1, 2, 1)
        # plt.plot(range(n_epochs),training_loss)
        # plt.plot(range(n_epochs),validating_loss)
        # plt.plot(range(n_epochs),testing_loss)
        # plt.suptitle('Adam on CE', fontsize=16)
        # plt.legend(['train loss', 'valid loss', 'test loss'], loc='upper right')
        # plt.subplot(1, 2, 2)
        plt.plot(range(n_epochs),train_accur)
        plt.plot(range(n_epochs),valid_accur)
        plt.plot(range(n_epochs),test_accur)
        plt.suptitle('Adam epsilon=1e-04 accuracy', fontsize=16)
        plt.legend(['train accuracy', 'valid accuracy', 'test accuracy'], loc='lower right')

        plt.show()



    # Your implementation here
    return trained_W, trained_b, (predictions>=0.5), trainTarget, l, optimizer, regularizer


if __name__ == '__main__':
    trained_W, trained_b, predictions, trainTarget, l, optimizer, regularizer = buildGraph(lossType = "MSE")

    plt.imshow(trained_W.reshape((28,28)))
    plt.suptitle('Visualize weight matrix', fontsize=12)
    plt.show()
