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
    beta = 0.1
    graph = tf.Graph()
    with graph.as_default():
        # Initialize weight and bias tensors
        W = tf.Variable(tf.truncated_normal(shape=(784, 1), mean=0.0, stddev=0.5, dtype=tf.float32, seed =None, name=None))
        b = tf.Variable(tf.zeros(1))

        x = tf.placeholder(tf.float32, shape=(3500, 784))
        y = tf.placeholder(tf.float32, shape=(3500, 1))
        reg = tf.placeholder(tf.float32, shape = (1))

        valid_data = tf.placeholder(tf.float32, shape=(100, 784))
        valid_label = tf.placeholder(tf.int8, shape=(100, 1))

        test_data = tf.placeholder(tf.float32, shape=(145, 784))
        test_label = tf.placeholder(tf.int8, shape=(145, 1))

        tf.set_random_seed(421)
        if lossType == "MSE":
            predictions = tf.matmul(x,W)+b
            loss = tf.losses.mean_squared_error(y, predictions)
            regularizer = tf.nn.l2_loss(W)
            loss = loss + beta/2.0 * regularizer
            optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
            # Predictions for the training, validation, and test data.
            train_prediction = tf.matmul(x,W)+b
            valid_prediction = tf.matmul(valid_data,W)+b
            test_prediction = tf.matmul(test_data,W)+b

        elif lossType == "CE":
            logits = tf.matmul(x, W) + b 
            # Original loss function
            loss = tf.losses.sigmoid_cross_entropy(y, logits)
            # Loss function using L2 Regularization
            regularizer = tf.nn.l2_loss(W)
            loss = loss + beta/2.0 * regularizer
            
            # Optimizer.
            optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
          
            # Predictions for the training, validation, and test data.
            train_prediction = tf.sigmoid(logits)
            valid_prediction = tf.sigmoid(tf.matmul(valid_data, W) + b)
            test_prediction = tf.sigmoid(tf.matmul(test_data, W) + b)

        num_steps = 5000
        trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
        trainData = trainData.reshape((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
        validData = validData.reshape((-1,validData.shape[1]*validData.shape[2])) 

        testData = testData.reshape((-1,testData.shape[1]*testData.shape[2]))
        def accuracy(predictions, labels):
            return (100.0 * np.sum((predictions>=0.5)==labels) / np.shape(predictions)[0])


        with tf.Session(graph=graph) as session:
            # This is a one-time operation which ensures the parameters get initialized as
            # we described in the graph: random weights for the matrix, zeros for the
            # biases. 
            feed_dict = {
                        x: trainData, 
                        y: trainTarget,
                        valid_data: validData,
                        valid_label: validTarget,
                        test_data: testData,
                        test_label: testTarget
            }
            tf.global_variables_initializer().run()
            print('Initialized')
            for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
                _, trained_W, trained_b, l, predictions, v_prediction, t_prediction = session.run([optimizer, W, b, loss, train_prediction, valid_prediction, test_prediction], feed_dict=feed_dict)
                if (step % 100== 0):
                    print('Loss at step {}: {}'.format(step, l))
                    print('Training accuracy: {:.1f}'.format(accuracy(predictions, trainTarget)))
                    # Calling .eval() on valid_prediction is basically like calling run(), but
                    # just to get that one numpy array. Note that it recomputes all its graph
                    # dependencies.
                    
                    # You don't have to do .eval above because we already ran the session for the

                    # train_prediction
                    print('Validation accuracy: {:.1f}'.format(accuracy(v_prediction, validTarget)))
                    print('Test accuracy: {:.1f}'.format(accuracy(t_prediction, testTarget))) 

    # Your implementation here
    return trained_W, trained_b, (predictions>=0.5), trainTarget, l, optimizer, regularizer


if __name__ == '__main__':
    trained_W, trained_b, predictions, trainTarget, l, optimizer, regularizer = buildGraph(lossType = "CE")

    print(regularizer)