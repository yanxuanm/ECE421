
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# %matplotlib inline
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], \
        padding='SAME', use_cudnn_on_gpu=True)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], \
        strides=[1, k, k, 1],padding='SAME')

def conv_net(x, weights, biases): 
    epsilon = 1e-3


    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # ReLU activation 
    ReLU_conv1 = fc1 = tf.nn.relu(conv1)
    # A batch normalization layer
    batch_mean2, batch_var2 = tf.nn.moments(ReLU_conv1, [0, 1, 2])
    scale2 = tf.Variable(tf.ones([32]))
    beta2 = tf.Variable(tf.zeros([32]))
    batch_norm = tf.nn.batch_normalization(ReLU_conv1,batch_mean2,batch_var2,beta2,scale2,epsilon)
#     print(np.shape(batch_var2))
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    pool_conv1 = maxpool2d(batch_norm, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(pool_conv1, [-1, weights['wc2'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wc2']), biases['bc2'])
    fc1 = tf.nn.relu(fc1)
    
    fc2 = tf.reshape(fc1, [-1, weights['wc3'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wc3']), biases['bc3'])
    fc2 = tf.nn.relu(fc2)
#     prediction = tf.nn.softmax(fc2, name="softmax_tensor")
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out



training_iters = 50 
learning_rate = 1e-4 
batch_size = 32
# MNIST data input (img shape: 28*28)
n_input = 28

# MNIST total classes (A-J)
n_classes = 10

x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])
weights = {
    'wc1': tf.get_variable('W0', shape=(4,4,1,32), \
        initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(14*14*32,64), \
        initializer=tf.contrib.layers.xavier_initializer()),  
    'wc3': tf.get_variable('W3', shape=(64,128), \
        initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,n_classes), \
        initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), \
        initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), \
        initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), \
        initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), \
        initializer=tf.contrib.layers.xavier_initializer()),
}

pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    valid_losses = []
    test_losses = []
    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape(-1, 28, 28, 1)
    validData = validData.reshape(-1, 28, 28, 1)
    testData = testData.reshape(-1, 28, 28, 1)
    
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    
    for i in range(training_iters):
        trainData, trainTarget = shuffle(trainData, trainTarget)
        newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
        for batch in range(len(trainData)//batch_size):
            batch_x = trainData[batch*batch_size:min((batch+1)*batch_size,len(trainData))]
            batch_y = newtrain[batch*batch_size:min((batch+1)*batch_size,len(newtrain))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        loss, acc = sess.run([cost, accuracy], feed_dict={x: trainData, y: newtrain})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,test_loss = sess.run([accuracy,cost], feed_dict={x: testData,y : newtest})
        valid_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: validData,y : newvalid})
        train_loss.append(loss)
        valid_losses.append(valid_loss)
        test_losses.append(test_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        valid_accuracy.append(valid_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()

plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
plt.plot(range(len(train_loss)), test_losses, 'r', label='Test loss')
plt.plot(range(len(train_loss)), valid_losses, 'g', label='Valid loss')
plt.title('Training, valid and Test loss')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()

plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
plt.plot(range(len(train_loss)), valid_accuracy, 'g', label='valid Accuracy')
plt.title('Training, valid and Test Accuracy')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()