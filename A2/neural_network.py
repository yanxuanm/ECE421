import tensorflow as tf
import numpy as np


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


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    beta = 0
    graph = tf.Graph()
    batch_size = 500
    n_epochs = 700
    with graph.as_default():
        stddev_o = 1.0/(hidden_units+10)
        stddev_h = 1.0/(trainData.shape[0]+hidden_units)
        # Store layers weight &amp; bias

         weights = {
        # 4x4 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([4, 4, 1, 32], mean=0.0, stddev = np.sqrt(stddev_o)))
        
        }
         
        biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
        }

        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

        # Convolutional Layer #1
        # In our first convolutional layer, we want to apply 32 4x4 filters to the input layer, 
        # with a ReLU activation function. 
        # We can use the conv2d() method in the layers module to create this layer as follows:
        conv1 = tf.nn.conv2d(
          input=input_layer,
          filter=[4, 4, 1, 32],
          strides= [1, 1, 1, 1],
          # To specify that the output tensor should have the same height and width values 
          # as the input tensor, we set padding=same here, which instructs TensorFlow to 
          # add 0 values to the edges of the input tensor to preserve height and width of 28.
          padding="SAME",
          use_cudnn_on_gpu=True)

        # ReLU activation
        conv2 = tf.nn.relu(conv1, name="ReLU activation")

        # Batch normalization layer
        batch_norm = tf.nn.batch_normalization(conv2, name = "batch_norm")
        # Pooling Layer #1

        pool1 = tf.layers.max_pooling2d(inputs=batch_norm, pool_size=[2, 2], strides=1)

        # Dense Layer
        pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 32])
        dense = tf.layers.dense(inputs=pool1_flat, units=1000, activation=tf.nn.relu)
        # dropout = tf.layers.dropout(
        #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        logits = tf.layers.dense(inputs=dense, units=10)

        predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Load training and eval data
# ((train_data, train_labels),
#  (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

# train_data = train_data/np.float32(255)
# train_labels = train_labels.astype(np.int32)  # not required

# eval_data = eval_data/np.float32(255)
# eval_labels = eval_labels.astype(np.int32)  # not required



if __name__ == '__main__':
    # Set up logging for predictions
    hidden_units = 1000
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape((trainData.shape[0], -1))
    validData = validData.reshape((validData.shape[0], -1))
    testData = testData.reshape((testData.shape[0], -1))
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
   
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="~/Documents/2019WINTER/ECE421/A2")
    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": trainData},
        y=newtrain,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # train one step and display the probabilties
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])


    mnist_classifier.train(input_fn=train_input_fn, steps=1000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": validData},
        y=newvalid,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)