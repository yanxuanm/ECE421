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

            logits = tf.matmul(x, W) + b 
            train_prediction = tf.sigmoid(logits)
            loss = tf.losses.sigmoid_cross_entropy(y, train_prediction)
            # Loss function using L2 Regularization
            regularizer = tf.nn.l2_loss(W)
            loss = loss + beta/2.0 * regularizer
            
            # Optimizer.
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta2=0.9999).minimize(loss)
          
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

    return trained_W, trained_b, (train_prediction>=0.5), trainTarget, l, optimizer, regularizer
