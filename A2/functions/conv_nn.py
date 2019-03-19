def conv_net(x, weights, biases): 
    # 1. Input Layer is x. 
    epsilon = 1e-3

    # 2. A 4 × 4 convolutional layer, with 32 filters, using vertical and horizontal strides of 1.
    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    # 3. ReLU activation 
    # ReLU_conv1 = fc1 = tf.nn.relu(conv1)
    
    # 4. A batch normalization layer
    batch_mean2, batch_var2 = tf.nn.moments(conv1, [0, 1, 2])
    scale2 = tf.Variable(tf.ones([32]))
    beta2 = tf.Variable(tf.zeros([32]))
    batch_norm = tf.nn.batch_normalization(conv1,batch_mean2,batch_var2,beta2,scale2,epsilon)
    
    # 5. A max 2 × 2 max pooling layer.
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    pool_conv1 = maxpool2d(batch_norm, k=2)

    # 6. Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(pool_conv1, [-1, weights['wc2'].get_shape().as_list()[0]])
    
    # 7. Fully connected layer
    fc1 = tf.add(tf.matmul(fc1, weights['wc2']), biases['bc2'])
    
    # 7. ReLU activation
    fc1 = tf.nn.relu(fc1)
    
    # 8. Fully connected layer
    fc2 = tf.reshape(fc1, [-1, weights['wc3'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wc3']), biases['bc3'])
    
    # 9. Outputlayer
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out


pred = conv_net(x, weights, biases)

# 10. Cross Entropy loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Here you check whether the index of the maximum value of the predicted image 
# is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
