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