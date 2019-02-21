def learning(trainData, target, W_o, v_o, W_h, v_h, epochs, \
        gamma, learning_rate, bias_o, bias_h, validData, newvalid, testData, newtest):

    W_v_o = v_o
    b_v_o = bias_o
    W_v_h = v_h
    b_v_h = bias_h
    accuracy = []
    accuracy_valid = []
    accuracy_test = []
    loss = []
    loss_valid = []
    loss_test = []

    for i in range(epochs):

        hidden_input = np.add(np.matmul(trainData, W_h), bias_h)
        hidden_out = relu(hidden_input)
        prediction = softmax(np.add(np.matmul(hidden_out, W_o), bias_o))
        loss.append(CE(target, prediction))
        predict_result_matrix = np.argmax(prediction, axis = 1)
        actural_result_matrix = np.argmax(target, axis = 1)
        compare = np.equal(predict_result_matrix, actural_result_matrix)
        accuracy.append(np.sum((compare==True))/(trainData.shape[0]))


        hidden_input_valid = np.add(np.matmul(validData, W_h), bias_h)
        hidden_out_valid = relu(hidden_input_valid)
        prediction_valid = softmax(np.add(np.matmul(hidden_out_valid, W_o), bias_o))
        loss_valid.append(CE(newvalid, prediction_valid))
        predict_result_matrix_valid = np.argmax(prediction_valid, axis = 1)
        actural_result_matrix_valid = np.argmax(newvalid, axis = 1)
        compare_valid = np.equal(predict_result_matrix_valid, actural_result_matrix_valid)
        accuracy_valid.append(np.sum((compare_valid==True))/(validData.shape[0]))


        hidden_input_test = np.add(np.matmul(testData, W_h), bias_h)
        hidden_out_test = relu(hidden_input_test)
        prediction_test = softmax(np.add(np.matmul(hidden_out_test, W_o), bias_o))
        loss_test.append(CE(newtest, prediction_test))
        predict_result_matrix_test = np.argmax(prediction_test, axis = 1)
        actural_result_matrix_test = np.argmax(newtest, axis = 1)
        compare_test = np.equal(predict_result_matrix_test, actural_result_matrix_test)
        accuracy_test.append(np.sum((compare_test==True))/(testData.shape[0]))

        print("Iteration:", i)
        W_v_o = gamma*W_v_o + learning_rate*back_out_weight(target, prediction, hidden_out)
        W_o = W_o - W_v_o
        b_v_o = gamma*b_v_o + learning_rate*back_out_bias(target, prediction)
        bias_o = bias_o - b_v_o

        W_v_h = gamma*W_v_h + learning_rate*back_hidden_weight(target, \
            prediction, trainData, hidden_input, W_o)
        W_h = W_h - W_v_h
        b_v_h = gamma*b_v_h + learning_rate*back_hidden_bias(target, prediction, hidden_input, W_o)
        bias_h = bias_h - b_v_h
        # print("prediction: ", W_o)

    return W_o, bias_o, W_h, bias_h, accuracy, accuracy_valid, accuracy_test, loss, loss_valid, loss_test


if __name__ == '__main__':
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
    validData = validData.reshape((-1,validData.shape[1]*validData.shape[2])) 
    testData = testData.reshape((-1,testData.shape[1]*testData.shape[2]))

    hidden_units = 1000
    epochs = 200
    gamma = 0.99
    learning_rate = 0.0000001


    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
    mu = 0 # mean and standard deviation
    stddev_o = 1.0/(hidden_units+10)
    W_o = np.random.normal(mu, np.sqrt(stddev_o), (hidden_units,10))
    v_o = np.full((hidden_units, 10), 1e-5)

    stddev_h = 1.0/(trainData.shape[1]+hidden_units)
    W_h = np.random.normal(mu, np.sqrt(stddev_h), (trainData.shape[1],hidden_units))
    v_h = np.full((trainData.shape[1],hidden_units), 1e-5)

    bias_o = np.zeros((1, 10))
    bias_h = np.zeros((1, hidden_units))

    weight_o, bias_o, weight_h, bias_h, accuracy, accuracy_valid, accuracy_test, loss, \
        loss_valid, loss_test = learning(trainData, newtrain, W_o, v_o, W_h, v_h, epochs, \
        gamma, learning_rate, bias_o, bias_h, validData, newvalid, testData, newtest)


    hidden_out = relu(np.add(np.matmul(trainData, weight_h), bias_h))
    prediction = softmax(np.add(np.matmul(hidden_out, weight_o), bias_o))
    loss.append(CE(newtrain, prediction))
    predict_result_matrix = np.argmax(prediction, axis = 1)
    actural_result_matrix = np.argmax(newtrain, axis = 1)
    compare = np.equal(predict_result_matrix, actural_result_matrix)
    print("trainData accuracy: ", np.sum((compare==True))/(trainData.shape[0]))
    accuracy.append(np.sum((compare==True))/(trainData.shape[0]))

    hidden_input_valid = np.add(np.matmul(validData, weight_h), bias_h)
    hidden_out_valid = relu(hidden_input_valid)
    prediction_valid = softmax(np.add(np.matmul(hidden_out_valid, weight_o), bias_o))
    loss_valid.append(CE(newvalid, prediction_valid))
    predict_result_matrix_valid = np.argmax(prediction_valid, axis = 1)
    actural_result_matrix_valid = np.argmax(newvalid, axis = 1)
    compare_valid = np.equal(predict_result_matrix_valid, actural_result_matrix_valid)
    print("validData accuracy: ", np.sum((compare_valid==True))/(validData.shape[0]))
    accuracy_valid.append(np.sum((compare_valid==True))/(validData.shape[0]))

    hidden_input_test = np.add(np.matmul(testData, weight_h), bias_h)
    hidden_out_test = relu(hidden_input_test)
    prediction_test = softmax(np.add(np.matmul(hidden_out_test, weight_o), bias_o))
    loss_test.append(CE(newtest, prediction_test))
    predict_result_matrix_test = np.argmax(prediction_test, axis = 1)
    actural_result_matrix_test = np.argmax(newtest, axis = 1)
    compare_test = np.equal(predict_result_matrix_test, actural_result_matrix_test)
    print("testData accuracy: ", np.sum((compare_test==True))/(testData.shape[0]))
    accuracy_test.append(np.sum((compare_test==True))/(testData.shape[0]))

    iterations = range(len(accuracy))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, accuracy)
    plt.plot(iterations, accuracy_valid)
    plt.plot(iterations, accuracy_test)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    # plt.legend(['train accuracy', 'valid accuracy', 'test accuracy'], loc='lower right')
    plt.suptitle('Train, valid and test accuracy', fontsize=16)
    plt.subplot(1, 2, 2)
    plt.plot(iterations, loss)
    plt.plot(iterations, loss_valid)
    plt.plot(iterations, loss_test)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train loss', 'valid loss', 'test loss'], loc='upper right')
    plt.suptitle('Train, valid, test sets accuracy and loss', fontsize=16)
    plt.title
    plt.show()
