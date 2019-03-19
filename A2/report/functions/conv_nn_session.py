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