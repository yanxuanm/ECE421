def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType = "None"):
    cost_history = np.zeros(iterations)
    #theta_history = np.zeros((iterations, W.shape[0] + 1))

    if lossType == "MSE":
        for i in range(iterations):
            gradMSE_w,gradMSE_b = gradMSE(W,b,trainingData,trainingLabels,reg)
            W = W - alpha*gradMSE_w
            b = b - alpha*gradMSE_b
            #theta_history[i] = np.transpose([b,W])
            cost_history[i] = MSE(W,b,trainingData,trainingLabels,reg)
            if i != 0:
                old_cost = cost_history[i-1]
                cur_cost = cost_history[i]
                if(abs(old_cost - cur_cost) < EPS):
                    return W, b, cost_history
        return W, b, cost_history
    elif lossType == "CE":
        for i in range(iterations):
            gradCE_w, gradCE_b = gradCE(W, b, trainingData, trainingLabels, reg)
            W = W - alpha * gradCE_w
            b = b - alpha * gradCE_b
            #theta_history[i] = np.transpose([b, W])
            cost_history[i] = crossEntropyLoss(W, b, trainingData, trainingLabels, reg)
            if i != 0:
                old_cost = cost_history[i-1]
                cur_cost = cost_history[i]
                if(abs(old_cost - cur_cost) < EPS):
                    return W, b, cost_history
        return W, b, cost_history
