def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    for i in range(iterations):
        grad_mse_W, grad_mse_b = gradMSE(W, b, trainingData, trainingLabels, reg)
        new_W = W - alpha * grad_mse_W
        new_b = b - alpha * grad_mse_b
        mag = np.linalg.norm(new_W-W)
        if mag<EPS:
            return new_W,new_b
        else:
            W = new_W
            b = new_b
    return W,b