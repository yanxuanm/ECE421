def crossEntropyLoss(W, b, x, y, reg):
    epsilon = 0.001
    h = 1.0/(1.0+np.exp((x @ -W)+b))
    error = -(np.sum(y*np.log(h+epsilon) + (1.0-y)*np.log(1.0-h + epsilon)))
    n = y.shape[0]
    CE = (1.0/n) * error + (reg / 2.0 * np.dot(np.transpose(W), W))
    return CE
