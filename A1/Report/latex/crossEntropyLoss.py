def crossEntropyLoss(W, b, x, y, reg):
    y_hat = 1.0/(1.0+np.exp(-(np.matmul(x,W)+b)))
    cross_entropy_loss = (np.sum(-(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))))/(np.shape(y)[0]) + reg/2*np.sum(W*W)
    print(cross_entropy_loss)
    return cross_entropy_loss