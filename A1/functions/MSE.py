def MSE(W, b, x, y, reg):
    error = np.matmul(x,W) + b - y
    mse = (np.sum(error*error))/((2*np.shape(y)[0])) + reg/2*np.sum(W*W)
    return mse