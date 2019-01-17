def gradMSE(W, b, x, y, reg):
    error = np.matmul(x,W) + b - y
    grad_mse_W = np.matmul(np.transpose(x),error)/(np.shape(y)[0]) + 2*reg*W
    grad_mse_b = (np.sum(error))/(np.shape(y)[0])
    return grad_mse_W, grad_mse_b