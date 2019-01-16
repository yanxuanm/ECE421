def gradMSE(W, b, x, y, reg):
    # Your implementation here
    transpose_W = np.transpose(W)
    error = np.matmul(transpose_W, x) + b - y
    grad_mse = (np.sum(np.transpose(error)* x))/(np.shape(y)[0]) + 2 * reg * W
    return grad_mse