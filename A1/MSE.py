def MSE(W, b, x, y, reg):
    error = np.dot(x,W) + b -y
    n = y.shape[0]
    #error_norm = np.linalg.norm(error)
    #W_norm = np.linalg.norm(W)
    mse = ((1/(2.0*n))*np.dot(np.transpose(error),error)) + ((reg/2.0)*np.dot(np.transpose(W),W))

    return mse
