def gradMSE(W, b, x, y, reg):
    error = np.dot(x,W) + b - y
    n = y.shape[0]
    gradMSE_w = np.dot(np.transpose(x),((1.0 / n) * error)) + reg*W
    gradMSE_b = np.sum(error)/n
    return gradMSE_w, gradMSE_b
