def gradCE(W, b, x, y, reg):
    y_hat = 1.0/(1.0+np.exp(-(np.matmul(x,W)+b)))
    der_w =  np.matmul(np.transpose(x), (y_hat - y))/(np.shape(y)[0]) + 2*reg*W
    return der_w, np.sum((y_hat - y))/(np.shape(y)[0])