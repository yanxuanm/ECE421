def norm_equation(x, y):
    x_b = np.ones((np.shape(x)[0],1))
    xx = np.hstack((x,x_b))
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(xx), xx)), np.transpose(xx)), y)
    print(W.shape)
    return W[:-1,:],W[-1][0]