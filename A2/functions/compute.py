def computeLayer(X, W, b):
    compute_layer = np.matmul(X_trans, W) + b
    return compute_layer
