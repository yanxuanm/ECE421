def softmax(x):
    softmax_x = np.exp(x)/sum(np.exp(x))
    return softmax_x
