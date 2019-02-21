def back_out_bias(target, prediction):
    softmax_ce = gradCE(target, prediction)
    ones = np.ones((1, target.shape[0]))
    grad_out_bias = np.matmul(ones, softmax_ce)
    return grad_out_bias