def back_hidden_bias(target, prediction, input_out, out_weight):
    input_out[input_out > 0] = 1
    input_out[input_out < 0] = 0
    ones = np.ones((1, input_out.shape[0]))
    softmax_ce = gradCE(target, prediction)
    grad_hidden_bias = np.matmul(ones, \
     (input_out * np.matmul(softmax_ce, np.transpose(out_weight))))
    return grad_hidden_bias