def back_hidden_weight(target, prediction, input, input_out, out_weight):
    input_out[input_out > 0] = 1
    input_out[input_out < 0] = 0
    softmax_ce = gradCE(target, prediction)
    grad_hidden_weight = np.matmul(np.transpose(input), \
     (input_out * np.matmul(softmax_ce, np.transpose(out_weight))))
    return grad_hidden_weight