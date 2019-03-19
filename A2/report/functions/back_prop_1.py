def back_out_weight(target, prediction, hidden_out):
    softmax_ce = gradCE(target, prediction)
    hidden_out_transpose = np.transpose(hidden_out)
    grad_out_weight = np.matmul(hidden_out_transpose, softmax_ce)
    return grad_out_weight