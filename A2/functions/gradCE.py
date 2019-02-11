def gradCE(target, prediction):
    softmax_ce = prediction - target
    return softmax_ce