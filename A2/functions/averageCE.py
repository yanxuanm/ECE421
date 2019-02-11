def CE(target, prediction):
    ce = -np.mean(np.matmul(target, np.log(prediction)))
    return ce 