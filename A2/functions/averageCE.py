def CE(target, prediction):
    ce = -np.mean(target*np.log(prediction))
    return ce 
