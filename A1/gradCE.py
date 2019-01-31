def gradCE(W, b, x, y, reg):
    #https://math.stackexchange.com/questions/477207/derivative-of-cost-function-for-logistic-regression
    h = 1.0/(1.0+np.exp((x @ -W)+b))
    n = y.shape[0]
    gradCE_w =(np.dot(np.transpose(x),(h-y)))/n + reg*W
    gradCE_b = (np.sum(h-y))/n
    return gradCE_w, gradCE_b
