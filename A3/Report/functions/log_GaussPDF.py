def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K
    dim = tf.to_float(tf.rank(X))
    xmu = distanceFunc(X,mu)
    xmuSqu = tf.multiply(xmu,xmu)
    sigma = tf.squeeze(sigma)
    coef = tf.log(2 * np.pi * sigma)
    exp = xmu / (2 * sigma)
    mul = -0.5*dim * coef
    PDF = mul - exp
    return PDF