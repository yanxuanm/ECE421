    learning_rate = 0.003
    s_stddev=0.05
    X = tf.placeholder("float", [None, D], "X")
    mu = tf.Variable(tf.random_normal([K, D], stddev = s_stddev)) 
    sigma = tf.Variable(tf.random_normal([K, 1], stddev = s_stddev))
    sigma = tf.exp(sigma)
    log_PDF = log_GaussPDF(X, mu, sigma) 

    initial_pi = tf.Variable(tf.random_normal([K, 1], stddev = s_stddev))
    log_pi = tf.squeeze(hlp.logsoftmax(initial_pi))
