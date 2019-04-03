def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K
    log_pi = tf.squeeze(log_pi)
    log_prob = tf.add(log_pi,log_PDF)
    log_sum = hlp.reduce_logsumexp(log_prob + log_pi,keep_dims=True)
    output = log_prob - log_sum
    return output