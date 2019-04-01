import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp


# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    newX = tf.expand_dims(X, 0)
    newMU = tf.expand_dims(MU, 1)
    dis = tf.reduce_sum(tf.square(tf.subtract(newX, newMU)), 2)
    output = tf.transpose(dis)
    return output

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K
    #SIGMA = np.dot(sigma, sigma.T)

    #guass equation
    xmu = distanceFunc(X,mu)
    xmuSqu = tf.multiply(xmu,xmu)
    sigmaSqu = tf.multiply(sigma,sigma)
    coef = 1.0/(tf.sqrt(2.0*np.pi)*sigma)
    exp = tf.exp(tf.divide(xmuSqu,(tf.transpose(-1.0/(2.0*sigmaSqu)))))
    mul = tf.multiply(tf.transpose(coef),exp)
    PDF = tf.log(mul)
    return PDF



def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    #pi*guass divide by the total probability
    log_prob = tf.add(tf.transpose(log_pi),log_PDF)
    log_sum = hlp.reduce_logsumexp(log_prob,keep_dims=True)
    output = log_prob - log_sum
    return output

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
is_valid = False
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

K = 3
D = dim
N = num_pts
inter = 600
X = tf.placeholder("float", shape=[N,D])
MU_init = tf.truncated_normal([K,D],stddev=0.25)
MU = tf.Variable(MU_init)


#for initial sigma
sig_init = tf.Variable(tf.ones([K,1])*(-3))
sig = tf.exp(sig_init)

#for initial pi
pi_init = tf.Variable(tf.zeros([K,1]))
pi = hlp.logsoftmax(pi_init)

#calculate loss
logPDF = log_GaussPDF(X,MU,sig)
logPoster = log_posterior(logPDF,pi)

loss = -1.0*(tf.reduce_sum(logPoster))
optimizer = tf.train.AdamOptimizer(0.03, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

init_g = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_g)

for step in range(inter):
    cenVal,lossVal,_ = sess.run([MU,loss,optimizer], feed_dict={X:data})
    #loss_history = np.append(loss_history,lossVal)
    if step%10 ==0:
        print("iteration:", step, "loss", lossVal) #problem: no decay:(
