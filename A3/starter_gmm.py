import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from collections import Counter

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

    
def kmeans(K, is_valid=False):
  
    # Loading data
    #data = np.load('data2D.npy')
    data = np.load('data100D.npy')
    [N, D] = np.shape(data)


    #plt.scatter(data.T[0], data.T[1])
    # For Validation set
    if is_valid:
        valid_batch = int(N / 3.0)
        np.random.seed(45689)
        rnd_idx = np.arange(N)
        np.random.shuffle(rnd_idx)
        val_data = data[rnd_idx[:valid_batch]]
        data = data[rnd_idx[valid_batch:]]

    np.random.seed(521)

    num_ep = 1000
    losses = []
    assg = []
    valid_losses = []
    
    learning_rate = 0.003
    s_stddev=0.05
    X = tf.placeholder("float", [None, D], "X")
    mu = tf.Variable(tf.random_normal([K, D], stddev = s_stddev)) 
    sigma = tf.Variable(tf.random_normal([K, 1], stddev = s_stddev))
    sigma = tf.exp(sigma)
    log_PDF = log_GaussPDF(X, mu, sigma) 

    initial_pi = tf.Variable(tf.random_normal([K, 1], stddev = s_stddev))
    log_pi = tf.squeeze(hlp.logsoftmax(initial_pi))

    # reduce the total loss
    loss = - tf.reduce_sum(hlp.reduce_logsumexp(log_PDF + log_pi, 1, keep_dims=True))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    # determine the clusters
    pred = tf.argmax(tf.nn.softmax(log_posterior(log_PDF, log_pi)), 1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for i in range(num_ep):
            cenVal, cur_l, _, assg = sess.run([mu, loss, train, pred], feed_dict={X:data})
            losses.append(cur_l)
#             if i%10 ==0:
#                 print("iteration:", i, "loss", cur_l)
            if is_valid:
                _, valid_loss, _, _ = sess.run([mu, loss, train, pred], feed_dict={X: val_data})
                valid_losses.append(valid_loss)

        print("K = {}, Final loss: {}".format(K, losses[-1]))
        clusters = Counter(assg)
        assg=np.int32(assg)
        for i in range(K):
            print("Cluster {}: {}%".format(i, clusters[i]*100.0/N))
            plt.scatter(data[:, 0], data[:, 1], c=assg, cmap=plt.get_cmap('Set3'), s=25, alpha=0.6)
        plt.scatter(cenVal[:, 0], cenVal[:, 1], marker='*', c="black", cmap=plt.get_cmap('Set1'), s=80, linewidths=2)
        plt.title('K-Means Clustering')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.grid()
        plt.show()

        

#     if is_valid:
#         print("K = {}, Validation loss: {}".format(K, valid_loss))

    plt.figure(1)
    plt.plot(range(len(losses)),losses,c="c", label="train_loss")
    plt.plot(range(len(valid_losses)),valid_losses,c="r", label="valid_loss")
    plt.legend(loc = "best")
    plt.title('K-Means History')
    plt.xlabel('# of Inter')
    plt.ylabel('Loss')
    plt.show()
    
    return valid_losses


if __name__ == "__main__":
    
    valid = []
    valid.append(kmeans(5, True))
    valid.append(kmeans(10, True))
    valid.append(kmeans(15, True))
    valid.append(kmeans(20, True))
    valid.append(kmeans(25, True))
    valid.append(kmeans(30, True))

plt.figure(1)
plt.plot(range(len(valid[0])),valid[0],c="r", label="K = 5")
# plt.plot(range(len(valid_losses)),valid_losses,c="r", label="valid_loss")
plt.plot(range(len(valid[1])),valid[1],c="g", label="K = 10")
plt.plot(range(len(valid[2])),valid[2],c="b", label="K = 15")
plt.plot(range(len(valid[3])),valid[3],c="m", label="K = 20")
plt.plot(range(len(valid[4])),valid[4],c="y", label="K = 25")
plt.plot(range(len(valid[5])),valid[5],c="c", label="K = 30")
plt.legend(loc = "best")
plt.title('Validation loss with different K, using data100D')
plt.xlabel('# of Inter')
plt.ylabel('Loss')
plt.show()
