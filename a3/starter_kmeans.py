import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp


def assign_data(X, MU):

    dists = distanceFunc(X, MU) 
    min_dist = tf.argmin(dists, 1)  
    return min_dist

# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    #y = np.power((X[:, np.newaxis] - MU), 2)
    #newY = tf.convert_to_tensor(y, dtype=tf.float32)
    #output = np.sum(y, axis=2)
    newX = tf.expand_dims(X,0)
    newMU = tf.expand_dims(MU, 1)
    dis = tf.reduce_sum(tf.square(tf.subtract(newX,newMU)),2)
    output = tf.transpose(dis)
    return output

  
  
def k_means(k, is_validm = False):
# Loading data
# data = np.load('data2D.npy')
    data = np.load('data100D.npy')
    [num_pts, dim] = np.shape(data)

    # For Validation set
    is_valid = is_validm
    if is_valid:
        valid_batch = int(num_pts / 3.0)
        np.random.seed(45689)
        rnd_idx = np.arange(num_pts)
        np.random.shuffle(rnd_idx)
        val_data = data[rnd_idx[:valid_batch]]
        data = data[rnd_idx[valid_batch:]]

    #graph = tf.Graph()
    #with graph.as_default():
    loss_history = np.empty(shape=[0],dtype=float)
    val_loss_history = []
    K = k
    N = num_pts
    D = dim
    inter = 1000

    X = tf.placeholder("float", shape=[None,D])
    MU_init = tf.truncated_normal([K,D],stddev=0.05)
    MU = tf.Variable(MU_init)

    distance = distanceFunc(X,MU)
    #centroid = np.power(distance,2)
    #loss = np.sum(np.amin(distance, axis=1))
    loss = tf.reduce_sum(tf.reduce_min(distance,axis = 1))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.003).minimize(loss)

    init_g = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_g)

    for step in range(inter):
        cenVal,lossVal,_ = sess.run([MU,loss,optimizer], feed_dict={X:data})
        
        loss_history = np.append(loss_history,lossVal)
        if is_valid:
            val_cenVal,val_lossVal,_ = sess.run([MU,loss,optimizer], feed_dict={X:val_data})
            val_loss_history = np.append(val_loss_history,val_lossVal)
        if step%10 ==0:
            print("iteration:", step, "loss", lossVal)
    clustering = sess.run(assign_data(X, MU),feed_dict={X: data, MU:cenVal})
    print("Final Training loss", loss_history[-1])
#     print("Final Validation loss", val_loss_history[-1])
    percentages = np.zeros(K)
#     for i in range(K):
#              percentages[i] = np.sum(np.equal(i, clustering))*100.0/len(clustering)
#              print("class:", i, "percentage:", percentages[i])
    plt.figure(1)
    plt.plot(range(len(loss_history)),loss_history,c="c", label="train_loss")
#     plt.plot(range(len(val_loss_history)),loss_history,c="b", label="validation_loss")
    plt.legend(loc = "best")
    plt.title('K-Means History')
    plt.xlabel('# of Inter')
    plt.ylabel('Loss')
    plt.show()

    k = len(cenVal)
    plt.scatter(data[:, 0], data[:, 1], c=clustering, 
                cmap=plt.get_cmap('Set3'), s=25, alpha=0.6)
    plt.scatter(cenVal[:, 0], cenVal[:, 1], marker='*', c="black", 
                cmap=plt.get_cmap('Set1'), s=50, linewidths=1)
    plt.title('K-Means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()
    return val_loss_history




if __name__ == "__main__":
    
    valid = []
    valid.append(k_means(5, True))
    valid.append(k_means(10, True))
    valid.append(k_means(15, True))
    valid.append(k_means(20, True))
    valid.append(k_means(25, True))
    valid.append(k_means(30, True))
    

plt.figure(1)
plt.plot(range(len(valid[0])),valid[0],c="r", label="K = 5")
# plt.plot(range(len(valid_losses)),valid_losses,c="r", label="valid_loss")
plt.plot(range(len(valid[1])),valid[1],c="g", label="K = 10")
plt.plot(range(len(valid[2])),valid[2],c="b", label="K = 15")
plt.plot(range(len(valid[3])),valid[3],c="m", label="K = 20")
plt.plot(range(len(valid[4])),valid[4],c="y", label="K = 25")
plt.plot(range(len(valid[5])),valid[5],c="c", label="K = 30")
plt.legend(loc = "best")
plt.title('Validation loss with different K, using K_means')
plt.xlabel('# of Inter')
plt.ylabel('Loss')
plt.show()
