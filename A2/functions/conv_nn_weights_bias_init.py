x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])
weights = {
    'wc1': tf.get_variable('W0', shape=(4,4,1,32), \
        initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(14*14*32,64), \
        initializer=tf.contrib.layers.xavier_initializer()),  
    'wc3': tf.get_variable('W3', shape=(64,128), \
        initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,n_classes), \
        initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), \
        initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), \
        initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), \
        initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), \
        initializer=tf.contrib.layers.xavier_initializer()),
}