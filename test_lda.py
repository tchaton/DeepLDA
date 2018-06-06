import os
try:
    del os.environ["CUDA_VISIBLE_DEVICES"]
except:
    pass
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True
session = tf.Session(config=config)


def lda_loss(th=33):
    """
    The main loss function (inner_lda_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """

    def inner_lda_objective(y_true, y_pred):
        """
        It is the loss function of LDA as introduced in the original paper.
        It is adopted from the the original implementation in the following link:
        https://github.com/CPJKU/deep_lda
        Note: it is implemented by Theano tensor operations, and does not work on Tensorflow backend
        """
        r = 1e-4

        # init groups
        # yt = tf.cast(tf.contrib.layers.flatten(y_true), tf.float32)
        print(y_true.get_shape ())
        #indexes = tf.argmax (y_true, axis=-1)
        locations = tf.where (tf.equal (y_true, 1))
        indices = locations[:, 1]
        y, idx = tf.unique (indices)

        def fn(unique, indexes, preds):
            u_indexes = tf.where (tf.equal (unique, indexes))
            u_indexes = tf.reshape (u_indexes, (1, -1))
            X = tf.gather (preds, u_indexes)
            X_mean = X - tf.reduce_mean (X, axis=1)
            m = tf.cast (tf.shape (X_mean)[1], tf.float32)
            # X_mean = tf.squeeze(X_mean)
            print(tf.shape (X_mean))
            return (1 / (m - 1)) * tf.matmul (tf.transpose (X_mean[0]), X_mean[0])

        covs_t = tf.map_fn (lambda x: fn (x, indices, y_pred), y, dtype=tf.float32)[0]

        Sw_t = tf.reduce_mean (covs_t, axis=0)

        # compute total scatter
        Xt_bar = y_pred - tf.reduce_mean (y_pred, axis=0)
        # m = T.cast(Xt_bar.shape[0], 'float32')
        m = tf.cast (tf.shape (Xt_bar)[1], tf.float32)
        St_t = (1 / (m - 1)) * tf.matmul (tf.transpose (Xt_bar), Xt_bar)

        # compute between scatter
        Sb_t = St_t - Sw_t

        # cope for numerical instability (regularize)
        dim = tf.shape (y)[0]
        Sw_t += tf.eye (dim) * r

        evals_t = np.abs (tf.linalg.eigvalsh (tf.matrix_inverse (St_t) * Sw_t))
        # compute eigenvalues
        # evals_t = X = tf.matrix_solve(Sb_t, St_t) #Sb_t, St_t # B^{-1} A x = \lambda x
        th_eigenvals = tf.contrib.distributions.percentile (evals_t, th)
        evals_t_mask = evals_t > th_eigenvals
        evals_t = tf.boolean_mask (evals_t, evals_t_mask)
        cost = tf.reduce_mean (evals_t)
        return -cost
    return inner_lda_objective

input_size = 784
output_size = 10
encoding_dims = [1024, 1024, 10]
dims = [input_size] + [1024, 1024, 10]
L = len(dims) - 1
epochs = 10
batch_size = 32

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("/home/thomas/mnist/", one_hot=True)


def dense_block(x, W, b, activation='relu'):
    x = tf.matmul(x, W) + b
    if activation == 'relu':
        return tf.nn.relu(x)
    if activation == 'linear':
        return x

input_tensor = tf.placeholder(tf.float32, [None, input_size])
y_true = tf.placeholder(tf.float32, [None, output_size])

initializer = tf.contrib.layers.xavier_initializer()
Ws = [tf.Variable(initializer([dims[i], dims[i+1]])) for i in range(L)]
Bs = [tf.Variable(initializer([dims[i+1]])) for i in range(L)]

x = input_tensor

for i in range(L):
    if i == L:
        x = dense_block(x, Ws[i], Bs[i], activation='linear')
    else:
        x = dense_block(x, Ws[i], Bs[i])
a_final = x

cost = lda_loss()(y_true, a_final)
opti = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run([init])
    for i in range(epochs):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict = {input_tensor:x_batch, y_true:y_batch}
        r = sess.run([cost], feed_dict=feed_dict)
        print(np.array(r))
    sess.close()
