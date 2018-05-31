# DeepLDA

This is an implementation of Deep Linear Discriminant Analysis (Deep LDA) in Python. It needs Tensoflow and Keras libraries to be installed. DeepLDA https://arxiv.org/pdf/1704.06305.pdffrom (Qing Tian Oct 2017).

Deep LDA is a non-linear version of LDA which uses neural networks as the mapping functions instead of linear transformers. Deep LDA is originally proposed in the following paper:

Matthias Dorfer, Rainer Kelz, Gerhard Widmer, "Deep Linear Discriminant Analysis", ICLR, 2016.

It uses the Keras library with the tensorflow as a backend. But it doesn't work in Tensorflow due to the non-support of TF of the GEP solver.

The model used is an MLP and it is different from the ones used in the original paper. The base modeling network can easily get substituted with a more efficient and powerful network like CNN.
Dataset

The model is trained on MNIST dataset.
Differences with the original paper

    The main difference between my implementation and the original paper is the network architecture. They used CNN in the original paper, but this implementation uses MLP. You can easily change the model.
    I used linear SVM to train a classifier on the new features but they employed a simpler classification approach which does not need more training.

Other Implementation

The following is the link to the original implementation given by the authors of the DeepLDA paper. I adopted the loss function from here:

    Theano and Lasagne implementation from https://github.com/CPJKU/
    https://github.com/VahidooX/DeepLDA


# The implementation
 - It has been done in Tensorflow
 - Unfortunately, tensorflow doesn't support LDA as a loss


