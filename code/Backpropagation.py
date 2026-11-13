import numpy as np


def Backpropagation(X, Y, Neurons, batch_size, Nepochs=1000, alpha=0.5,
                    sigma=None, dsigma=None):
    """Trains a fully connected neural network with backpropagation and
    mini-batch gradient descent.

    W, b, loss = Backpropagation(X, Y, Neurons, batch_size, Nepochs, alpha, sigma, dsigma)

    X           inputs, one column per sample (n0 x M)
    Y           target outputs, one column per sample (nL x M)
    Neurons     neurons per layer, Neurons = [n0, n1, ..., nL]
    batch_size  mini-batch size (1: SGD; M: full batch)
    Nepochs     number of epochs (default 1000)
    alpha       learning rate (default 0.5)
    sigma       activation function (default: sigmoid)
    dsigma      its derivative, written in terms of A = sigma(Z)

    W[l], b[l]  weights and biases of layer l = 1..L (index 0 is unused)
    loss        mean loss of each epoch
    """
    if sigma is None:
        sigma = lambda Z: 1 / (1 + np.exp(-Z))
    if dsigma is None:
        dsigma = lambda A: A * (1 - A)

    L = len(Neurons) - 1      # number of layers (not counting the input layer)
    M = X.shape[1]            # number of samples
    batch_size = min(batch_size, M)

    # Initialization: random weights N(0, 1/n^[l-1]) and zero biases
    W = [None] * (L + 1);  b = [None] * (L + 1)
    for l in range(1, L + 1):
        W[l] = np.random.randn(Neurons[l], Neurons[l-1]) / np.sqrt(Neurons[l-1])
        b[l] = np.zeros((Neurons[l], 1))

    A = [None] * (L + 1);  dW = [None] * (L + 1);  db = [None] * (L + 1)
    loss = np.zeros(Nepochs)
    for epoch in range(Nepochs):
        order = np.random.permutation(M)                 # shuffle the samples
        for k in range(0, M, batch_size):
            batch = order[k : min(k + batch_size, M)]    # mini-batch indices
            Xb = X[:, batch];  Yb = Y[:, batch]
            Mb = len(batch)                              # size of this mini-batch

            # Forward pass
            A0 = Xb
            for l in range(1, L + 1):
                A[l] = sigma(W[l] @ A0 + b[l])
                A0 = A[l]
            E = A[L] - Yb
            loss[epoch] = loss[epoch] + np.sum(E**2) / (2 * M)

            # Backward pass (backpropagation)
            D = E * dsigma(A[L])
            for l in range(L, 0, -1):
                if l > 1: A0 = A[l-1]
                else: A0 = Xb
                dW[l] = D @ A0.T / Mb
                db[l] = np.sum(D, axis=1, keepdims=True) / Mb
                if l > 1:
                    D = (W[l].T @ D) * dsigma(A[l-1])

            # Gradient descent
            for l in range(1, L + 1):
                W[l] = W[l] - alpha * dW[l]
                b[l] = b[l] - alpha * db[l]
        # Learning rate decay
        # alpha = alpha * 0.95
    return W, b, loss


def predict(W, b, X, sigma=None):
    """Forward pass of a trained network: returns A^[L]."""
    if sigma is None:
        sigma = lambda Z: 1 / (1 + np.exp(-Z))
    A = X
    for l in range(1, len(W)):
        A = sigma(W[l] @ A + b[l])
    return A


def read_mnist(image_file, label_file):
    """Reads a pair of MNIST IDX files (already uncompressed).
    X: 784 x M, one image per column, with values in [0, 1]
    labels: 1 x M, digits from 0 to 9
    """
    with open(image_file, 'rb') as f:
        header = np.fromfile(f, dtype='>i4', count=4)   # magic number, M, rows, columns
        X = np.fromfile(f, dtype=np.uint8).reshape(header[1], header[2] * header[3]).T / 255
    with open(label_file, 'rb') as f:
        np.fromfile(f, dtype='>i4', count=2)         # magic number, M
        labels = np.fromfile(f, dtype=np.uint8).reshape(1, -1).astype(float)
    return X, labels
