import numpy as np
import matplotlib.pyplot as plt
from Backpropagation import Backpropagation, predict, read_mnist, download_mnist

# %% Two concentric rings

# Two concentric rings with a 2-8-8-1 network
M = 400
t = 2*np.pi*np.random.rand(1, M)                    # angle
c = np.random.rand(1, M) > 0.5                      # class: 0 inner ring, 1 outer ring
r = 0.5 + 0.5*c + 0.1*np.random.randn(1, M)         # noisy radius
X = np.vstack((r * np.cos(t), r * np.sin(t)))
Y = c.astype(float)

# Plot the training set
plt.figure()
plt.scatter(X[0, Y[0]==0], X[1, Y[0]==0], 12, 'r')
plt.scatter(X[0, Y[0]==1], X[1, Y[0]==1], 12, 'b')
th = np.linspace(0, 2*np.pi, 200)
plt.plot(0.75*np.cos(th), 0.75*np.sin(th), 'k--', linewidth=1.5)
plt.axis('equal')
plt.legend(['Class 0', 'Class 1', 'Boundary'])

# Three runs with different learning rates
alpha1 = 0.5
Nepochs = 1000
W, b, loss1 = Backpropagation(X, Y, [2, 8, 8, 1], 32, Nepochs, alpha1)
print('Rings, accuracy: %.3f  (alpha=%f)' % (np.mean(np.round(predict(W, b, X)) == Y), alpha1))

alpha2 = 1
W, b, loss2 = Backpropagation(X, Y, [2, 8, 8, 1], 32, Nepochs, alpha2)
print('Rings, accuracy: %.3f  (alpha=%f)' % (np.mean(np.round(predict(W, b, X)) == Y), alpha2))

alpha3 = 2
W, b, loss3 = Backpropagation(X, Y, [2, 8, 8, 1], 32, Nepochs, alpha3)
print('Rings, accuracy: %.3f  (alpha=%f)' % (np.mean(np.round(predict(W, b, X)) == Y), alpha3))

# Plot the loss curves
plt.figure()
plt.plot(range(1, Nepochs + 1), loss1)
plt.plot(range(1, Nepochs + 1), loss2)
plt.plot(range(1, Nepochs + 1), loss3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Rings')
plt.legend([r'$\alpha$ = %g' % alpha1, r'$\alpha$ = %g' % alpha2, r'$\alpha$ = %g' % alpha3])


# %% MNIST

# MNIST handwritten digits with a 784-30-10 network
download_mnist('mnist')                                 # downloaded on the first run only
files = ['mnist/train-images-idx3-ubyte', 'mnist/train-labels-idx1-ubyte',
            'mnist/t10k-images-idx3-ubyte', 'mnist/t10k-labels-idx1-ubyte']
X, labels = read_mnist(files[0], files[1])     # 60000 images
Xtest, Ytest = read_mnist(files[2], files[3])     # 10000 images
Y = (np.arange(10).reshape(-1, 1) == labels).astype(float)   # one-hot encoding (10 x M)

# Three runs with different learning rates
alpha1 = 0.5
Nepochs = 100
batch_size = 10
W, b, loss1 = Backpropagation(X, Y, [784, 30, 10], batch_size, Nepochs, alpha1)
pred1 = np.argmax(predict(W, b, Xtest), axis=0) + 1    # the class is the largest output
print('MNIST, test accuracy: %.4f' % np.mean(pred1 - 1 == Ytest))

alpha2 = 1
Nepochs = 100
batch_size = 10
W, b, loss2 = Backpropagation(X, Y, [784, 30, 10], batch_size, Nepochs, alpha2)
pred2 = np.argmax(predict(W, b, Xtest), axis=0) + 1    # the class is the largest output
print('MNIST, test accuracy: %.4f' % np.mean(pred2 - 1 == Ytest))

alpha3 = 3
Nepochs = 100
batch_size = 10
W, b, loss3 = Backpropagation(X, Y, [784, 30, 10], batch_size, Nepochs, alpha3)
pred3 = np.argmax(predict(W, b, Xtest), axis=0) + 1    # the class is the largest output
print('MNIST, test accuracy: %.4f' % np.mean(pred3 - 1 == Ytest))


# Plot the loss curves
plt.figure()
plt.plot(range(1, Nepochs + 1), loss1)
plt.plot(range(1, Nepochs + 1), loss2)
plt.plot(range(1, Nepochs + 1), loss3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MNIST')
plt.legend([r'$\alpha$ = %g' % alpha1, r'$\alpha$ = %g' % alpha2, r'$\alpha$ = %g' % alpha3])


# Split the test images into correctly and wrongly classified
correct = (pred1 - 1) == Ytest[0]
wrong = (pred1 - 1) != Ytest[0]
print('correct: %d' % np.sum(correct))
print('wrong: %d' % np.sum(wrong))

Xcorrect = Xtest[:, correct]
Xwrong = Xtest[:, wrong]
Ycorrect = Ytest[0, correct]
Ywrong = Ytest[0, wrong]

pred_correct = pred1[correct]
pred_wrong = pred1[wrong]


# Show N correctly classified images
N = 10
plt.figure()
for i in range(N):
    ax = plt.subplot(2, N, i + 1)
    plt.imshow(Xcorrect[:, i].reshape(28, 28), cmap='gray'); plt.axis('off')
    ax = plt.subplot(2, N, N + i + 1)
    plt.axis('off')
    plt.text(0.5, 0.5, str(pred_correct[i] - 1), ha='center', fontsize=12)


# Show N misclassified images, with the predicted class and, below it,
# the true class
N = 10
plt.figure()
for i in range(N):
    ax = plt.subplot(2, N, i + 1)
    plt.imshow(Xwrong[:, i].reshape(28, 28), cmap='gray'); plt.axis('off')
    ax = plt.subplot(2, N, N + i + 1)
    plt.axis('off')
    plt.text(0.5, 0.5, str(pred_wrong[i] - 1), ha='center', fontsize=12)
    plt.text(0.5, 0, str(int(Ywrong[i])), ha='center', fontsize=12)

plt.show()
