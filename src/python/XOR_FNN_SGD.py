import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y = np.array([[0, 1, 1, 0]])
M = y.shape[1]

# Number of neurons per layer
n0 = 2; n1 = 2; n2 = 1

# Hyperparameters
alpha = 0.5      # learning rate
Nepochs = 10000  # number of training iterations

# Weight initialization
#np.random.seed(42)
W1 = np.random.randn(n1, n0)
b1 = np.zeros((n1, 1))
W2 = np.random.randn(n2, n1)
b2 = np.zeros((n2, 1))

# Activation function and derivative
sigmoid = lambda z: 1 / (1 + np.exp(-z))
dsigmoid = lambda a: a * (1 - a)

loss_graph = np.zeros(Nepochs)
alpha_graph = np.zeros(Nepochs)

# Training loop
for epoch in range(1, Nepochs + 1):
    for i in range(M):
        # Take next sample
        xi = X[:, [i]]
        yi = y[0, i]

        # Forward pass
        z1 = W1 @ xi + b1
        a1 = sigmoid(z1)
        z2 = W2 @ a1 + b2
        y_hat = sigmoid(z2)

        # Compute loss
        loss = (y_hat.item() - yi)**2
        loss_graph[epoch-1] = loss

        # Backpropagation
        delta2 = (y_hat - yi) * dsigmoid(y_hat)
        dW2 = delta2 @ a1.T
        db2 = delta2

        delta1 = delta2 * W2.T * dsigmoid(a1)
        dW1 = delta1 @ xi.T
        db1 = delta1

        # Update weights with SGD
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
    # Display progress and update learning rate (StepLR)
    if epoch % 1000 == 0:
        print('Epoch %d | Loss = %.6f' % (epoch, loss))
        alpha = alpha*0.95
    alpha_graph[epoch-1] = alpha

# ============================
# Results
# ============================
print('\nFinal predictions:')
z1 = W1 @ X + b1
a1 = sigmoid(z1)
z2 = W2 @ a1 + b2
y_hat = sigmoid(z2)
print(np.vstack((X, y_hat)))

print('\nRounded predictions (0 or 1):')
print(np.vstack((X, np.round(y_hat))))

plt.plot(range(1, Nepochs + 1), alpha_graph)
plt.figure()
plt.plot(range(1, Nepochs + 1), loss_graph)
plt.show()
