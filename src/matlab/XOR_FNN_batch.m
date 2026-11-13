clc; clear

% Dataset
X = [0 0 1 1; 0 1 0 1];
y = [0 1 1 0];
M = length(y);

% Number of neurons per layer
n0 = 2; n1 = 2; n2 = 1;

% Weight initialization
%rng(42); 
W1 = randn(n1, n0);
b1 = zeros(n1, 1);
W2 = randn(n2, n1);
b2 = zeros(n2, 1);

% Hyperparameters
alpha = 0.5;     % learning rate
Nepochs = 10000; % number of iterations

% Activation function and derivative
sigmoid = @(z) 1 ./ (1 + exp(-z));
dsigmoid = @(a) a .* (1 - a);

loss_graph=zeros(1, Nepochs);
alpha_graph=zeros(1, Nepochs);

% Training loop
for epoch = 1:Nepochs
    % Forward pass 
    z1 = W1 * X + b1;       
    a1 = sigmoid(z1);       
    z2 = W2 * a1 + b2;       
    y_hat = sigmoid(z2);     

    % Compute loss 
    loss = mean((y_hat - y).^2);
    loss_graph(epoch) = loss;

    % Backpropagation 
    delta2 = (y_hat - y)/M .* dsigmoid(y_hat);
    dW2 = delta2 * a1';
    db2 = sum(delta2, 2);

    delta1 = W2' * delta2 .* dsigmoid(a1);
    dW1 = delta1 * X';
    db1 = sum(delta1, 2);

    % Update weights (full-batch gradient descent)
    W2 = W2 - alpha * dW2;
    b2 = b2 - alpha * db2;
    W1 = W1 - alpha * dW1;
    b1 = b1 - alpha * db1;

    % Display progress and update learning rate (StepLR)
    if mod(epoch, 1000) == 0
        fprintf('Epoch %d | Loss = %.6f\n', epoch, loss);
       alpha = alpha * 0.85;
    end
    alpha_graph(epoch) = alpha;
end

% Results
fprintf('\nFinal predictions:\n');
z1 = W1 * X + b1;
a1 = sigmoid(z1);
z2 = W2 * a1 + b2;
y_hat = sigmoid(z2);
disp([X; y_hat]);

fprintf('\nRounded predictions (0 or 1):\n');
disp([X; round(y_hat)]);

plot(1:Nepochs, alpha_graph)
figure
plot(1:Nepochs, loss_graph)


