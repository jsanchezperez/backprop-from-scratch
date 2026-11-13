function [W, b, loss] = Backpropagation(X, Y, Neurons, batch_size, Nepochs, alpha, sigma, dsigma)
%BACKPROPAGATION Trains a fully connected neural network with
%   backpropagation and mini-batch gradient descent.
%
%   [W, b, loss] = Backpropagation(X, Y, Neurons, batch_size, Nepochs, alpha, sigma, dsigma)
%
%   X           inputs, one column per sample (n0 x M)
%   Y           target outputs, one column per sample (nL x M)
%   Neurons     neurons per layer, Neurons = [n0 n1 ... nL]
%   batch_size  mini-batch size (1: SGD; M: full batch)
%   Nepochs     number of epochs (default 1000)
%   alpha       learning rate (default 0.5)
%   sigma       activation function (default: sigmoid)
%   dsigma      its derivative, written in terms of A = sigma(Z)
%
%   W{l}, b{l}  weights and biases of layer l = 1..L
%   loss        mean loss of each epoch

if nargin < 5, Nepochs = 1000; end
if nargin < 6, alpha = 0.5; end
if nargin < 7, sigma = @(Z) 1 ./ (1 + exp(-Z)); end
if nargin < 8, dsigma = @(A) A .* (1 - A); end

L = numel(Neurons) - 1;   % number of layers (not counting the input layer)
M = size(X, 2);           % number of samples
batch_size = min(batch_size, M);

% Initialization: random weights N(0, 1/n^[l-1]) and zero biases
W = cell(1, L);  b = cell(1, L);
for l = 1:L
    W{l} = randn(Neurons(l+1), Neurons(l)) / sqrt(Neurons(l));
    b{l} = zeros(Neurons(l+1), 1);
end

A = cell(1, L);  dW = cell(1, L);  db = cell(1, L);
loss = zeros(1, Nepochs);
for epoch = 1:Nepochs
    order = randperm(M);                                 % shuffle the samples
    for k = 1:batch_size:M
        batch = order(k : min(k + batch_size - 1, M));   % mini-batch indices
        Xb = X(:, batch);  Yb = Y(:, batch);
        Mb = numel(batch);                               % size of this mini-batch

        % Forward pass
        A0 = Xb;
        for l = 1:L
            A{l} = sigma(W{l} * A0 + b{l});
            A0 = A{l};
        end
        E = A{L} - Yb;
        loss(epoch) = loss(epoch) + sum(E(:).^2) / (2 * M);

        % Backward pass (backpropagation)
        D = E .* dsigma(A{L});                 
        for l = L:-1:1
            if l > 1, A0 = A{l-1}; else, A0 = Xb; end
            dW{l} = D * A0' / Mb;              
            db{l} = sum(D, 2) / Mb;
            if l > 1                           
                D = (W{l}' * D) .* dsigma(A{l-1});
            end
        end

        % Gradient descent
        for l = 1:L
            W{l} = W{l} - alpha * dW{l};
            b{l} = b{l} - alpha * db{l};
        end
    end
    % Learning rate decay
    % alpha = alpha * 0.95;
end
end
