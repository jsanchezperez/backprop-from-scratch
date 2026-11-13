function A = predict(W, b, X, sigma)
%PREDICT Forward pass of a trained network: returns A^[L].
if nargin < 4, sigma = @(Z) 1 ./ (1 + exp(-Z)); end
A = X;
for l = 1:numel(W)
    A = sigma(W{l} * A + b{l});
end
end
