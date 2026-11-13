%% Two concentric rings

% Two concentric rings with a 2-8-8-1 network
M = 400;
t = 2*pi*rand(1, M);                      % angle
c = rand(1, M) > 0.5;                     % class: 0 inner ring, 1 outer ring
r = 0.5 + 0.5*c + 0.1*randn(1, M);        % noisy radius
X = [r .* cos(t); r .* sin(t)];
Y = double(c);

% Plot the training set
figure
scatter(X(1,Y==0), X(2,Y==0), 12, 'r', 'filled');
hold on
scatter(X(1,Y==1), X(2,Y==1), 12, 'b', 'filled');
th = linspace(0, 2*pi, 200);
plot(0.75*cos(th), 0.75*sin(th), 'k--', 'LineWidth', 1.5);
axis equal;
hold off
legend('Class 0', 'Class 1', 'Boundary')

% Three runs with different learning rates
alpha1=0.5;
Nepochs=1000;
[W, b, loss1] = Backpropagation(X, Y, [2 8 8 1], 32, Nepochs, alpha1);
fprintf('Rings, accuracy: %.3f  (alpha=%f)\n', mean(round(predict(W, b, X)) == Y), alpha1);

alpha2=1;
[W, b, loss2] = Backpropagation(X, Y, [2 8 8 1], 32, Nepochs, alpha2);
fprintf('Rings, accuracy: %.3f  (alpha=%f)\n', mean(round(predict(W, b, X)) == Y), alpha2);

alpha3=2;
[W, b, loss3] = Backpropagation(X, Y, [2 8 8 1], 32, Nepochs, alpha3);
fprintf('Rings, accuracy: %.3f  (alpha=%f)\n', mean(round(predict(W, b, X)) == Y), alpha3);

% Plot the loss curves
figure; 
plot(loss1)
hold on
plot(loss2)
plot(loss3)
xlabel('Epoch'); 
ylabel('Loss'); 
title('Rings');
hold off
legend(sprintf('\\alpha = %g', alpha1), sprintf('\\alpha = %g', alpha2),sprintf('\\alpha = %g', alpha3));

%% MNIST

% MNIST handwritten digits with a 784-30-10 network
download_mnist('mnist');                               % downloaded on the first run only
files = {'mnist/train-images-idx3-ubyte', 'mnist/train-labels-idx1-ubyte', ...
            'mnist/t10k-images-idx3-ubyte', 'mnist/t10k-labels-idx1-ubyte'};
[X, labels] = read_mnist(files{1}, files{2});    % 60000 images
[Xtest, Ytest] = read_mnist(files{3}, files{4});    % 10000 images
Y = double((0:9)' == labels);           % one-hot encoding (10 x M)

% Three runs with different learning rates
alpha1 = 0.5;
Nepochs = 100;
batch_size = 10;
[W, b, loss1] = Backpropagation(X, Y, [784 30 10], batch_size, Nepochs, alpha1);
[~, pred1] = max(predict(W, b, Xtest));   % the class is the largest output
fprintf('MNIST, test accuracy: %.4f\n', mean(pred1 - 1 == Ytest));

alpha2 = 1;
Nepochs = 100;
batch_size = 10;
[W, b, loss2] = Backpropagation(X, Y, [784 30 10], batch_size, Nepochs, alpha2);
[~, pred2] = max(predict(W, b, Xtest));   % the class is the largest output
fprintf('MNIST, test accuracy: %.4f\n', mean(pred2 - 1 == Ytest));

alpha3 = 3;
Nepochs = 100;
batch_size = 10;
[W, b, loss3] = Backpropagation(X, Y, [784 30 10], batch_size, Nepochs, alpha3);
[~, pred3] = max(predict(W, b, Xtest));   % the class is the largest output
fprintf('MNIST, test accuracy: %.4f\n', mean(pred3 - 1 == Ytest));


% Plot the loss curves
figure; 
plot(loss1)
hold on
plot(loss2)
plot(loss3)
xlabel('Epoch'); 
ylabel('Loss'); 
title('MNIST');
hold off
legend(sprintf('\\alpha = %g', alpha1), sprintf('\\alpha = %g', alpha2),sprintf('\\alpha = %g', alpha3));


% Split the test images into correctly and wrongly classified
correct = (pred1 - 1) == Ytest;
wrong = (pred1 - 1) ~= Ytest;
disp("correct: " + sum(correct))
disp("wrong: "   + sum(wrong))

Xcorrect = Xtest(:,correct);
Xwrong = Xtest(:,wrong);
Ycorrect = Ytest(correct);
Ywrong = Ytest(wrong);

pred_correct = pred1(correct);
pred_wrong = pred1(wrong);


% Show N correctly classified images 
N=10;
figure
for i = 1:N
    ax = subplot(2, N, i);
    imshow(reshape(Xcorrect(:,i),28,28)');
    ax = subplot(2, N, N + i);
    axis off
    text(0.5,0.5, string(pred_correct(i)-1), 'HorizontalAlignment','center','FontSize',12)
end


% Show N misclassified images, with the predicted class and, below it,
% the true class
N=10;
figure
for i = 1:N
    ax = subplot(2, N, i);
    imshow(reshape(Xwrong(:,i),28,28)');
    ax = subplot(2, N, N + i);
    axis off
    text(0.5,0.5, string(pred_wrong(i)-1), 'HorizontalAlignment','center','FontSize',12)
    text(0.5,0, string(Ywrong(i)), 'HorizontalAlignment','center','FontSize',12)
end
