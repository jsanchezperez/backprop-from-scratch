function [X, labels] = read_mnist(image_file, label_file)
%READ_MNIST Reads a pair of MNIST IDX files (already uncompressed).
%   X: 784 x M, one image per column, with values in [0, 1]
%   labels: 1 x M, digits from 0 to 9
f = fopen(image_file, 'r', 'ieee-be');
header = fread(f, 4, 'int32');      % magic number, M, rows, columns
X = fread(f, [header(3) * header(4), header(2)], 'uint8=>double') / 255;
fclose(f);
f = fopen(label_file, 'r', 'ieee-be');
fread(f, 2, 'int32');            % magic number, M
labels = fread(f, [1, Inf], 'uint8=>double');
fclose(f);
end
