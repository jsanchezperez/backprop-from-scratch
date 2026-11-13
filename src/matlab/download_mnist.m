function download_mnist(folder)
%DOWNLOAD_MNIST Downloads and uncompresses the four MNIST files.
%   download_mnist(folder) saves the files in FOLDER (default 'mnist').
%   Files that are already there are not downloaded again.
if nargin < 1, folder = 'mnist'; end
names = {'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'};
% Mirrors of the original files, tried in this order
mirrors = {'https://ossci-datasets.s3.amazonaws.com/mnist/', ...
           'https://storage.googleapis.com/cvdf-datasets/mnist/', ...
           'https://raw.githubusercontent.com/fgnt/mnist/master/'};
if ~exist(folder, 'dir'), mkdir(folder); end
for i = 1:numel(names)
    target = fullfile(folder, names{i});
    if exist(target, 'file'), continue; end
    done = false;
    for j = 1:numel(mirrors)
        url = [mirrors{j} names{i} '.gz'];
        try
            fprintf('Downloading %s\n', url);
            if exist('websave', 'file')
                websave([target '.gz'], url);     % Matlab
            else
                urlwrite(url, [target '.gz']);    % GNU Octave
            end
            gunzip([target '.gz']);
            if exist([target '.gz'], 'file'), delete([target '.gz']); end
            done = true;
            break
        catch
            if exist([target '.gz'], 'file'), delete([target '.gz']); end
        end
    end
    if ~done
        error('Could not download %s. Download it by hand and save it in "%s".', names{i}, folder);
    end
end
end
