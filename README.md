# Backpropagation from scratch

A small training engine for fully connected neural networks, written from scratch in
MATLAB and Python/NumPy: backpropagation plus mini-batch gradient descent, with no deep
learning library.

This is the companion code for a tutorial by Javier Sánchez (Universidad de Las Palmas
de Gran Canaria), currently in preparation. The tutorial builds the engine step by step:
it starts with the XOR problem on a 2-2-1 network and a single sample, moves to the
matrix formulation for mini-batches, and ends with one generic function for networks
with any number of layers.

The Python code is a line-by-line translation of the MATLAB code.

## Contents

| File (in `code/`) | Description |
|---|---|
| `XOR_FNN_SGD.m` / `.py` | XOR with a 2-2-1 network, one sample at a time (SGD). |
| `XOR_FNN_batch.m` / `.py` | XOR with the full batch in matrix form. |
| `Backpropagation.m` / `.py` | Generic training function for fully connected networks with L layers. |
| `predict.m` | Forward pass of a trained network (in Python, inside `Backpropagation.py`). |
| `read_mnist.m` | Reader for the MNIST files (in Python, inside `Backpropagation.py`). |
| `test_Backpropagation.m` / `.py` | Experiments: two concentric rings and MNIST. |

## Requirements

- **MATLAB**: base functions only, no toolboxes. The code also runs in GNU Octave.
- **Python 3** with NumPy and Matplotlib: `pip install -r requirements.txt`

## Usage

Run the programs from the `code/` folder.

```
cd code
python XOR_FNN_SGD.py
python XOR_FNN_batch.py
python test_Backpropagation.py
```

In MATLAB or Octave, change to `code/` and type the script name, for example
`XOR_FNN_SGD`.

The generic function has the same interface in both languages:

```
[W, b, loss] = Backpropagation(X, Y, Neurons, batch_size, Nepochs, alpha, sigma, dsigma)
```

`X` and `Y` hold one sample per column, and `Neurons` lists the number of neurons in
each layer, for example `[784 30 10]`. The last four arguments are optional (defaults:
1000 epochs, learning rate 0.5, sigmoid activation). The function returns the weights
and biases of each layer and the mean loss of each epoch.

### MNIST

The MNIST files are not included. Download them, uncompress them and place them in
`code/mnist/` with these names:

```
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

### A note on XOR

Weights are initialized at random, and a 2-2-1 network has just enough capacity for
XOR, so some runs do not solve it. This is expected and is discussed in the tutorial;
run the script again.

## Citation

See `CITATION.cff`. The reference to the tutorial will be added here once it is
available.

## License

MIT (see `LICENSE`).
