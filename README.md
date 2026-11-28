# A neural network in C, with no ML library.

The point is to see the whole thing: where the weights live, how a forward pass is only matrix multiplies and a sigmoid, and how a cost can push those weights without a framework doing it for you. The example is XOR, four rows, one output.

## What runs today

`neurnet.h` is the library. A network is an architecture array. `{2, 2, 1}` means two inputs, a hidden layer of two, and one output. Each layer has a weight matrix, a bias row, and an activation row.

Training, as of this version:

1. Forward pass. Multiply, add the bias, apply the sigmoid.
2. Cost. Mean squared error against the expected outputs.
3. Finite differences. Nudge one weight or bias, measure the cost again, and divide by the nudge. That gradient is a measurement, not automatic differentiation.
4. Learn. Subtract `rate` times that gradient from the parameter.

`abst.c` trains the XOR net this way and prints the cost and the four predictions. `intro.c` is the same idea written out by hand for a fixed `2-2-1` layout, before the general allocator. `grad.tex` is notes toward the real derivative. The program does not use that derivative yet.

Backpropagation, and a fuller training stack, is the work in progress. Nothing here is a benchmark.

## Build

Requires `clang` and `libm`.

```sh
./build.sh
