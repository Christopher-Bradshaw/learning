# Neural Net

A simple implementation of a NN. Currently can only do fully connected layers, with ReLU actuvation and mean squared error loss.

## TODO

Batch normalization
Demo a simple use case
Convolutions?

## Backprop

Backprop is implemented manually (i.e. each type of layer needs to describe how to compute gradients). I did this rather than use some sort of autograd to force me to think about how backprop works...

### Backprop notes

Let's say we have a network made up of layers (l1, l2, ..., ln) each of which map xi = li(xi-1), for example x1 = l1(x0) -> the output from the first layer.
Note that all of these xn are vectors of some (potentially varying between layers) size.
We'll also say that xn, the output from the last layer is the loss, L.

We can write that (for a 3 layer network) L = l3(l2(l1(x0))), where any of the layers might also include a weight argument, wi (where each wi is a vector). These weights can be thought of as properties of the layers.
We want to minimze the loss by modifying w. However, to do this we need to know how each w affects the loss.

Let's start at the last layer. Let's assume it has weights
L = l3(x2, w3)
We can simple compute dL/dw3. This is a vector that tells us how to change w3 to maximally increase L. It is the gradient of L.
We want to decrease L. To do that we should move in the negative direction!

Let's assume the last layer doesn't have weights. You might not think we need to do anything with it (why would we? the whole reason for doing backprop is to update weights). But, let's try update the second last layer.

L = l3(x2) = l3(l2(x1, w2)).
By the chain rule, dL/dw2 = d/dx2 l3(x2) * d/dw2 x2

So, for each layer we need to know how the output changes wrt to the weights.
And how the output changes wrt to the input.
The gradient is then:

dL/dw_i = d/dw x_i * (dli+1/d_xi * dli+2/d_xi+1 * ...)
