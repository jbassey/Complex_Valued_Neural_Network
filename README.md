# Complex_Valued_Neural_Network
Contains materials and codes for building neural network that handle complex numbers

There are two main approaches. One approach follows the same approach as with current real-valued neural networks 
and uses Clifford Algebra for complex domain back propagation.

the second approach follows the approach by Igor Aizenberg in multi-layer multi-valumahmood codeed neuron (MLMVN) neural networks.
which make use of the phase of the weighed sum, and does not need differentiation during backpropagation.

The MLMVN approach is tested using IRIS and MNIST datasets and using differrent error criteria: MSE, RMSE, error rate, Accuracy etc.

There is also a complex valued convolutional neural network implementation applied to MNIST dataset.

TODO: Benchmarking of all models against their real-valued counterpart.
mahmood code
