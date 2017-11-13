# Mesh

Mesh is an artificial neural network simulator, primarily designed as a
fast, general-purpose backpropagation simulator with flexibility and
extensibility in mind. It supports:

* Various architectures: Feed Forward Networks (FFNs), Simple Recurrent
  Networks (SRNs), and Recurrent Neural Networks (RNNs);
* Training algorithms: Backpropagation (BP) and Backpropagation Through Time (BPTT);
* Weight update algorithms: Steepest Descent, Bounded Steepest Descent, four
  flavours of Resilient Propagation (Rprop+, Rprop-, iRprop+, iRprop-),
  Quickprop, and Delta-Bar-Delta (DBD);
* Activation functions: Binary sigmoid (logistic), Bipolar sigmoid, Softmax,
  Hyperbolic tangent (tanh), Linear, Step, and Softplus (smooth ReLU);
* Error functions: Sum of squares, Cross entropy, Divergence;
* Weight randomization algorithms: Gaussian, Range, Nguyen-Widrow, Fan-In, and Binary.
* Multi-threading (through OpenMP);

## Why another Neural Network simulator/toolkit/library?

Mesh is a simulator that focuses on traditional connectionist or Parallel
Distributed Processing (PDP) neural network architectures and learning
algorithms. 

Mesh was developed as part of my PhD thesis on connectionist modeling of the
electrophysiology of language comprehension (Brouwer, 2014). Development
started before the deep learning revolution, and hence before large-scale
deep learning frameworks like Caffe and Tenserflow became available.

In
fact, if you are interested in contemporary deep learning, Mesh is probably
not for you, as it focuses on traditional connectionist or
Parallel Distributed Processing (PDP).
