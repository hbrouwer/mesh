# Mesh

Mesh is an artificial neural network simulator, primarily designed as a
fast, general-purpose backpropagation simulator with flexibility and
extensibility in mind. It comes with support for:

* Different architectures: ```Feed Forward Networks (FFNs)```, ```Simple
  Recurrent Networks (SRNs)``` (Elman, 1990), and ```Recurrent Neural
  Networks (RNNs)```;

* Training algorithms: ```Backpropagation (BP)``` (Rumelhart et al., 1986a)
  and ```Backpropagation Through Time (BPTT)``` (Rumelhart et al., 1986b);

* Weight update algorithms: ```Steepest (gradient) descent```, ```Bounded
  steepest descent``` (Rohde, 2002), four flavours of Resilient propagation:
  ```Rprop+```, ```Rprop-```, ```iRprop+```, ```iRprop-``` (Igel & Husken,
  2000), ```Quickprop```, (Fahlman, 1988) and ```Delta-Bar-Delta``` (Jacobs,
  1988);

* Activation functions: ```Logistic```, ```Bipolar sigmoid```,
  ```Softmax```, ```Hyperbolic tangent (tanh)```, ```Linear*```
  ```Recitified Linear Unit (ReLU)```, ```Leaky ReLU```, ```Binary ReLU
  (bounded ReLU)```, and ```Softplus (smooth ReLU)```;

* Error functions: ```Sum of squares```, ```Cross entropy```, and
  ```Divergence```;

* Weight randomization algorithms: ```Gaussian```, ```Range```,
  ```Nguyen-Widrow``` (Nguyen & Widrow, 1990), ```Fan-In```, and
  ```Binary``` randomization.

* Multithreading (through [OpenMP](https://www.openmp.org/));

* A module for navivating semantic spaces: ```Distributed Situation-state
  Spaces (DSS)``` (Frank et al., 2003, 2009) and ```Distributional Formal
  Semantics (DFS)``` (Venhuizen et al., 2021);

* A module for modeling electrophysiological correlates: ```N400``` and
  ```P600``` components of the Event-Related Potentials (ERP) signal
  (Brouwer, 2014; Brouwer et al., 2017);

* Pretty printing of vectors and matrices (through ANSI escape codes);

* And finally, it is completely self-contained: it is fully dependency free
  and you only need a C99 compiler to build Mesh.

## What MESH is *not*

Mesh is **not** the next [PyTorch](https://pytorch.org/) or
[TensorFlow](https://www.tensorflow.org/). Mesh is a simulator that focuses
on traditional connectionist/Parallel Distributed Processing (PDP)
architectures and learning algorithms. It was developed along with my PhD
dissertation in cognitive science (Brouwer, 2014), in which I used it to
build a neurocomputational model of the electrophysiology of language
comprehension (Brouwer et al., 2017). Crucially, I started developing Mesh
before the deep learning revolution, and hence before large-scale deep
learning frameworks like PyTorch and became available. Indeed, if you are
interested in deep learning, you are probably better off with such a widely
supported framework. Remember, Mesh is a one-man show, whereas PyTorch and
TensorFlow are backed by Facebook and Google, respectively.

## So, why Mesh?

**I learned a lot**: I built Mesh from scratch using
classical papers as technical references (see below). I have waded through
many slides, books, and websites, in order to put the different pieces
together (again this was prior to the deep learning revolution, and
hence prior to the wealth of information that has become available over the
last few years).  I implemented various flavours of the backpropagation
algorithm (e.g., Rprop, Quickprop, Delta-Bar-Delta), as well as
backpropagation through time. In sum, I learned an enormous amount about
neural networks, and for that reason alone building Mesh has been
worthwhile.

**It does what I want it to do, in the way I want it to do it**: We use Mesh
on a daily basis to run cognitive models of human language comprehension
(e.g., it is used in Brouwer et al., 2017 as well as Venhuizen et al.,
2021).

**It is great for teaching**: Mesh is fully command driven, and hence ideal
for teaching (e.g., we use it to teach [Connectionist Language
Processing](https://hbrouwer.github.io/courses/clp21/)). 

# Building and running Mesh

Building Mesh should be as straightforward as:

```
$ cmake .
$ make
```

Note that you need an [OpenMP](https://www.openmp.org/)-enabled compiler if
you want to enable multithreading. Passing the flag ```-DOPENMP=OFF``` to
CMake disables multithreading.

You can then run Mesh as:

```
$ ./mesh
Mesh, version 0.1.0: https://github.com/hbrouwer/mesh (? for help)
+ [ OpenMP ]: 10 processor(s) available (10 thread(s) max)
+ [ OpenMP ]: Static schedule (chunk size: 0)
  [:>
```

As Mesh is fully command driven, it is recommended to use
[rlwrap](https://github.com/hanslub42/rlwrap).

# Documentation

**TODO.** Minimal documentation is available by typing ```?``` or ```help```
within Mesh. Also, see the [example
networks](https://github.com/hbrouwer/mesh-examples).

# Examples

Various psycholinguistic connectionist models are available as [example
networks](https://github.com/hbrouwer/mesh-examples).

# References

Brouwer, H. (2014). The Electrophysiology of Language Comprehension:
A Neurocomputational Model. PhD thesis, University of Groningen.

Brouwer, H., Crocker, M. W., Venhuizen, N. J., and Hoeks, J. C. J. (2017).
A Neurocomputational Model of the N400 and the P600 in Language Processing.
*Cognitive Science, 41*(S6), 1318-1352.

Elman, J. L. (1990). Finding structure in time. *Cognitive Science, 14*(2),
179-211.

Fahlman, S. E. (1988). An empirical study of learning speed in
back-propagation networks. Technical report CMU-CS-88-162. School of
Computer Science, Carnegie Mellon University, Pittsburgh, PA 15213.

Frank, S. L., Koppen, M., Noordman, L. G., & Vonk, W. (2003). Modeling
knowledge-based inferences in story comprehension. *Cognitive Science,
27*(6), 875–910. doi:10.1207/s15516709cog2706_3

Frank, S. L., Haselager, W. F., & van Rooij, I. (2009). Connectionist
semantic systematicity. *Cognition, 110*(3), 358–379.
doi:10.1016/j.cognition.2008.11.013

Igel, C., & Husken, M. (2000). Improving the Rprop Algorithm. Proceedings of
the Second International Symposium on Neural Computation, NC'2000, pp.
115-121, ICSC, Academic Press, 2000.

Jacobs, R. A. (1988). Increased Rates of Convergence Through Learning Rate
Adapation. *Neural Networks, 1*, 295-307.

Nguyen, D. & Widrow, B. (1990). Improving the learning speed of 2-layer
neural networks by choosing initial values of adaptive weights. Proceedings
of the International Joint Conference on Neural Networks (IJCNN), 3:21-26,
June 1990.

Rohde, D. L. T. (2002). A connectionist model of sentence comprehension and
production. PhD thesis, Carnegie Mellon University.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986a). Learning
representations by back-propagating errors. *Nature, 323*, 553-536.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986b). Learning
internal representations by error propagation. In: D. E. Rumelhart & J. L.
McClelland (Eds.), *Parallel distributed processing: Explorations in the
microstructure of cognition, Volume 1: Foundations,* pp. 318-362, Cambridge,
MA: MIT Press.

Venhuizen, N. J., Hendriks, P., Crocker, M. W., and Brouwer, H. (in press).
Distributional Formal Semantics. *Information and Computation*. arXiv
preprint arXiv:2103.01713
