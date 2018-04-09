# Mesh

Mesh is an artificial neural network simulator, primarily designed as a
fast, general-purpose backpropagation simulator with flexibility and
extensibility in mind. It comes with support for:

* Various architectures: *Feed Forward Networks (FFNs)*, *Simple Recurrent
  Networks (SRNs)* (Elman, 1990), and *Recurrent Neural Networks (RNNs)*;

* Training algorithms: *Backpropagation* (Rumelhart et al., 1986a) and
  *Backpropagation Through Time* (Rumelhart et al., 1986b);

* Weight update algorithms: *Steepest (gradient) descent*, *Bounded steepest
  descent* (Rohde, 2002), four flavours of Resilient propagation: *Rprop+*,
  *Rprop-*, *iRprop+*, *iRprop-*, *Quickprop*, (Fahlman, 1988) and
  *Delta-Bar-Delta* (Jacobs, 1988);

* Activation functions: *Logistic*, *Bipolar sigmoid*, *Softmax*,
  *Hyperbolic tangent (tanh)*, *Linear*, *Recitified Linear Unit* (ReLU),
  *Leaky ReLU*, *Binary ReLU* (bounded ReLU), and *Softplus* (smooth ReLU);

* Error functions: *Sum of squares*, *Cross entropy*, and *Divergence*;

* Weight randomization algorithms: *Gaussian*, *Range*, *Nguyen-Widrow*
  (Nguyen & Widrow, 1990), *Fan-In*, and *Binary* randomization.

* Multi-threading (through OpenMP);

## Is Mesh the new Caffe2 or TensorFlow?

No, Mesh is not, nor will ever be the next Caffe or TensorFlow. Mesh is a
simulator that focuses on traditional connectionist/Parallel Distributed
Processing (PDP) architectures and learning algorithms. It was developed
along with my PhD dissertation in cognitive science (Brouwer, 2014), in
which I used it to build a neurocomputational model of the electrophysiology
of language comprehension (Brouwer et al., 2017). Crucially, I started
developing Mesh before deep learning took over the world, and hence before
large-scale deep learning frameworks like [Caffe2](https://caffe2.ai) and
[TenserFlow](https://www.tensorflow.org/) became available. Indeed, if you
are interested in deep learning, you are better of with such a widely
supported framework. Remember, Mesh is a one-man show, whereas Caffe2 and
TensorFlow are backed by Facebook and Google, respectively. 

## So, why Mesh? Well, here's why:

**I learned a lot. Like really, a lot**: I built Mesh from scratch using
classical papers as technical references (see below). I have waded through
many slides, books, and websites, in order to put the different pieces
together (note that this was prior to the deep learning revolution, and
hence prior to the wealth of information that has become available over the
last few years).  I implemented various flavours of the backpropagation
algorithm (e.g., Rprop, Quickprop, Delta-Bar-Delta), as well as
backpropagation through time. In sum, I learned an enormous amount about
neural networks, and for that reason alone building Mesh has been
worthwhile.

**It does what I want it to do, in the way I want it to do it**: We use Mesh
on a daily basis to run cognitive models of human language comprehension. 

**It is completely dependency free**: You only need a C99 compiler to
compile Mesh.

**Maybe it is of use to someone else**:

# References

Brouwer, H. (2014). The Electrophysiology of Language Comprehension: A
Neurocomputational Model. PhD thesis, University of Groningen.

Brouwer, H., Crocker, M. W., Venhuizen, N. J., and Hoeks, J. C. J. (2017). A
Neurocomputational Model of the N400 and the P600 in Language Processing.
*Cognitive Science*, 41(S6), 1318-1352.

Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2),
179-211.

Fahlman, S. E. (1988). An empirical study of learning speed in
back-propagation networks. Technical report CMU-CS-88-162. School of
Computer Science, Carnegie Mellon University, Pittsburgh, PA 15213.

Igel, C., & Husken, M. (2000). Improving the Rprop Algorithm. Proceedings of
the Second International Symposium on Neural Computation, NC'2000, pp.
115-121, ICSC, Academic Press, 2000.

Jacobs, R. A. (1988). Increased Rates of Convergence Through Learning Rate
Adapation. *Neural Network*s, 1, 295-307.

Nguyen, D. & Widrow, B. (1990). Improving the learning speed of 2-layer
neural networks by choosing initial values of adaptive weights. Proceedings
of the International Joint Conference on Neural Networks (IJCNN), 3:21-26,
June 1990.

Rohde, D. L. T. (2002). A connectionist model of sentence comprehension and
production. PhD thesis, Carnegie Mellon University.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986a). Learning
representations by back-propagating errors. *Nature*, 323, 553-536.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986b). Learning
internal representations by error propagation. In: D. E. Rumelhart & J. L.
McClelland (Eds.), *Parallel distributed processing: Explorations in the
microstructure of cognition, Volume 1: Foundations,* pp. 318-362, Cambridge,
MA: MIT Press.
