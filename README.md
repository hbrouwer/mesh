```
         ______
    __---   )  --_      - - - - - - - - - - - - - - - - - - - - - - - - -
  --       /      -_
 /     o  (         )   Mesh: https://github.com/hbrouwer/mesh
(     o   ____  o    )  (c) 2012-2022 Harm Brouwer <me@hbrouwer.eu>
(    o _--     o      )
 (____/       o _____)  Licensed under the Apache License, Version 2.0
      (____  ---  )
           \ \-__/      - - - - - - - - - - - - - - - - - - - - - - - - -
```

# Mesh

Mesh is a lightweight and versatile artificial neural network simulator,
primarily designed as a general-purpose backpropagation simulator with
flexibility and extensibility in mind.

[![build](https://github.com/hbrouwer/mesh/actions/workflows/build.yml/badge.svg)](https://github.com/hbrouwer/mesh/actions/workflows/build.yml)

## Features

Mesh comes with support for:

* **Different architectures:** feed forward networks (`ffn`), simple
  recurrent networks (`srn`) (Elman, 1990), and recurrent neural networks
  (`rnn`);

* **Training algorithms:** backpropagation (`bp`) (Rumelhart et al., 1986a)
  and backpropagation through time (`bptt`) (Rumelhart et al., 1986b);

* **Weight update algorithms:** steepest/gradient descent (`steepest`),
  bounded steepest descent (`bounded`) (Rohde, 2002), four flavours of
  resilient propagation (`rprop+`, `rprop-`, `irprop+`, and `irprop-`) (Igel
  & Husken, 2000), quickprop (`qprop`) (Fahlman, 1988), and delta-bar-delta
  (`dbd`) (Jacobs, 1988);

* **Activation functions:** logistic (sigmoid)  (`logistic`), bipolar
  sigmoid (`bipolar_sigmoid`), softmax (`softmax`), hyperbolic tangent
  (`tanh`), linear (`linear`), (bounded) recitified linear (`relu`), leaky
  rectified linear (`leaky_relu`), and exponential linear (`elu`);

* **Error functions:** sum squared error (`sum_of_squares`) , cross entropy
  error (`cross_entropy`), and Kullback-Leibler divergence (`divergence`);

* **Weight randomization algorithms:** gaussian (`gaussian`), uniform range
  (`range`), Nguyen-Widrow (`nguyen_windrow`) (Nguyen & Widrow, 1990),
  Fan-In (`fan_in`), and binary (`binary`).

* **Multithreading** (through [OpenMP](https://www.openmp.org/));

* A module for navigating **propositional meaning spaces**: meaning spaces
  derived from the Distributed Situation-state Space (DSS) model (Frank et
  al., 2003) and Distributional Formal Semantics (DFS) (Venhuizen et al.,
  2021);

* A module for modeling **electrophysiological correlates**: the N400 and
  P600 components of the Event-Related brain Potentials (ERP) signal
  (Brouwer, 2014; Brouwer et al., 2017);

* **Pretty printing** of vectors and matrices (through ANSI escape codes);

* And finally, it is **dependency-free**: you only need a C99-compliant
  (and for multithreading OpenMP-enabled) compiler to build Mesh.

## Why Mesh?

> "What I cannot create, I do not understand“ -Richard Feynman

**Is Mesh the new PyTorch or TensorFlow?** No, Mesh is **not** the next
[PyTorch](https://pytorch.org/) or
[TensorFlow](https://www.tensorflow.org/). Mesh is a simulator that focuses
on traditional connectionist / Parallel Distributed Processing (PDP)
architectures and learning algorithms (in the tradition of simulators like
[Tlearn](https://crl.ucsd.edu/innate/tlearn.html) and
[Lens](https://ni.cmu.edu/~plaut/Lens/Manual/), among others). It was
developed along with my PhD dissertation in cognitive neuroscience
([Brouwer, 2014](https://hbrouwer.github.io/papers/Brouwer2014ElectrophysiologyLanguage.pdf)),
in which I used it to build a neurocomputational model of the
electrophysiology of language comprehension (Brouwer et al., 2017). Mesh is
a one-man show; I started developing Mesh before the deep learning
revolution, and hence before large-scale deep learning frameworks like
PyTorch and TensorFlow, backed by
[Facebook](https://www.facebook.com/)/[Meta](https://about.facebook.com/meta)
and [Google](https://www.google.com/), respectively, became available.

**I learned a lot implementing Mesh:** I built Mesh from scratch using
classical papers as technical references. I have waded through many slides,
books, and websites, in order to put the different pieces together. Again
this was prior to the deep learning revolution, and hence prior to the
wealth of information that has become available over the last few years.
Indeed, as the late Jeff Elman (author of
[Tlearn](https://crl.ucsd.edu/innate/tlearn.html)) pointed out to me:
I learned an enormous amount about neural networks by implementing Mesh, and
for that reason alone it has been worthwhile.

**Mesh is lightweight yet versatile:** We use Mesh on a daily basis to run
cognitive models of human language comprehension. In Brouwer et al. (2017),
for instance, it is used to model electrophysiological correlates of online
comprehension, and in Venhuizen et al. (2021) we use it to navigate
a propositional meaning space from Distributional Formal Semantics (DFS).
Moreover, as Mesh is fully command driven, it is also ideal for teaching
connectionistm/PDP: it has for instance been used in a course on
[Connectionist Language
Processing](https://hbrouwer.github.io/courses/clp21/) at [Saarland
University](https://www.uni-saarland.de/start.html).

# Building and running Mesh

Building Mesh should be as straightforward as:

```
$ cmake .
$ make
```

You can then run Mesh as:

```
$ ./mesh
Mesh, version 1.0.0: https://github.com/hbrouwer/mesh (`?` for help)
+ [ OpenMP ]: 10 processor(s) available (10 thread(s) max)
+ [ OpenMP ]: Dynamic schedule (chunk size: 1)
```

Note that as Mesh is fully command driven, it is recommended to use
[rlwrap](https://github.com/hanslub42/rlwrap).

# Usage

```
$ ./mesh --help
Mesh, version 1.0.0: https://github.com/hbrouwer/mesh (`?` for help)
+ [ OpenMP ]: 10 processor(s) available (10 thread(s) max)
+ [ OpenMP ]: Static schedule (chunk size: 0)

Usage: mesh [file | option]

[file]:
Mesh will load and run the specified script file.

[option]:
`--help`                         Show this help message
`--version`                      Show version information

When no arguments are specified, Mesh will start in CLI mode.

```

# Documentation and Examples

Documentation is available within Mesh, by typing `?` or `help`, as well as
[here](docs/welcome.md) in Markdown format. 

Tutorials are available in the
[mesh-examples](https://github.com/hbrouwer/mesh-examples) repository, and
cover networks implementing simple boolean functions, as well as various
psycholinguistic connectionist models.

# Multithreading

When Mesh is compiled with `-DOPENMP=ON` (default), multithreading is
implemented through [OpenMP](https://www.openmp.org/), and controlled with
its [environment
variables](https://www.openmp.org/spec-html/5.0/openmpch6.html). For
example, the following limits the number of threads to `2`, and enables
`auto` scheduling:

```
$ OMP_NUM_THREADS=2 OMP_SCHEDULE=auto ./mesh
Mesh, version 1.0.0: https://github.com/hbrouwer/mesh (`?` for help)
+ [ OpenMP ]: 10 processor(s) available (2 thread(s) max)
+ [ OpenMP ]: Auto schedule
  [:>
```

Note that in order to use multithreading, you need to activate it in Mesh as
well for a given network using `toggleMultithreading` (default: off):

```
$ OMP_NUM_THREADS=2 mesh plaut.mesh
Mesh, version 1.0.0: https://github.com/hbrouwer/mesh (`?` for help)
+ [ OpenMP ]: 10 processor(s) available (2 thread(s) max)
+ [ OpenMP ]: Dynamic schedule (chunk size: 1)
...
> Loaded file                    [ plaut.mesh ]
  [plaut:train> toggleMultithreading
> Toggled multithreading         [ on ]
```

You can inspect the multithreading status of an active network using
`inspect`:

```
  [plaut:train> inspect
| Name:                          plaut
| Type:                          ffn
...
| Multithreading enabled:        true
| Processor(s) available:        10
| Maximum #threads:              2
| Schedule:                      dynamic
| Chunk size                     1
```

To compile Mesh without multithreading, pass the flag `-DOPENMP=OFF` to
CMake.

**Warning:** If multithreading is enabled, Mesh will always distribute
computations among the available threads. Depending on network size,
however, this may not always lead to improved performance over
single-threaded execution. In fact, the overhead of multithreading may even
damage performance. 

# Fast exponentiation 

Mesh implements Nicol N. Schraudolph's fast, compact approximation of the
exponential function (Schraudolph, 1999). This feature is disabled by
default, but can be enabled by passing the flag `-DFAST_EXP=ON` to CMake. If
enabled, Mesh will report this on startup:

```
$ ./mesh
Mesh, version 1.0.0: https://github.com/hbrouwer/mesh (`?` for help)
+ [ FastExp ]: Using Schraudolph's exp() approximation (c: 60801)
...
  [:>
```

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

Schraudolph, N. N. (1999). A fast, compact approximation of the exponential
function. Neural Computation, 11, 854-862.

Venhuizen, N. J., Hendriks, P., Crocker, M. W., and Brouwer, H. (in press).
Distributional Formal Semantics. *Information and Computation*. arXiv
preprint arXiv:2103.01713
