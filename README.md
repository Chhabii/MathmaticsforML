# MathmaticsforML

### 1) Introduction & Motivation
- We represent data as vectors.
- We choose an appropriate model, either using the probabilistic or optimization view.
- We learn from available data by using numerical optimization methods
with the aim that the model performs well on data not used for training.

![Foundations & Four pillars of Machine Learning](https://github.com/Chhabii/MathmaticsforML/assets/60286478/b990b263-28ba-46c0-b5c1-ce99f0a88d79)


#### Foundations: 

- We represent numerical data as vectors and represent a table of such data as a matrix. The study of vectors and matrices is called linear algebra.
- Given two vectors representing two objects in the real world, we want
to make statements about their similarity. The idea is that vectors that
are similar should be predicted to have similar outputs by our machine
learning algorithm. To formalize the idea of similarity between vectors, we need to introduce operations that take two vectors as
input and return a numerical value representing their similarity. The construction of similarity and distances is central to analytic geometry.

- Matrix Decomposition: Some operations on matrices are extremely
useful in machine learning, and they allow for an intuitive interpretation
of the data and more efficient learning.

-  when we look at data, it's often like looking at a blurry picture of what's really happening. We use machine learning to try to figure out what's going on behind the blur. But to do this well, we need a way to measure how blurry our picture is—that's what "noise" means here. Also, sometimes we want to know how sure we are about our guesses. For example, if we predict the weather will be sunny tomorrow, how confident are we in that prediction? This idea of measuring our confidence is what we call uncertainty. When we talk about uncertainty in predictions, we're diving into the world of probability theory, which is like a toolkit for understanding and dealing with uncertainty.

- To train machine learning models, we typically find parameters that
maximize some performance measure. Many optimization techniques re-
quire the concept of a gradient, which tells us the direction in which to
search for a solution.

#### Four Pillars:
- Linear regression, where our
objective is to find functions that map inputs x ∈ RD to corresponding observed function values y ∈ R, which we can interpret as the labels of their
respective inputs. We will discuss classical model fitting (parameter estimation) via maximum likelihood and Maximum a posteriori estimation,
as well as Bayesian linear regression, where we integrate the parameters
out instead of optimizing them.

- Dimensionality reduction using principal component analysis. The key objective of dimensionality reduction is to find a compact, lower-dimensional representation
of high-dimensional data x ∈ RD, which is often easier to analyze than
the original data. Unlike regression, dimensionality reduction is only concerned with modeling the data – there are no labels associated with a
data point x.

- The objective of density estimation is to find a probability distribution that de-
scribes a given dataset. We will focus on Gaussian mixture models for this
purpose, and we will discuss an iterative scheme to find the parameters of
this model. As in dimensionality reduction, there are no labels associated
with the data points x ∈ RD. However, we do not seek a low-dimensional
representation of the data. Instead, we are interested in a density model
that describes the data.

### 2.) Linear Algebra

- Linear algebra is the study of vectors and certain rules to manipulate vectors.
- vectors are special objects that can be added together and multiplied by scalars to produce another object of the same kind.
#### Geometric vectors:
- Geometric vectors are directed segments, which can be drawn (at least in two dimensions). Two geometric vectors x and, y can be added, such that x + y = z
is another geometric vector. Furthermore, multiplication by a scalar λx, λ ∈ R, is also a geometric vector. In fact, it is the original vector
scaled by λ.

#### Polynomials are also vectors:
- Two polynomials can be added together, which results in another polynomial and they can be multiplied by a scaler $\lambda \in R$ and the result is polynomial as well.
- Therefore, polynomials are (rather unusual) instances of vectors.
Note that polynomials are very different from geometric vectors. While
geometric vectors are concrete “drawings”, polynomials are abstract
concepts. However, they are both vectors.

#### Audio signals are vectors:
Audio signals are represented as a series of
numbers. We can add audio signals together, and their sum is a new
audio signal. If we scale an audio signal, we also obtain an audio signal.
Therefore, audio signals are a type of vector, too.


📹 [3B1B](https://youtu.be/fNk_zzaMoSs?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)





