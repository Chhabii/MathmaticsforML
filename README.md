# MathmaticsforML

### Introduction & Motivation
- We represent data as vectors.
- We choose an appropriate model, either using the probabilistic or opti-
mization view.
- We learn from available data by using numerical optimization methods
with the aim that the model performs well on data not used for training.

![Foundations & Four pillars of Machine Learning](image-1.png)

#### Foundations: 

- We represent numerical data as vectors and represent a table of such data as a matrix. The study of vectors and matrices is called linear algebra.
- Given two vectors representing two objects in the real world, we want
to make statements about their similarity. The idea is that vectors that
are similar should be predicted to have similar outputs by our machine
learning algorithm. To formalize the idea of similarity between vectors, we need to introduce operations that take two vectors as
input and return a numerical value representing their similarity. The con-
struction of similarity and distances is central to analytic geometry.

- Matrix Decomposition: Some operations on matrices are extremely
useful in machine learning, and they allow for an intuitive interpretation
of the data and more efficient learning.

-  when we look at data, it's often like looking at a blurry picture of what's really happening. We use machine learning to try to figure out what's going on behind the blur. But to do this well, we need a way to measure how blurry our picture is—that's what "noise" means here. Also, sometimes we want to know how sure we are about our guesses. For example, if we predict the weather will be sunny tomorrow, how confident are we in that prediction? This idea of measuring our confidence is what we call uncertainty. And when we talk about uncertainty in predictions, we're diving into the world of probability theory, which is like a toolkit for understanding and dealing with uncertainty.

- To train machine learning models, we typically find parameters that
maximize some performance measure. Many optimization techniques re-
quire the concept of a gradient, which tells us the direction in which to
search for a solution.

#### Four Pillars:
- Linear regression, where our
objective is to find functions that map inputs x ∈ RD to corresponding ob-
served function values y ∈ R, which we can interpret as the labels of their
respective inputs. We will discuss classical model fitting (parameter esti-
mation) via maximum likelihood and maximum posteriori estimation,
as well as Bayesian linear regression, where we integrate the parameters
out instead of optimizing them.

- Dimensionality reduction using principal component analysis. The key objective of dimensionality reduction is to find a compact, lower-dimensional representation
of high-dimensional data x ∈ RD , which is often easier to analyze than
the original data. Unlike regression, dimensionality reduction is only concerned about modeling the data – there are no labels associated with a
data point x.

- The objective of density estimation is to find a probability distribution that de-
scribes a given dataset. We will focus on Gaussian mixture models for this
purpose, and we will discuss an iterative scheme to find the parameters of
this model. As in dimensionality reduction, there are no labels associated
with the data points x ∈ RD . However, we do not seek a low-dimensional
representation of the data. Instead, we are interested in a density model
that describes the data.


