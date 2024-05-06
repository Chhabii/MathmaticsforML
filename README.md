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

#### Elements of $R_{n}$ (tuples of n real numbers) are vectors:
$R_{n}$ is more abstract than polynomials, and it is the concept we focus on. For instance,


$\alpha = [1,2,3] in R$

is an example of triplets of numbers.Adding two vectors $(a, b) in R_{3}$  
component-wise results in another vector: a + b = c ∈ Rn . Moreover,
multiplying a ∈ Rn by λ ∈ R results in a scaled vector λa ∈ Rn .


- One major idea in mathematics is the idea of “closure”. This is the question: What is the set of all things that can result from my proposed operations?
In the case of vectors: What is the set of vectors that can result by
starting with a small set of vectors, and adding them to each other and
scaling them? This results in a vector space.
- The concept of a vector space and its properties underlie much of machine learning. The
concepts introduced in this chapter are summarized in Figure below:

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/2f93ad1f-fb76-46c9-b5d1-fd9dbd5e51ac)
 **Vector:** A vector is an element of a vector space that has both magnitude and direction. It can be composed of other vectors, and it is central to the concept of a vector space.
 
**Matrix:** A matrix is a rectangular array of numbers and can represent vectors, linear transformations, and systems of linear equations. Matrices can also be composed to represent sequential linear transformations.

**Vector Space:** This is a collection of vectors that can be added together and multiplied by scalars (numbers) to produce another vector within the same space. It has a property of closure under addition and scalar multiplication, making it an Abelian group with respect to addition.

**Group:** A group is a mathematical concept that includes a set of elements and an operation (like addition), fulfilling certain conditions like closure, associativity, identity element, and inverse elements. A vector space is an example of an Abelian group where the operation is vector addition. 
📹 [Group & Abelian Group](https://youtu.be/8TjYHK804mU)

**Linear/Affine Mapping:** This is a function between vector spaces that preserves vector addition and scalar multiplication. Such mappings can be represented by matrices and are fundamental in understanding transformations in geometry.

📹 [3B1B: Linear transformations and matrices ](https://youtu.be/kYB8IZa5AuE)

**Linear Independence:** This concept involves vectors that do not express any vector in the set as a linear combination of the others. Linear independence is critical for defining the basis of a vector space.


**Basis:** A basis of a vector space is a set of linearly independent vectors that span the entire vector space. Every vector in the space can be expressed uniquely as a linear combination of basis vectors.
📹 [Linear combinations, span, and basis vectors](https://youtu.be/k7RM-ot2NWY?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)


**System of Linear Equations:** Systems of linear equations play a central part in linear algebra. Many
problems can be formulated as systems of linear equations, and linear
algebra gives us the tools for solving them.

**Matrix Inverse:** This is a matrix that, when multiplied with the original matrix, yields the identity matrix. The inverse of a matrix is particularly useful in solving systems of linear equations.

**Gaussian Elimination:** This is a systematic method for solving systems of linear equations. It involves performing operations on the rows of the coefficient matrix to achieve a row-echelon form, from which the solutions can be easily deduced.

**Analytic Geometry** likely introduces concepts related to the geometry of vectors and linear transformations.

**Vector Calculus** likely expands upon these concepts, applying calculus to vector fields and other vector-related functions.

**Dimensionality Reduction** might apply the concepts of basis and linear independence to reduce the number of variables in a dataset without losing much information, which is a key technique in fields like machine learning.

**Classification** could involve using vectors and matrices to classify data into different categories, a common task in machine learning algorithms.

### System of Linear Equations:
**For Example:**

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/0be1f33e-b86f-4898-bf26-e56b1724c15b)

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/8b8ac637-48ea-4212-b131-deebf95de272)

For a real-valued system of linear equations, we obtain either
no, exactly one, or infinitely many solutions. Linear equation solves a version of the above example when we cannot solve the system of linear equations.

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/7c6c0356-51ad-41d6-9a05-b9b333c9d944)

Remark (Geometric Interpretation of Systems of Linear Equations). In a
system of linear equations with two variables x1 , x2 , each linear equation
defines a line on the x1 x2 -plane. Since a solution to a system of linear
equations must satisfy all equations simultaneously, the solution set is the
intersection of these lines. This intersection set can be a line (if the linear
equations describe the same line), a point, or empty (when the lines are
parallel)


### Matrices

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/7c397823-9cd6-4b8a-ab2a-e100808f5041)

### Inverse Matrix

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/678d76a6-1567-46ed-abb9-f3e026e46972)

## Solving a system of Linear Equations 
### Elementary Transformations

- Key to solving a system of linear equations are elementary transformations
that keep the solution set the same, but that transform the equation system
into a simpler form.

**Remark**: Pivots and Staircase Structure
The leading coefficient of a row (the first nonzero number from the left) is called the pivot and is always strictly to the right of the pivot of the row above it. Therefore, any equation system in row-echelon form always has a “staircase” structure.

### Row-Echelon Form
A matrix is in row-echelon form if:

- All rows that contain only zeros are at the bottom of the matrix; correspondingly, all rows that contain at least one nonzero element are on top of rows that contain only zeros.
Looking at nonzero rows only, the first nonzero number from the left (also called the pivot or the leading coefficient) is always strictly to the right of the pivot of the row above it.
**Remark:** Basic and Free Variables
The variables corresponding to the pivots in the row-echelon form are called basic variables and the other variables are free variables.

### Reduced Row-Echelon Form
An equation system is in reduced row-echelon form (also: row-reduced echelon form or row canonical form) if

- It is in row-echelon form.
- Every pivot is 1.
- The pivot is the only nonzero entry in its column.

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/e9e1b5d3-03c3-4bad-9a9b-984b9663ea26)

## Minus-1 Trick: 
The "Minus-1 Trick"  is a practical method for determining the solution space of a homogeneous system of linear equations, $Ax = 0$. This can be particularly useful in machine learning, especially in areas involving optimization and handling underdetermined systems.

### Understanding the Minus-1 Trick
The idea is to manipulate a matrix in reduced row echelon form (RREF) to clearly identify a basis for the null space (kernel) of the matrix $A$. The "Minus-1 Trick" involves:

1. Starting with $A$ in RREF.
2. Augmenting $A$ by adding extra rows where each row has a "-1" in the positions corresponding to missing pivots (free variables) and "0" elsewhere.
3. The columns of the augmented matrix $\tilde{A}$ that contain "-1" as pivots essentially describe the vectors in the null space of $A$. These vectors form a basis for the solution space of the homogeneous equation $Ax = 0$.

## Application in Machine Learning
### 1. Feature Selection and Dimensionality Reduction:
In machine learning, especially in high-dimensional datasets, not all features (variables) might be relevant or independent. Finding a smaller subset of features that can explain the data (or target) can greatly enhance model performance and interpretability. Techniques such as Principal Component Analysis (PCA) or Linear Discriminant Analysis (LDA) often involve finding a basis for the null space of a matrix (e.g., covariance matrix in PCA). The "Minus-1 Trick" could be adapted to help identify linear dependencies and redundancies among features.

### 2. Regularization and Constraint Handling:
In scenarios involving regularization (like Ridge, Lasso), or where constraints are explicitly part of the model (as in constrained optimization), the ability to characterize and compute the null space becomes crucial. For example, in optimization problems where you want to minimize a loss function subject to certain linear constraints, understanding the null space of the constraint matrix can guide how perturbations affect the feasible solutions.

### 3. Solving Underdetermined Systems:
Often in machine learning, we deal with underdetermined systems where there are more features (variables) than observations. Such systems are typical in deep learning with a huge number of parameters. Techniques that involve understanding and exploiting the kernel of a matrix, like the "Minus-1 Trick", can help in finding solutions that not only fit the training data but also generalize well by encapsulating the essential structure of the data.

💡# IDEA | [Reference(Inverse matrices, column space and null space)](https://youtu.be/uQhTuRlWMxw?t=179)

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/5523f00a-7ae9-454d-8966-30f051e0bace)

💡 In linear regression, Ax=v, x is the weights that try to fall exactly on the b vector. A is the data that transforms the x to v. that x we obtain works as a model. 
💡 Inverse: Transforming the vector v by A_inverse to obtain x.

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/ebe6566a-172f-43e0-a8f2-56b444b41584)

# [RANK](https://youtu.be/uQhTuRlWMxw?t=488)

### Definition of Rank:
In the context of linear algebra, the rank of a matrix is defined as the maximum number of linearly independent column vectors in the matrix, which is the same as the maximum number of linearly independent row vectors in the matrix. Rank gives us a measure of the dimensionality of the vector space spanned by its rows or columns.

### Mathematical Expression:
For a matrix 
A, the rank can be determined through methods such as:

- Performing Gaussian elimination and counting the number of non-zero rows in the Row-Echelon Form (REF).
#### - Computing the number of non-zero singular values in its Singular Value Decomposition (SVD).

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/9daa1e38-25ae-4430-91a6-af2dd70dae0b)

# [Singular Value Decomposition](https://youtu.be/vSczTbgc8Rc)

📹 ## [Visualizing, identity matrix, scalar matrix, reflection matrix, diagonal matrix, zero matrix, shear matrix, orthogonal matrix, projection matrix, inverse of a matrix. ](https://youtu.be/7Gtxd-ew4lk)

# REMAIN
https://youtu.be/mhy-ZKSARxI

https://youtu.be/vSczTbgc8Rc



# [Determinant](https://youtu.be/Ip3X9LOh2dk): 

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/0d31405b-6558-4dd8-bf8b-704b2541838b)

![image](https://github.com/Chhabii/MathmaticsforML/assets/60286478/97f27bf3-8304-4501-a812-36d367d0a3ff)
