# robust0

Robust statistics in C++, for didactic purposes. The point here is very much to do things properly and efficiently, not much naivety, but we do use BLAS and LAPACK for simplicity, although custom implementations for specific needs are sometimes more useful. These programs run on the CPU, not GPU, but we implement these algorithms with the GPU in mind; if there is a good algorithm for the CPU that runs awfully on the GPU, we either use the GPU version or write a secondary unused GPU version.

## BLAS, LAPACK

The LAPACK C headers have this trailing `FORTRAN_STRLEN` that does not appear to be necessary in actual function calls. It is a mess, and it seems common practice to just not use headers and declare the functions manually for both LAPACK and BLAS.  
The naive types should be consistent between Fortran and C, in particular integers should be the same as defined by the OS, and practically 32 bit.

LAPACK has no determinant function for reasons. Determinants can be calculated via diagonalization procedures instead, as the product of the diagonal elements (product of eigenvalues). Practically from `DSYTRF` (factorization of symmetric matrices) we gain a block diagonal matrix, with the blocks at most 2x2. This is fine, slightly more generally the determiannt is still the product of the block diagonal determinants obviouisly. It's good here to use log scaling for stability.

## Statistics as linear algebra in BLAS/LAPACK

Data matrix $X_{d,n}$ for $d$-variate data of size $n$, so that in column-major order of BLAS/LAPACK, objects append in memory. This is the most efficient in realistic scenarios where e.g. data is measured over time.  
The sample mean is $m_d=X_{d,n}^\top1_n$, with $m_d$ and $1_n$ column vectors.  
The centered data is $X_{d,n}-m_d^\top1_n$.  
The unbiased sample covariance is $S_{d,d}=\frac1{n-1}X_{d,n}X_{d,n}^\top$.  
The unbiased sample covariance can nicely be calculated in BLAS/LAPACK, but the sample mean and centered data can't; they would require the construction of these awkward intermediary vector and matrix objects.  
It is also worthwhile to think about error propagation here, which is different for naive and divide-and-conquer algorithms. This is true for any kind of tensor contraction, whether it is a dot product, matrix multiplication, or sum of an array.

## RNG

Using 64-bit PCG XSL-RR, simple and high quality.  
On CPU, the fastest way to sample the standard normal distribution is via the Marsaglia polar method, but this is a Monte-Carlo method. On the GPU it makes more sense to use the Box-Muller transform, which is quite similar really.  
The Fisher-Yates algorithm is used to sample random permutations.

## Sampling multivariate Gaussian

Given a covariance matrix $\Sigma_{d,d}$, any kind of root of $\Sigma_{d,d}$ can be used to sample the distribution. E.g. via the Cholesky decomposition $\Sigma_{d,d}=L_{d,d}L^\top_{d,d}$, if $x_d\sim\mathcal N$ standard multivariate normal (each component standard normal), then $L_{d,d}x_d+\mu_d\sim\mathcal N(\mu_d,\Sigma_{d,d})$.

## Mahalanobis distance

Mahalanobis distance requires computing the inverse of the covariance matrix, which is also symmetric.

## Partial sort

For the fast MCD algorithm, we need to a partial sorting algorithm, finding the indices of the smallest few elements in a long array. A nice way to do this is via a max heap, with higher priority for the *largest* numbers. The trick is to only add elements to the queue if they are *small enough*, i.e. smaller than the *largest* element, the root of the priority queue. Other than this trick it's straight forward, just empty the queue, that gives the smallest numbers, the largest coming out first.

## MCD (minimum covariance determinant)

We use the usual fast MCD algorithm.

General review: Hubert, Debruyne, Rousseeuw, 'Minimum covariance determinant and extensions'  
Consistency factor calculations: Croux, Haesbroeck, 'Influence Function and Efficiency of the Minimum Covariance Determinant Scatter Matrix Estimator'  
Finite sample corrections for consistency factors: Pison, Aelst, Willems, 'Small sample corrections for LTS and MCD'

A robust estimator for the multivariate normal mean and covariance.
We use GSL for $\chi^2$ quantiles and CDF, they are required for cutoffs as well as consistency factors.

There are two consistency factors, $c_0$ for raw MCD and $c_1$ for reweighted MCD. These are quite tricky.  
When the number of samples is small, Monte Carlo simulations are generally used to infer the correct consistency factors by naively running the MCD algorithms (raw and reweighted) on known distributions. While not too difficult to implement in principle, we don't do this for now.  
For a large number of samples, $c_0$ and $c_1$ are both ultimately related to $\chi^2$ CDFs, or a different expectation of the $\chi^2$ distribution. There are analytical descriptions, and Monte-Carlo becomes prohibitively expensive, so it is sensible to use these analytical descriptions.
$c_0$ is quite straight forward, $c_1$ less so, in part just because it is not discussed in the review article, which is simpler than (Croux, Haesbroeck). So for now we just set $c_1=1$.
$c_1$ does not appear to matter too much, setting $c_1=1$ gives correct results, at least for the example considered here of polluted normal data. This sounds positive but it's actually a problem, it means it is difficult to test an implementation of the 'correct' consistency factor. For now we just set $c_1=1$, maybe later we come back to this having found a better test or having asked an expert.

As a test, we generate 10000 trivariate normal samples of two distributions with vastly different means and combine them, 99% of the one, 1% of the other. The usual unbiased sample covariance gives garbage estimates of the original distribution of course, because it assumes those 1% outliers are real data. The MCD algorithm with $\alpha=75%$ gives good results however, reproducing the distribution of the 99% (with of course some statistical error).
The test outputs the following data, displaying what we just described. Here symmetric matrices are printed. (These are LAPACK dense symmetric matrices, the zeros are not actual zero matrix elements, they are just unused data; the actual covariance matrix can be inferred from symmetry.)  
MCD analysis is not inexpensive, this analysis takes on the order of 1s for the 10000 samples on 1 CPU thread. We compile statically to avoid some of the overhead for operations on small matrix sizes in LAPACK, but it does not make a substantial difference. And obviously a serious implementation would do all this on the GPU. But the implementation is proper, 1s is not slow.
```
mean
actual   [+1.00e+00, +2.00e+00, +3.00e+00]
estimate [+1.05e+00, +2.02e+00, +2.94e+00] (non-robust)
estimate [+9.93e-01, +2.00e+00, +3.00e+00] (robust)
cov
actual   [+3.00e-01, +0.00e+00, +1.00e-01, +0.00e+00, +2.00e-01, +0.00e+00, +0.00e+00, +0.00e+00, +1.00e-01]
estimate [+6.54e-01, +1.24e-01, -2.54e-01, +0.00e+00, +2.42e-01, -1.20e-01, +0.00e+00, +0.00e+00, +4.55e-01] (non-robust)
estimate [+2.99e-01, +3.19e-03, +1.01e-01, +0.00e+00, +2.00e-01, +8.11e-04, +0.00e+00, +0.00e+00, +1.01e-01] (robust)
```

## LTS (least trimmed squares)

We will use the usual fast LTS algorithm, this has not been implemented yet, but is based on the MCD via the fast MCD algorithm.
