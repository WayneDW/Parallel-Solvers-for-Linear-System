# Parallelism-in-Matrix-Computations

This repository contains 5 sections of codes to solve matrix computations of various types. The following codes (except for the 1st one) used the MPI/OpenMP programming paradigm, and were tested on four (or eight) nodes of the MC cluster.

### 1. Implement BBTS

Block banded triangular solver to solve Lx = b, where L is a banded lower triangular matrix with m subdiagonals.

### 2. Implement a SPIKE-like algorithm

Solve Lˆy = f, where Lˆ is a banded lower triangular matrix of order N = 32, 768 = 215, with t = 128 subdiagonals on a cluster of multicore nodes. 

The SPIKE-like algorithm should be implemented using MPI. In this algorithm, use BBTS (from Part (a)) to solve the system involvings Lˆi.


### 3. Block LU Factorization with boosting
Solve the dense linear system Ax = b using the approximate block LU Factorization algorithm with the procedure of ”diagonal boosting”

Let α be a multiple of the unit roundoff, e.g. 10−6, and aj be the jth column of the updated matrix A after step j − 1. In step j, if the diagonal pivot does not satisfy |pivot| > α||aj ||1, its value is ”boosted” as, pivot = pivot + β||aj ||1, if pivot > 0, pivot = pivot − β||aj ||1, if pivot < 0, where β is often taken as the square root of α.


### 4. The Spike algorithm:
Implement the Spike algorithm in Chapter 5.2.1 to solve Ax = b, where A is a banded matrix, on a cluster of multicore nodes. 

### 5. Sparse matrix solver:
Designed a parallel C++ solver with libraries MPI, MKL and LAPACK based on Kaczmarz scheme to solve a sparse linear system Ax = f of order 1 million

Implemented the RCK scheme to do reordering in sparse matrix and utilized Block Gauss-Seidel and permutations to transform the model into independent least square problems within a parallel Conjugate Gradient framework, resulting in a more robust and scalable method compared with GMRES using ILU preconditioner


Data Source: http://yifanhu.net/GALLERY/GRAPHS/search.html


Jul.3 2016
