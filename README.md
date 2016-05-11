# Parallelism-in-Matrix-Computations

Implement BBTS: Block banded triangular solver
to solve Lx = b, where L is a banded lower triangular
matrix with m subdiagonals.

Implement a SPIKE-like algorithm to solve Lˆy = f, where Lˆ is a banded lower triangular matrix of order N = 32, 768 = 215, with t = 128 subdiagonals on a cluster of multicore nodes. Your code should be implemented using the MPI/OpenMP programming paradigm, and be tested on four nodes of the MC cluster. The SPIKE-like algorithm should be implemented using MPI. In this algorithm, use BBTS (from Part (a)) to solve the system involvings Lˆi.


Block LU Factorization with boosting
Solve the dense linear system Ax = b using the approximate block LU Factorization algorithm with the procedure of ”diagonal boosting”1
. Your code should be implemented using the MPI/OpenMP programming paradigm, and tested on eight nodes of the MC cluster. Let α be a multiple of the unit roundoff, e.g. 10−6, and aj be the jth column of the updated matrix A after step j − 1. In step j, if the diagonal pivot does not satisfy |pivot| > α||aj ||1, its value is ”boosted” as,

pivot = pivot + β||aj ||1, if pivot > 0,
pivot = pivot − β||aj ||1, if pivot < 0,
where β is often taken as the square root of α.


The Spike algorithm:
Implement the Spike algorithm in Chapter 5.2.1 to solve Ax = b, where A is a banded matrix, on a cluster of multicore nodes. Your code should be implemented using the MPI/OpenMP programming paradigm, and be tested on eight nodes of the MC cluster.

Final:
Given a large sparse linear system Ax = f, of order n = 100,000 that can be effectively reduced to a banded matrix using the reordering scheme "reverse Cuthill-McKee", use the method of Row Projection, accelerated via the Conjugate Algorithm to yield a parallel solver. Compare the robustness and parallel scalability/speed with preconditioned Krylov subspace methods preconditioned via approximate LU-factorization.
