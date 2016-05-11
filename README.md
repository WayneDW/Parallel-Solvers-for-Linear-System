# Parallelism-in-Matrix-Computations

Given a large sparse linear system Ax = f, of order n = 100,000 that can be effectively reduced to a banded matrix using the reordering scheme "reverse Cuthill-McKee", use the method of Row Projection, accelerated via the Conjugate Algorithm to yield a parallel solver. Compare the robustness and parallel scalability/speed with preconditioned Krylov subspace methods preconditioned via approximate LU-factorization.
