#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include "mpi.h"
#include "mkl.h"
#include "DenseMat.hpp"

int main(int argc, char *argv[]) {
    int core, id, nthreads, chunk, steps;
    int local_col, dest, last, ori, tag, tmp;
    int i, j, k, m, n, row, dim, round;
    long long *pivot;
    FILE *file[3];

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &core);

    double *AA, *A, *b, *BlockA[core][core], *B[core], *x;
    double *G[core][core], *L[core][core], *U[core][core];

    tag = 0;
    chunk = 4;
    readArg(argc, argv, file);
    if (id == 0)    printf("Reading vector b\n");
    b = readVector(file[1], &n);
    if (id == 0)    printf("Reading matrix A\n");
    AA = readMat(file[0], &row);
    // didn't notice the matrix is saved by col, trans instead
    A = Trans(AA, row, row);

    dim = row / core;
    pivot = new long long[dim];
    double  tic = omp_get_wtime();
    for (i = 0; i < dim; i++)
        pivot[i] = i + 1;

    if (id == 0) {
        printf("Block LU factorization start... \n");
        BlockA[id][id] = Seg(A, row, dim, id * dim * row + id * dim, false);
        steps = 1;
    }
    else {
        // Receive info from node ori, core id receive id times
        for (round = 0; round < id; round++) {
            G[tag][id] = new double[dim * dim];
            L[id][tag] = new double[dim * dim];
            MPI_Recv(&G[tag][id][0], dim * dim, MPI_DOUBLE, tag, tag * dim + id, MPI_COMM_WORLD, &status);
            MPI_Recv(&L[id][tag][0], dim * dim, MPI_DOUBLE, tag, 9 * core * core + id * dim + tag, MPI_COMM_WORLD, &status);
            //for (dest = tag + 1; dest < core; dest++) {
            for (dest = tag + 1; dest < core; dest++) {
                if (round == 0) {
                       BlockA[tag][dest] = Seg(A, row, dim, tag * dim * row + dest * dim, false);
                    BlockA[id][dest] = Seg(A, row, dim, id * dim * row + dest * dim, false);
                }
                // Update A[id][id + 1], A[id][id + 2], ...
                cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, -1,
                G[tag][id], dim, BlockA[tag][dest], dim, 1, BlockA[id][dest], dim);
                // Update A[tag + 1 : core][tag + 1 : core]
                for (k = tag + 1; k < core; k++) {
                    if (id != k)
                        BlockA[k][dest] = new double[dim * dim];
                    MPI_Bcast(BlockA[k][dest], dim * dim, MPI_DOUBLE, k, MPI_COMM_WORLD);
                }
            }
            tag++;
        }
    }

    // Show the iteration number
    if (id != 0)
        MPI_Recv(&steps, 1, MPI_INT, id - 1, id - 1, MPI_COMM_WORLD, &status);

    // LU factorization for A[id][id]
    matLU(BlockA[id][id], dim, 0, chunk, id, dim, &steps);

    if (id != core - 1) 
        MPI_Send(&steps, 1, MPI_INT, id + 1, id, MPI_COMM_WORLD);
        
    
    // Get L, U matrix
    L[id][id] = new double[dim * dim]();
    U[id][id] = new double[dim * dim]();
    getLU(BlockA[id][id], dim, L[id][id], U[id][id]);

    //Get inv(A[id][id])
    LAPACKE_dgetri(CblasRowMajor, dim, BlockA[id][id], dim, pivot);

    for (dest = id + 1; dest < core; dest++) {
        if (id == 0) {
            BlockA[dest][id] = Seg(A, row, dim, dest * dim * row + id * dim, false);
            BlockA[id][dest] = Seg(A, row, dim, id * dim * row + dest * dim, false);
        }
        G[id][dest] = new double[dim * dim];
        // Solve G[id][j] = A[j][id] * inv(A[id][id])
        cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1,
        BlockA[dest][id], dim, BlockA[id][id], dim, 0, G[id][dest], dim);

        MPI_Send(&G[id][dest][0], dim * dim, MPI_DOUBLE, dest, tag * dim + dest, MPI_COMM_WORLD);
        // Change G[][] to element of Lower triangular matrix
        cblas_dtrmm (CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, dim, dim, 1, L[id][id], dim, G[id][dest], dim);

        L[dest][id] = new double[dim * dim];
        cblas_dcopy (dim * dim, G[id][dest], 1, L[dest][id], 1);
        // Send L[dest][id] from Bcast
        MPI_Send(&L[dest][id][0], dim * dim, MPI_DOUBLE, dest, 9 * core * core + dest * dim + id, MPI_COMM_WORLD);

        // Update A[1,2], A[1,3] inv(L) * A[1][2] by L x = A[1][2]
        LAPACKE_dtrtrs (CblasRowMajor, 'L', 'N', 'U', dim, dim, L[id][id], dim, BlockA[id][dest], dim);
    }
    
    for (tag = id + 1; tag < core; tag++) {
        for (dest = tag; dest < core; dest++){
            for (k = tag; k < core; k++) {
                if (id != k)
                    BlockA[k][dest] = new double[dim * dim];
                MPI_Bcast(BlockA[k][dest], dim * dim, MPI_DOUBLE, k, MPI_COMM_WORLD);
            }
        }
    }
    
    double  toc1 = omp_get_wtime();
    // To make sure all nodes have completed
    MPI_Bcast(&tag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (id == 0) printf("Time of LU factorization = %.3f seconds\n", toc1 - tic);
    
    // *************************** PART 2 ************************************
    // Step 2: Backward sweep using pipeline, Solve y for L[i,i] * y[i] = B[i]
    B[id] = Seg(b, row, dim, id * dim, true);
    for (i = 0; i < id; i++) {
        B[i] = new double[dim];
        MPI_Recv(&B[i][0], dim, MPI_DOUBLE, i, 7 * core * core + i, MPI_COMM_WORLD, &status);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, -1, L[id][i], dim, B[i], 1, 1, B[id], 1);
    }
    // Solve L[i][i] * y[i] = b[i] and Bcast to other core
    cblas_dtrsv (CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit, dim, L[id][id], dim, B[id], 1);
    for (i = id + 1; i < core; i++) {
        MPI_Send(&B[id][0], dim, MPI_DOUBLE, i, 7 * core * core + id, MPI_COMM_WORLD);
    }
    
    double  toc2 = omp_get_wtime();
    MPI_Bcast(&tag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (id == 0) printf("Time of block backward sweep = %.3f seconds\n",  toc2 - toc1);
    
    // *************************** PART 3 *********************
    // Step 3: Forward sweep using pipeline, Solve x for Ux = y
    for (i = core - 1; i > id; i--) {
        B[i] = new double[dim];
        MPI_Recv(&B[i][0], dim, MPI_DOUBLE, i, 9 * core * core + i, MPI_COMM_WORLD, &status);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, -1, BlockA[id][i], dim, B[i], 1, 1, B[id], 1);
    }
    // Solve U[i][i] * x[i] = y[i] and calculate the norm
    cblas_dtrsv (CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, dim, U[id][id], dim, B[id], 1);
    for (i = 0; i < id; i++)
        MPI_Send(&B[id][0], dim, MPI_DOUBLE, i, 9 * core * core + id, MPI_COMM_WORLD);
    
    // Final summary
    double  toc3 = omp_get_wtime();
    if (id == 0) {
        x = new double [row];
        for (i = 0; i < core; i++)
            for (j = 0; j < dim; j++)
                x[j + i * dim] = B[i][j];
        double norm = cblas_dnrm2 (row, b, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, row, row, -1, A, row, x, 1, 1, b, 1);
        norm = cblas_dnrm2 (row, b, 1) / norm;
        printf("Time of block forward sweep = %.3f seconds\n", toc3 - toc2);
        printf("Relative residual is %e\n", norm);
        fileWrite(file[2], x, row);
    }

    MPI_Finalize();
    return 0;
}
