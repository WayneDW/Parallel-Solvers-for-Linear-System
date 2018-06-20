#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include "mpi.h"
#include "mkl.h"
#include "DenseMat.hpp"

float *readMat(FILE *file, int *row, int *col) {
    int i, m, n, len;
    float value, *L;
    for (i = 0; i < 2; i++)
        fscanf(file, "%*[^\n]%*c");
    fscanf(file, "%d %d", row, col);
    L = new float[*row * *col]();
    for (i = 0; i < *row * *col; i++) {
        fscanf(file, "%f\n", &L[i]);
    }
    return L;
}

// type: true for vec, false for square matrix split
float *Seg(float *vec, int dim, int n, int i0, char type) {
    int i, j, m, tmp;
    float *val;
    if (type == true) {
        val = new float[n];
        for (i = 0; i < n; i++)
            val[i] = vec[i0 + i];
    }
    else {
        val = new float[n * n];
        for (i = 0; i < n; i++) {
            tmp = i0 + i * dim;
            for (j = 0; j < n; j++)
                val[j + n * i] = vec[tmp + j];
        }
    }
    return val;
}

void getLU(float *vec, int dim, float *L, float *U) {
    int i, j;
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            if (i == j) {
                L[j * dim + j] = 1;
                U[i * dim + j] = vec[j * dim + j];
            }
            else if (i > j)
                L[j + i * dim] = vec[j + i * dim];
            else
                U[j + i * dim] = vec[j + i * dim];
        }
    }
}

float *Trans(float *A, int m, int n) {
    int i, j, k;
    float *val;
    val = new float [n * m];
    for (i = 0; i < m * n; i++) {
        j = i / m;
        k = i % m;
        val[k * m + j] = A[i];
    }
    return val;
}

void matLU(float *A, int m, int tag, int chunk) {
    int i, j;
    float beta, norm, tmp;
    norm = 0;
    beta = 0.001;
    #pragma omp parallel for reduction(+:norm)
    for (i = 0; i < m; i++)
        norm += abs(A[tag + i * m]);
    #pragma omp parallel shared(A, m, tag, norm, tmp) private(i, j) 
    {
        #pragma omp for schedule (static, chunk)
        for (i = tag + 1; i < m; i++) {
            for (j = tag; j < m; j++) {
                if (j == tag) {
                    if (abs(A[tag + tag * m]) <= beta * beta * norm)
                        if (A[tag + tag * m] > 0)
                            A[tag + tag * m] += beta * norm;
                        else
                            A[tag + tag * m] -= beta * norm;
                    A[j + i * m] /= A[tag + tag * m];
                    
                }
                else
                    A[j + i * m] -= A[tag + i * m] * A[j + tag * m];
            }
        }
    }
    if (tag < m - 2)
        matLU(A, m, tag + 1, chunk);
}

int main(int argc, char *argv[]) {
    int core, id, nthreads, chunk;
    int local_col, dest, last, ori, tag, tmp;
    int i, j, k, m, n, row, col, dim, round;
    long long *pivot;
    FILE *file[3];

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &core);

    float *AA, *A, *b, *BlockA[core][core], *B[core], *x;
    float *G[core][core], *Bak[core][core], *L[core][core], *U[core][core];

    tag = 0;
    chunk = 4;
    readArg(argc, argv, file);
    if (id == 0)    printf("Reading vector b\n");
    b = readVector(file[1], &n);
    if (id == 0)    printf("Reading matrix A\n");
    A = readMat(file[0], &row, &col);
    // didn't notice the matrix is saved by col, trans instead
    //A = Trans(AA, row, col);

    dim = row / core;
    pivot = new long long[dim];
    double  tic = omp_get_wtime();

    if (id == 0) {
        printf("Block LU factorization start... \n");
        BlockA[id][id] = Seg(A, row, dim, id * dim * row + id * dim, false);
    }
    else {
        // Receive info from node ori, core id receive id times
        for (round = 0; round < id; round++) {
            G[tag][id] = new float[dim * dim];
            L[id][tag] = new float[dim * dim];
            MPI_Recv(&G[tag][id][0], dim * dim, MPI_FLOAT, tag, tag * dim + id, MPI_COMM_WORLD, &status);
            MPI_Recv(&L[id][tag][0], dim * dim, MPI_FLOAT, tag, 9 * core * core + id * dim + tag, MPI_COMM_WORLD, &status);
            //for (dest = tag + 1; dest < core; dest++) {
            for (dest = tag + 1; dest < core; dest++) {
                if (round == 0) {
                       BlockA[tag][dest] = Seg(A, row, dim, tag * dim * row + dest * dim, false);
                    BlockA[id][dest] = Seg(A, row, dim, id * dim * row + dest * dim, false);
                }
                // Update A[id][id + 1], A[id][id + 2], ...
                cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, -1,
                G[tag][id], dim, BlockA[tag][dest], dim, 1, BlockA[id][dest], dim);
                // Update A[tag + 1 : core][tag + 1 : core]
                for (k = tag + 1; k < core; k++) {
                    if (id != k)
                        BlockA[k][dest] = new float[dim * dim];
                    MPI_Bcast(BlockA[k][dest], dim * dim, MPI_FLOAT, k, MPI_COMM_WORLD);
                }
            }
            tag++;
        }
    }
    // Copy BlockA to Bak
    Bak[id][id] = new float[dim * dim];
    cblas_scopy (dim * dim, BlockA[id][id], 1, Bak[id][id], 1);

    //LAPACKE_mkl_sgetrfnpi(CblasRowMajor, dim, dim, dim, Bak[id][id], dim);
    matLU(Bak[id][id], dim, 0, chunk);
    // Get L, U matrix
    L[id][id] = new float[dim * dim]();
    U[id][id] = new float[dim * dim]();
    getLU(Bak[id][id], dim, L[id][id], U[id][id]);

    // LU factorization for A[id][id]
    LAPACKE_sgetrf (CblasRowMajor, dim, dim, BlockA[id][id], dim, pivot);
    //Get inv(A[id][id])
    LAPACKE_sgetri(CblasRowMajor, dim, BlockA[id][id], dim, pivot);

    for (dest = id + 1; dest < core; dest++) {
        if (id == 0) {
            BlockA[dest][id] = Seg(A, row, dim, dest * dim * row + id * dim, false);
            BlockA[id][dest] = Seg(A, row, dim, id * dim * row + dest * dim, false);
        }
        G[id][dest] = new float[dim * dim];
        // Solve G[id][j] = A[j][id] * inv(A[id][id])
        cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1,
        BlockA[dest][id], dim, BlockA[id][id], dim, 0, G[id][dest], dim);

        MPI_Send(&G[id][dest][0], dim * dim, MPI_FLOAT, dest, tag * dim + dest, MPI_COMM_WORLD);
        // Change G[][] to element of Lower triangular matrix
        cblas_strmm (CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, dim, dim, 1, L[id][id], dim, G[id][dest], dim);

        L[dest][id] = new float[dim * dim];
        cblas_scopy (dim * dim, G[id][dest], 1, L[dest][id], 1);
        // Send L[dest][id] from Bcast
        MPI_Send(&L[dest][id][0], dim * dim, MPI_FLOAT, dest, 9 * core * core + dest * dim + id, MPI_COMM_WORLD);

        // Update A[1,2], A[1,3] inv(L) * A[1][2] by L x = A[1][2]
        LAPACKE_strtrs (CblasRowMajor, 'L', 'N', 'U', dim, dim, L[id][id], dim, BlockA[id][dest], dim);
    }
    
    for (tag = id + 1; tag < core; tag++) {
        for (dest = tag; dest < core; dest++){
            for (k = tag; k < core; k++) {
                if (id != k)
                    BlockA[k][dest] = new float[dim * dim];
                MPI_Bcast(BlockA[k][dest], dim * dim, MPI_FLOAT, k, MPI_COMM_WORLD);
            }
        }
    }
    
    double  toc1 = omp_get_wtime();
    // To make sure all nodes have completed
    MPI_Bcast(&tag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (id == 0) printf("Time of LU factorization = %.3f seconds\n", toc1 - tic);
    
    // *************************** PART 2 ************************************
    // Step 2: Backward sweep using pipeline, Solve y for L[i,i] * y[i] = B[i]
    B[id] = Seg(b, dim * dim, dim, id * dim, true);
    for (i = 0; i < id; i++) {
        B[i] = new float[dim * dim];
        MPI_Recv(&B[i][0], dim, MPI_FLOAT, i, 7 * core * core + i, MPI_COMM_WORLD, &status);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, -1, L[id][i], dim, B[i], 1, 1, B[id], 1);
    }
    // Solve L[i][i] * y[i] = b[i] and Bcast to other core
    cblas_strsv (CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit, dim, L[id][id], dim, B[id], 1);
    for (i = id + 1; i < core; i++) {
        MPI_Send(&B[id][0], dim, MPI_FLOAT, i, 7 * core * core + id, MPI_COMM_WORLD);
    }
    
    double  toc2 = omp_get_wtime();
    MPI_Bcast(&tag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (id == 0) printf("Time of block backward sweep = %.3f seconds\n",  toc2 - toc1);
    
    // *************************** PART 3 *********************
    // Step 3: Forward sweep using pipeline, Solve x for Ux = y
    for (i = core - 1; i > id; i--) {
        B[i] = new float[dim * dim];
        MPI_Recv(&B[i][0], dim, MPI_FLOAT, i, 9 * core * core + i, MPI_COMM_WORLD, &status);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, -1, BlockA[id][i], dim, B[i], 1, 1, B[id], 1);
    }
    // Solve U[i][i] * x[i] = y[i] and calculate the norm
    cblas_strsv (CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, dim, U[id][id], dim, B[id], 1);
    for (i = 0; i < id; i++)
        MPI_Send(&B[id][0], dim, MPI_FLOAT, i, 9 * core * core + id, MPI_COMM_WORLD);
    
    // Final summary
    double  toc3 = omp_get_wtime();
    if (id == 0) {
        x = new float [col];
        for (i = 0; i < core; i++)
            for (j = 0; j < dim; j++)
                x[j + i * dim] = B[i][j];
        float norm = cblas_snrm2 (col, b, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, col, col, -1, A, col, x, 1, 1, b, 1);
        norm = cblas_snrm2 (row, b, 1) / norm;
        printf("Time of block forward sweep = %.3f seconds\n", toc3 - toc2);
        printf("Relative residual is %e\n", norm);
        fileWrite(file[2], x, col);
    }

    MPI_Finalize();
    return 0;
}
