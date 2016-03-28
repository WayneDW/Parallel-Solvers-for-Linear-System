#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include "mpi.h"
#include "mkl.h"
#include "MatLib.hpp"


typedef double FLOAT_TYPE;


/* Parallelism in the Spike Algorithm to Solve A X = F 
 * where A is a banded nonsingular matrix of order N, e.g. 1 M
 * reference : Parallelism in Matrix Computations, Page 96 */

int SPIKE(FILE *file[3], char *argv[]) {
    int core, id, nthreads, chunk, loc;
    int local_col, dest, blockID, tmp;
    int dim, dim_s, len, supDiag, subDiag;
    int i, j, k, p, q, m, n, tag;
    long long *pivot[2];
    
    MPI_Status status;
    //MPI_Init (&argc,&argv);
    MPI_Init (NULL, NULL);
    MPI_Comm_rank (MPI_COMM_WORLD, &id);
    MPI_Comm_size (MPI_COMM_WORLD, &core);

    // Def Num : 0 -> Top, 1 -> Mid, 2 -> Bot
    FLOAT_TYPE *V[core][3], *W[core][3], *G[core][3];
    FLOAT_TYPE *A, *B, *C, *F, *S, *X, *X_i;
    FLOAT_TYPE *Solution, t1, t2, t3, t4, t5;
   
    // Get the Width of the Band Matrix
    if (id == core - 1) {
        printf("Nodes available: %d\n", core);
        printf("Number of off_Diagonals ...\n");
    }
    bandWidth(file[0], &dim, &len, &supDiag, &subDiag);
    if (id == core - 1)
        printf("SupDiag: %d\tSubDiag: %d\n\n", supDiag, subDiag);

    file[0] = fopen(argv[1], "r");
    // The order of the block diagonal matrix
    dim /= core;
    pivot[0] = new long long[dim];

    // For Band Storage, see reference below
    // https://software.intel.com/en-us/node/471382#BAND
    A = new FLOAT_TYPE[dim * (2 * supDiag + subDiag + 1)]();
    B = new FLOAT_TYPE[dim * supDiag]();
    C = new FLOAT_TYPE[dim * subDiag]();
    X_i = new FLOAT_TYPE[dim];
    F = new FLOAT_TYPE[dim];

    dim_s = (supDiag + subDiag) * (core - 1);
    if (id == core - 1) {
        S = new FLOAT_TYPE[dim_s * dim_s]();
        Solution = new FLOAT_TYPE[dim * core];
        pivot[1] = new long long[dim_s];
    }
    X = new FLOAT_TYPE[dim_s];

    // Take care this part;
    for (i = 0; i < 3; i++) {
        tmp = ((i == 1) ? MAX((dim - 2 * supDiag), 0) : supDiag);
        // Initialize X/V/W/G top/mid/bot for all nodes
        for (j = 0; j < core; j++) {
            G[j][i] = new FLOAT_TYPE[tmp];
            V[j][i] = new FLOAT_TYPE[tmp * supDiag];
            W[j][i] = new FLOAT_TYPE[tmp * supDiag];
        }
    }
    // Each node only reads its corresponding rows
    if (id == core - 1) 
        printf("Reading Vector ...\n");
    readVector(file[1], F, dim, id);
    if (id == core - 1)
        printf("Reading Band Matrix parallelly...\n\n");
    readBigBand(file[0], A, B, C, id, dim, len, supDiag, subDiag, core);
    // Make sure to start at the same time.
    //  *****************************  Step 1. The Spike Factorization Stage A = DS  **********************************
    if (id == core - 1) {
        printf("Step 1. Spike Factorization Stage, ");
        t1 = omp_get_wtime();
    }
    // LU factor Aj
    LAPACKE_dgbtrf (LAPACK_COL_MAJOR, dim, dim , subDiag, supDiag, A, supDiag + 2 * subDiag + 1, pivot[0]);
    // Solve Vj from Aj * Vj = [B'j], B'j = [0,...,Bj]T
    if (id != core - 1) {
        LAPACKE_dgbtrs (LAPACK_COL_MAJOR, 'N', dim, subDiag, supDiag, supDiag, A, supDiag + 2 * subDiag + 1, pivot[0], B, dim);
        chooseRows(B, dim, supDiag, supDiag, dim - supDiag, V[id][0], V[id][1], V[id][2], id);
    }

    // Solve Wj from Aj * Wj = [C'j], C'j = [Cj, 0,...]T
    if (id != 0) {
        LAPACKE_dgbtrs (LAPACK_COL_MAJOR, 'N', dim, subDiag, supDiag, subDiag, A, supDiag + 2 * subDiag + 1, pivot[0], C, dim);
        chooseRows(C, dim, subDiag, subDiag, dim - subDiag, W[id][0], W[id][1], W[id][2], id);
    }
    
    //  *****************************  Step 2. Postprocessing Stage   *************************************************
    //  *****************************  Step 2.1. Solve D G = F, F -> G  ***********************************************
    if (id == core - 1) {
        t2 = omp_get_wtime();
        printf("time used: %.3fs\nStep 2.1. Solve D G = F, ", t2 - t1);
    }
    LAPACKE_dgbtrs (LAPACK_COL_MAJOR, 'N', dim, subDiag, supDiag, 1, A, supDiag + 2 * subDiag + 1, pivot[0], F, dim);
    // Take care, if subDiag != supDiag, this part may make mistake
    chooseRows(F, dim, 1, subDiag, dim - subDiag, G[id][0], G[id][1], G[id][2], id);
    delete[] pivot[0];
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] F;
    // Share V, W, G top / bot for all nodes
    // Non-master nodes send G/V/W top/bot to MASTER
    if (id != core - 1) {
        for (j = 0; j < 3; j += 2) {
            MPI_Send(&G[id][j][0], supDiag, MPI_DOUBLE, core - 1, id, MPI_COMM_WORLD);
            MPI_Send(&V[id][j][0], supDiag * supDiag, MPI_DOUBLE, core - 1, id, MPI_COMM_WORLD);
            MPI_Send(&W[id][j][0], supDiag * supDiag, MPI_DOUBLE, core - 1, id, MPI_COMM_WORLD);
        }
    }
    
    // ******************************  Step 2.2. Solve S X = G in MASTER node *****************************************
    // ******************************      Get the reduced solution X_local   *****************************************
    
    if (id == core - 1) {
        t3 = omp_get_wtime();
        
        for (i = 0; i < core - 1; i++) {
            for (j = 0; j < 3; j += 2) {
                MPI_Recv(&G[i][j][0], supDiag, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
                MPI_Recv(&V[i][j][0], supDiag * supDiag, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
                MPI_Recv(&W[i][j][0], supDiag * supDiag, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
            }
        }
        
        printf("time used: %.3fs\nStep 2.2. Solve S X = G, ", t3 - t2);
        for (j = 0; j < dim_s; j++) {
            for (i = 0; i < dim_s; i++) {
                m = i / subDiag;
                n = j / subDiag;
                p = i % subDiag + subDiag * (j % subDiag);
                // We can speed up if we receive the data sequentially
                if (i == j)
                    S[j + i * dim_s] = 1;
                else if ((m % 2 == 0) && (n - m == 1))
                    S[i + j * dim_s] = V[m / 2][2][p];
                else if (((m + 1) % 2 == 0) && (n - m == 2))
                    S[i + j * dim_s] = V[(m + 1) / 2][0][p];
                else if ((n % 2 == 0) && (m - n == 1))
                    S[i + j * dim_s] = W[n / 2 + 1][0][p];
                else if ((n % 2 == 0) && (m - n == 2))
                    S[i + j * dim_s] = W[n / 2 + 1][2][p];
            }
        }
        // Factor S as L U
        LAPACKE_dgetrf (LAPACK_COL_MAJOR, dim_s, dim_s, S, dim_s, pivot[1]);
        // Solve S X = G, replace G with X
        for (i = 0; i < dim_s; i++) {
            blockID = (i / supDiag + 1) / 2;
            tmp = (((i / supDiag) % 2 == 1) ? 0 : 2);
            X[i] = G[blockID][tmp][i % supDiag];
        }

        LAPACKE_dgetrs (LAPACK_COL_MAJOR, 'N', dim_s, 1, S, dim_s, pivot[1], X, dim_s);
        delete[] pivot[1];
        delete[] S;
    }
    
    // Send X to all other nodes
    MPI_Bcast(X, dim_s, MPI_DOUBLE, core - 1, MPI_COMM_WORLD);
    //  *****************************  Step 3. Get the global solution iteratively  ***********************************
    if (id == core - 1) {
        t4 = omp_get_wtime();
        printf("time used: %.3fs\nStep 3. Get the global solution iteratively, ", t4 - t3);
    }
    
    if (id != core - 1) {
        vecSplit(X, (2 * id + 1) * supDiag, supDiag, G[id + 1][0]);
        cblas_dgemv (CblasColMajor, CblasNoTrans, supDiag, supDiag, -1, V[id][0], supDiag, G[id + 1][0], 1, 1, G[id][0], 1);
        cblas_dgemv (CblasColMajor, CblasNoTrans, supDiag, supDiag, -1, V[id][2], supDiag, G[id + 1][0], 1, 1, G[id][2], 1);
        if (dim - 2 * supDiag > 0)
            cblas_dgemv (CblasColMajor, CblasNoTrans, dim - 2 * supDiag, supDiag, -1, V[id][1], dim - 2 * supDiag, G[id + 1][0], 1, 1, G[id][1], 1);
        for (i = 0; i < core; i++)
            for (j = 0; j < 3; j++)
                delete[] V[i][j];
    }
    if (id != 0) {
        vecSplit(X, 2 * (id - 1) * supDiag, supDiag, G[id - 1][2]);
        cblas_dgemv (CblasColMajor, CblasNoTrans, supDiag, supDiag, -1, W[id][0], supDiag, G[id - 1][2], 1, 1, G[id][0], 1);
        cblas_dgemv (CblasColMajor, CblasNoTrans, supDiag, supDiag, -1, W[id][2], supDiag, G[id - 1][2], 1, 1, G[id][2], 1);
        if (dim - 2 * supDiag > 0)
            cblas_dgemv (CblasColMajor, CblasNoTrans, dim - 2 * supDiag, supDiag, -1, W[id][1], dim - 2 * supDiag, G[id - 1][2], 1, 1, G[id][1], 1);
        for (i = 0; i < core; i++)
            for (j = 0; j < 3; j++)
                delete[] W[i][j];
    }
    
    // Combine X_top/mid/bot into X_i
    for (i = 0; i < supDiag; i++)
        X_i[i] = G[id][0][i];
    for (i = supDiag; i < dim - supDiag; i++)
        X_i[i] = G[id][1][i - supDiag];
    for (i = dim - supDiag; i < dim; i++)
        X_i[i] = G[id][2][i - dim + supDiag];

    for (i = 0; i < core; i++)
        for (j = 0; j < 3; j++)
            delete[] G[i][j];
    
    // ******************************  Step 4. Gather all the local solutions *****************************************
    MPI_Gather(X_i, dim, MPI_DOUBLE, Solution, dim, MPI_DOUBLE, core - 1, MPI_COMM_WORLD);
    delete[] X_i;
    if (id == core - 1)  {
        t5 = omp_get_wtime();
        printf("time used: %.3f s\n\nTotal time used: %.2fs\n\n", t5 - t4, t5 - t1);
        fileWrite(file[2], Solution, dim * core);
        printf("The Last eight solutions are:\n");
        for (i = dim * core - 8; i < dim * core; i++)
            printf("X = %f\n", Solution[i]);
        delete[] Solution;
    }
    
    MPI_Finalize();
    return 1;
}

int main(int argc, char *argv[]) {
    FILE *file[3];
    readArg(argc, argv, file);
    SPIKE(file, argv);
}
