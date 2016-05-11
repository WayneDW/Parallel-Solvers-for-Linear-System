#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include "mpi.h"
#include "DenseMat.hpp"


#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

const int WID = 128;
const int cores = 4;

int main(int argc, char *argv[]) {
    int m, numtasks, taskid, mtype;
    int nthreads, chunk = 4;
    int local_col, dest;
    float **L, *x, *f;
    float **L_hat[4], **R_hat[4], *f_hat[4];
    float *gh[4], **M[4], **N[4], *g[4], *h[4];
    float **Ns, *hs, *u, *us[4], *yi[4];
    int i, j, k, col[1], num;
    float tmp, **Mat[2];
    FILE *file[3];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (taskid == MASTER) {
        readArg(argc, argv, file);
        printf("Reading vector f\n");
        f = readVector(file[1]);
        printf("Reading matrix L\n\n");
        L = readBandedMat(file[0], WID, col);
        data_seg(L, f, L_hat, R_hat, f_hat, col[0], cores, WID);
        double  tic = omp_get_wtime();
        for (i = 0; i < WID; i++)
            delete[] L[i];
        delete[] f;
        local_col = col[0] / cores;
        printf("Step 1: Sending L R f to node\n");
        mtype = FROM_MASTER;
        for (dest = 1; dest < cores; dest++) {
            MPI_Send(&local_col, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            for (i = 0; i < WID; i++)
                MPI_Send(&L_hat[dest][i][0], local_col, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
            for (i = 0; i < local_col; i++)
                MPI_Send(&R_hat[dest][i][0], WID, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&f_hat[dest][0], local_col, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
        }
        printf("Step 2: Solve g, h, M, N.\n");
        gh[taskid] = BBTS(L_hat[taskid], f_hat[taskid], local_col, WID, chunk, nthreads);
        
        printf("Step 3: Combine N and use cSweep to solve u.\n");
        mtype = FROM_WORKER;
        for (dest = 1; dest < cores; dest++) {
            N[dest] = matInit(WID, WID);
            h[dest] = new float[WID];
            for (i = 0; i < WID; i++)
                MPI_Recv(&N[dest][i][0], WID, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&h[dest][0], WID, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD, &status);
        }
        
        // Ns comes from N2/3/4
        Ns = matInit(WID * 4, WID * 4);
        for (i = 0; i < 4 * WID; i++) {
            for (j = 0; j < 4 * WID; j++) {
                if (i == j)
                    Ns[i][i] = 1.0;
                else if (i > j) {
                    num = i / WID;
                    if ((j >= num * WID - WID) && (j < num * WID))
                        Ns[i][j] = N[num][i - num * WID][j - num * WID + WID];
                } 
            }
        }

        // hs comes from h1/2/3/4
        hs = new float [WID * 4];
        h[0] = new float[WID];
        h[0] = vecSplit(gh[0], local_col - WID, WID);
        for (i = 0; i < 4 * WID; i++) {
            num = i / WID;
            hs[i] = h[num][i % WID];
        }

        // Step 3.5: cSweep to solve u
        u = cSweep(Ns, hs, 4 * WID, chunk, true);

        // Partition u to other nodes
        printf("Step 4: send u to other node.\n");
        mtype = FROM_MASTER;
        for (dest = 1; dest < cores; dest++) {
            us[dest - 1] = vecSplit(u, dest * WID - WID, dest * WID - 1);
            MPI_Send(&us[dest - 1][0], WID, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);   
        }
        printf("Step 5: gather yi and write to output.\n");
        mtype = FROM_WORKER;
        for (dest = 1; dest < cores; dest++) {
            yi[dest] = new float[local_col];
            MPI_Recv(&yi[dest][0], local_col, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD, &status);
        }

        x = new float[col[0]];
        for (i = 0; i < col[0]; i++) {
            if (i / local_col == 0)
                x[i] = gh[taskid][i];
            else
                x[i] = yi[i / local_col][i % local_col];
        }
        double  toc = omp_get_wtime();
        printf("Time = %.3f seconds\n", toc - tic);
        fileWrite(file[2], x, col[0]);
        
        
    }
    
    if (taskid > MASTER)
    {
        mtype = FROM_MASTER;
        MPI_Recv(&local_col, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        // Initialization
        L_hat[taskid] = matInit(WID, local_col);
        R_hat[taskid] = matInit(local_col, WID);
        f_hat[taskid] = new float[local_col]();
        N[taskid] = matInit(WID, WID);
        h[taskid] = new float[WID];
        us[taskid - 1] = new float[WID];
        yi[taskid] = new float[local_col];

        // Receive L R f from master node
        for (i = 0; i < WID; i++)
            MPI_Recv(&L_hat[taskid][i][0], local_col, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
        for (i = 0; i < local_col; i++)
            MPI_Recv(&R_hat[taskid][i][0], WID, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&f_hat[taskid][0], local_col, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
        

        // Step 2: Solve gi and hi from BBTS
        gh[taskid] = BBTS(L_hat[taskid], f_hat[taskid], local_col, WID, chunk, nthreads);

        // Step 2.1: Solve M, N from BBTS
        
        Mat[0] = matTrans(R_hat[taskid], local_col, WID);
        Mat[1] = BBTS_Mat(L_hat[taskid], Mat[0], local_col, WID, chunk, nthreads, taskid);
        Mat[0] = matTrans(Mat[1], WID, local_col);
        
        // Step 2.2: Get N
        for (i = 0; i <  WID; i++) {
            for (j = 0; j < WID; j++)
                N[taskid][i][j] = Mat[0][i + local_col - WID][j];
        }
        
        h[taskid] = vecSplit(gh[taskid], local_col - WID, WID);
        mtype = FROM_WORKER;
        for (i = 0; i <  WID; i++)
            MPI_Send(&N[taskid][i][0], WID, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&h[taskid][0], WID, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);
        // Step 4 Compute yi respectively.
        mtype = FROM_MASTER;
        MPI_Recv(&us[taskid - 1][0], WID, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);

        for (i = 0; i < local_col; i++) {
            yi[taskid][i] = gh[taskid][i];
            for (j = 0; j < WID; j++)
                yi[taskid][i] -= Mat[0][i][j] * us[taskid - 1][j];
        }
        // Step 5 final, send yi back to master
        mtype = FROM_WORKER;
        MPI_Send(&yi[taskid][0], local_col, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);

        for (i = 0; i <  WID; i++){
            delete[] L_hat[taskid][i];
            delete[] Mat[1][i];
            delete[] N[taskid][i];
        }
        
        for (i = 0; i <  local_col; i++)
            delete[] R_hat[taskid][i];
        delete[] h[taskid];
        delete[] N[taskid];
        delete[] R_hat[taskid];
        delete[] f_hat[taskid];
        delete[] L_hat[taskid];
        delete[] us[taskid - 1];
        delete[] yi[taskid];
        
    }
    
    MPI_Finalize();
    return 0;
}
