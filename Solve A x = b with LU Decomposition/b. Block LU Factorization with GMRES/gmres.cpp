/* Example to show how to use Intel's FGMRES with preconditioner to solve the linear system Ax=b in MPI.
* Based on Intel's example: solverc/source/fgmres_full_funct_c.c
* For CS51501 HW3 Part b
*
* Please read Intel Reference Manual, Chapter 6 Sparse Solve Routine, FGMRES Interface Description for the detail information.
*/
#include <stdio.h>
#include "mkl.h"
#include "mpi.h"
#include <omp.h>
#include "DenseMat.hpp"

#define MASTER 0               // taskid of first task 
#define RESTART 500
#define TOL 1e-6
#define MAXIT 10000

void mpi_dgemv(const MKL_INT m, const MKL_INT local_m, const double *A, const double *u, double *v, double *local_u, double *local_v, int taskid, MPI_Comm comm);
void mpi_preconditioner_solver(const MKL_INT m, const MKL_INT local_m, double *local_M, const double *u, double *v, double * local_u, int taskid, MPI_Comm comm);


int main(int argc, char *argv[])
{

	int	taskid;              // a task identifier 
	int	nthreads, numtasks;              // number of tasks in partition
	MPI_Comm comm;
	int m; // size of the matrix
	int	local_m, temp;                  // rows of matrix A sent to each worker 
	double *Ap, *b, *A;
	double *temp_1, *temp_2;
	double *local_A, *local_v, *local_u;
	double *local_M; // M is the preconditioner in this example, which is the diagonal element of A;
	int i, j, k;
    FILE *file[3];

    MPI_Status status;   
	MPI_Init(&argc, &argv);
	comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &taskid);
	MPI_Comm_size(comm, &numtasks);
    // initilization: A and b
    readArg(argc, argv, file);
    if (taskid == 0)    printf("Reading vector b\n");
    b = readVector(file[1], &m);
    if (taskid == 0)    printf("Reading matrix A\n");
    Ap = readMat(file[0], &m);
    /* end modification 1*/

	local_m = m / numtasks;
	local_A = (double *)malloc(sizeof(double)*(local_m*m));
	local_u = (double *)malloc(sizeof(double)*(local_m));
	local_v = (double *)malloc(sizeof(double)*m);
	//	partition A and send A_i to local_A on  node i
	MPI_Scatter(Ap, local_m*m, MPI_DOUBLE, local_A, local_m*m, MPI_DOUBLE, MASTER, comm);

	/* start modification 2: generate preconditioner M
	* In this example, TA choose the diagonal elements of A as the preconditioner.*/
    
    // Ap is saved by col, A is saved by row
	A = (double *)malloc(sizeof(double)*m * m);
    A = Trans(Ap, m, m);
    free(Ap);
	
	int chunk, steps, id, core, row;
	int local_col, dest, last, ori, tag;
	int n, dim, round;
    long long *pivot;
    
    core = numtasks;
    id = taskid;
    row = m;

    double *BlockA[core][core], *B[core], *x;
    double *G[core][core], *L[core][core], *U[core][core];
    tag = 0;
    chunk = 4;

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
            MPI_Recv(&G[tag][id][0], dim * dim, MPI_DOUBLE, tag, tag * dim + id, comm, &status);
            MPI_Recv(&L[id][tag][0], dim * dim, MPI_DOUBLE, tag, 9 * core * core + id * dim + tag, comm, &status);
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
        			MPI_Bcast(BlockA[k][dest], dim * dim, MPI_DOUBLE, k, comm);
                }
            }
            tag++;
		}
    }
    
    // Show the iteration number
    if (id != 0)
        MPI_Recv(&steps, 1, MPI_INT, id - 1, id - 1, comm, &status);
    // LU factorization for A[id][id]
    matLU(BlockA[id][id], dim, 0, chunk, id, dim, &steps);
    if (id != core - 1) 
        MPI_Send(&steps, 1, MPI_INT, id + 1, id, comm);
        
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

	    MPI_Send(&G[id][dest][0], dim * dim, MPI_DOUBLE, dest, tag * dim + dest, comm);
        // Change G[][] to element of Lower triangular matrix
        cblas_dtrmm (CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, dim, dim, 1, L[id][id], dim, G[id][dest], dim);

        L[dest][id] = new double[dim * dim];
        cblas_dcopy (dim * dim, G[id][dest], 1, L[dest][id], 1);
        // Send L[dest][id] from Bcast
        MPI_Send(&L[dest][id][0], dim * dim, MPI_DOUBLE, dest, 9 * core * core + dest * dim + id, comm);

        // Update A[1,2], A[1,3] inv(L) * A[1][2] by L x = A[1][2]
        LAPACKE_dtrtrs (CblasRowMajor, 'L', 'N', 'U', dim, dim, L[id][id], dim, BlockA[id][dest], dim);
    }
    
    for (tag = id + 1; tag < core; tag++) {
        for (dest = tag; dest < core; dest++){
            for (k = tag; k < core; k++) {
                if (id != k)
                    BlockA[k][dest] = new double[dim * dim];
                MPI_Bcast(BlockA[k][dest], dim * dim, MPI_DOUBLE, k, comm);
            }
        }
    }
    
    double  toc1 = omp_get_wtime();
    // To make sure all nodes have completed
    MPI_Bcast(&tag, 1, MPI_INT, 0, comm);
    if (id == 0)    printf("Time of LU factorization = %.3f seconds\n", toc1 - tic);
    
	/* end  modification 2*/

	/*---------------------------------------------------------------------------
	* GMRES: Allocate storage for the ?par parameters and the solution vectors
	*---------------------------------------------------------------------------*/
	MKL_INT RCI_request;
	int RCI_flag;
	double dvar;
	int flag = 0;

	MKL_INT ipar[128]; //specifies the integer set of data for the RCI FGMRES computations
	double dpar[128]; // specifies the double precision set of data
	double *tmp; //used to supply the double precision temporary space for theRCI FGMRES computations, specifically:
	double *computed_solution;
	double *residual;
	double *f;
	MKL_INT itercount, ierr = 0;;
	MKL_INT  ivar;
	double b_2norm;
	char cvar = 'N';
	MKL_INT incx = 1;
	if (taskid == MASTER) {
		ipar[14] = RESTART; // restart iteration number
		int n_tmp = (2 * ipar[14] + 1) * m + ipar[14] * (ipar[14] + 9) / 2 + 1;
		tmp = (double *)malloc(sizeof(double)*n_tmp);
		computed_solution = (double *)malloc(sizeof(double)*m);
		residual = (double *)malloc(sizeof(double)*m);
		f = (double *)malloc(sizeof(double)*m);

		ivar = m;
		/*---------------------------------------------------------------------------
		* Initialize the initial guess
		*---------------------------------------------------------------------------*/
		for (i = 0; i < m; i++)
		{
			computed_solution[i] = 0.5;
		}

		b_2norm = cblas_dnrm2(ivar, b, incx);
		//	printf("b_2norm=%f\n",b_2norm);
		/*---------------------------------------------------------------------------
		* Initialize the solver
		*---------------------------------------------------------------------------*/
		dfgmres_init(&ivar, computed_solution, b, &RCI_request, ipar, dpar, tmp);
		RCI_flag = RCI_request;
	}
	MPI_Bcast(&RCI_flag, 1, MPI_INT, MASTER, comm);
	if (RCI_flag != 0) {
		goto FAILED;
    }

	if (taskid == MASTER) {
		/*---------------------------------------------------------------------------
		* GMRES: Set the desired parameters:
		*---------------------------------------------------------------------------*/
		ipar[14] = RESTART; // restart iteration number
		ipar[7] = 1; //do the stopping test
		ipar[10] = 1; // use preconditioner
		dpar[0] = TOL;
		/*---------------------------------------------------------------------------
		* Check the correctness and consistency of the newly set parameters
		*---------------------------------------------------------------------------*/
		dfgmres_check(&ivar, computed_solution, b, &RCI_request, ipar, dpar,
			tmp);
		RCI_flag = RCI_request;
	}

	MPI_Bcast(&RCI_flag, 1, MPI_INT, MASTER, comm);
	if (RCI_flag != 0) {
		goto FAILED;
    }

	/*---------------------------------------------------------------------------
	* Compute the solution by RCI (P)FGMRES solver with preconditioning
	* Reverse Communication starts here
	*---------------------------------------------------------------------------*/
ONE:
	if (taskid == MASTER) {
		dfgmres(&ivar, computed_solution, b, &RCI_request, ipar, dpar, tmp);
		RCI_flag = RCI_request;
	}
	MPI_Bcast(&RCI_flag, 1, MPI_INT, MASTER, comm); // send RCI_request from node MASTER to all other nodes.

	/*---------------------------------------------------------------------------
     * * If RCI_request=0, then the solution was found with the required precision
     * *---------------------------------------------------------------------------*/
	if (RCI_flag == 0)
		goto COMPLETE;
	/*---------------------------------------------------------------------------
	* If RCI_request=1, then compute the vector A*tmp[ipar[21]-1]
	* and put the result in vector tmp[ipar[22]-1]
	*---------------------------------------------------------------------------
	* NOTE that ipar[21] and ipar[22] contain FORTRAN style addresses,
	* therefore, in C code it is required to subtract 1 from them to get C style
	* addresses
	*---------------------------------------------------------------------------*/
	if (RCI_flag == 1)
	{
		if (taskid == MASTER) {
			temp_1 = &tmp[ipar[21] - 1];
			temp_2 = &tmp[ipar[22] - 1];
		}
		mpi_dgemv(m, local_m, local_A, temp_1, temp_2, local_u, local_v, taskid, comm);

		goto ONE;
	}
	/*---------------------------------------------------------------------------
	* If RCI_request=2, then do the user-defined stopping test
	* The residual stopping test for the computed solution is performed here
	*---------------------------------------------------------------------------
	*/
	if (RCI_flag == 2)
	{
		/* Request to the dfgmres_get routine to put the solution into b[N] via ipar[12]
		--------------------------------------------------------------------------------
		WARNING: beware that the call to dfgmres_get routine with ipar[12]=0 at this
		stage may destroy the convergence of the FGMRES method, therefore, only
		advanced users should exploit this option with care */
		if (taskid == MASTER) {
			ipar[12] = 1;
			/* Get the current FGMRES solution in the vector f */
			dfgmres_get(&ivar, computed_solution, f, &RCI_request, ipar, dpar, tmp, &itercount);
			temp_1 = f;
			temp_2 = residual;
		}
		/* Compute the current true residual via mpi mat_vec multiplication */
		mpi_dgemv(m, local_m, local_A, temp_1, temp_2, local_u, local_v, taskid, comm);

		if (taskid == MASTER) {
			dvar = -1.0E0;
			cblas_daxpy(ivar, dvar, b, incx, residual, incx);
			dvar = cblas_dnrm2(ivar, residual, incx);
			printf("iteration %d, relative residual:%e\n", itercount, dvar / b_2norm);
		}

		MPI_Bcast(&dvar, 1, MPI_DOUBLE, MASTER, comm);
		if (dvar < TOL) {
			goto COMPLETE;
		}
		else
			goto ONE;
	}
	/*---------------------------------------------------------------------------
	* If RCI_request=3, then apply the preconditioner on the vector
	* tmp[ipar[21]-1] and put the result in vector tmp[ipar[22]-1]
	*---------------------------------------------------------------------------
	* NOTE that ipar[21] and ipar[22] contain FORTRAN style addresses,
	* therefore, in C code it is required to subtract 1 from them to get C style
	* addresses
	*---------------------------------------------------------------------------*/
	if (RCI_flag == 3)
	{
		if (taskid == MASTER) {
			temp_1 = &tmp[ipar[21] - 1];
			temp_2 = &tmp[ipar[22] - 1];
		}
        else
            temp_1 = new double[m];
        MPI_Bcast(temp_1, m, MPI_DOUBLE, MASTER, comm);
		/* start modification 3: solve L U temp_2 = temp_1   */
		//*************************** PART 2 ************************************
		// Step 2: Backward sweep using pipeline, Solve y for L[i,i] * y[i] = B[i]
		//B[id] = Seg(temp_1, dim * dim, dim, id * dim, true);
        
        B[id] = Seg(temp_1, row, dim, id * dim, true);
		for (i = 0; i < id; i++) {
			B[i] = new double[dim];
			MPI_Recv(&B[i][0], dim, MPI_DOUBLE, i, 7 * core * core + i, comm, &status);
			cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, -1, L[id][i], dim, B[i], 1, 1, B[id], 1);
		}
		// Solve L[i][i] * y[i] = b[i] and Bcast to other core
		cblas_dtrsv (CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit, dim, L[id][id], dim, B[id], 1);
		for (i = id + 1; i < core; i++) {
			MPI_Send(&B[id][0], dim, MPI_DOUBLE, i, 7 * core * core + id, comm);
		}
		// *************************** PART 3 *********************
		// Step 3: Forward sweep using pipeline, Solve x for Ux = y
		for (i = core - 1; i > id; i--) {
			B[i] = new double[dim];
			MPI_Recv(&B[i][0], dim, MPI_DOUBLE, i, 9 * core * core + i, comm, &status);
			cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, -1, BlockA[id][i], dim, B[i], 1, 1, B[id], 1);
		}
		// Solve U[i][i] * x[i] = y[i] and calculate the norm
		cblas_dtrsv (CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, dim, U[id][id], dim, B[id], 1);
		for (i = 0; i < id; i++)
			MPI_Send(&B[id][0], dim, MPI_DOUBLE, i, 9 * core * core + id, comm);
		
		// Final summary
		if (id == 0) {
			//temp_2 = new double [row];
			for (i = 0; i < core; i++)
				for (j = 0; j < dim; j++)
					temp_2[j + i * dim] = B[i][j];
		}
           
		/* end modification 3 */
		goto ONE;
	}
	/*---------------------------------------------------------------------------
	* If RCI_request=4, then check if the norm of the next generated vector is
	* not zero up to rounding and computational errors. The norm is contained
	* in dpar[6] parameter
	*---------------------------------------------------------------------------*/
	if (RCI_flag == 4)
	{
		if (taskid == MASTER)
			dvar = dpar[6];
		MPI_Bcast(&dvar, 1, MPI_DOUBLE, MASTER, comm);
		if (dvar <1.0E-12) {
			goto COMPLETE;
		}
		else
			goto ONE;
	}
	/*---------------------------------------------------------------------------
	* If RCI_request=anything else, then dfgmres subroutine failed
	* to compute the solution vector: computed_solution[N]
	*---------------------------------------------------------------------------*/
	else
	{
		goto FAILED;
	}
	/*---------------------------------------------------------------------------
	* Reverse Communication ends here
	* Get the current iteration number and the FGMRES solution (DO NOT FORGET to
	* call dfgmres_get routine as computed_solution is still containing
	* the initial guess!). Request to dfgmres_get to put the solution
	* into vector computed_solution[N] via ipar[12]
	*---------------------------------------------------------------------------*/
COMPLETE:if (taskid == MASTER) {
	    ipar[12] = 0;
    	dfgmres_get(&ivar, computed_solution, b, &RCI_request, ipar, dpar, tmp, &itercount);
	    /*---------------------------------------------------------------------------
    	* Print solution vector: computed_solution[N] and the number of iterations: itercount
	    *---------------------------------------------------------------------------*/
    	printf("The system has been solved in %d iterations \n", itercount);
	    printf("The following solution has been obtained (first 4 elements): \n");
	    for (i = 0; i < 4; i++)
    	{
	    	printf("computed_solution[%d]=", i);
		    printf("%e\n", computed_solution[i]);
    	}

        fileWrite(file[2], computed_solution, m);
    	/*-------------------------------------------------------------------------*/
	    /* Release internal MKL memory that might be used for computations         */
    	/* NOTE: It is important to call the routine below to avoid memory leaks   */
	    /* unless you disable MKL Memory Manager                                   */
    	/*-------------------------------------------------------------------------*/
	    MKL_Free_Buffers();
    	temp_1 = computed_solution;
        temp_2 = residual;
    }
		 
    // compute the relative residual
    mpi_dgemv(m, local_m, local_A, temp_1, temp_2, local_u, local_v, taskid, comm);
    if (taskid == MASTER) {
        dvar = -1.0E0;
        cblas_daxpy(ivar, dvar, b, incx, residual, incx);
	    dvar = cblas_dnrm2(ivar, residual, incx);
		printf("relative residual:%e\n", dvar / b_2norm);

	    if (itercount<MAXIT && dvar<TOL)
				 flag = 0; //success
			 else
				 flag = 1; //fail
		}

		 MPI_Bcast(&flag, 1, MPI_INT, MASTER, comm);

		 free(local_A);
		 free(local_u);
		 free(local_v);
		 if (taskid == MASTER) {
			 free(tmp);
			 free(b);
			 free(computed_solution);
			 free(residual);
		 }

		 if (flag == 0) {
			 MPI_Finalize();
			 return 0;
		 }
		 else {
			 MPI_Finalize();
			 return 1;
		 }
		 /* Release internal MKL memory that might be used for computations         */
		 /* NOTE: It is important to call the routine below to avoid memory leaks   */
		 /* unless you disable MKL Memory Manager                                   */
		 /*-------------------------------------------------------------------------*/
	 FAILED:
		 if (taskid == MASTER) {
			 printf("\nThis example FAILED as the solver has returned the ERROR code %d", RCI_request);
			 MKL_Free_Buffers();
		 }
		 free(local_A);
		 free(local_u);
		 free(local_v);
		 if (taskid == MASTER) {
			 free(tmp);
			 free(b);
			 free(computed_solution);
			 free(residual);
		 }

		 MPI_Finalize();
		 return 1;





}


void mpi_dgemv(const MKL_INT m, const MKL_INT local_m, const double *local_A, const double *u, double *v, double *local_u, double *local_v, int taskid, MPI_Comm comm) {
	// compute v=A*u in MPI
	CBLAS_LAYOUT    layout = CblasColMajor; //col major
	CBLAS_TRANSPOSE trans = CblasNoTrans; // no transfer

	MPI_Scatter(u, local_m, MPI_DOUBLE, local_u, local_m, MPI_DOUBLE, MASTER, comm);
	cblas_dgemv(layout, trans, m, local_m, 1.0, local_A, m, local_u, 1, 0.0, local_v, 1);

	//  Apply a reduction operation on all nodes and place the result in vector v.
	MPI_Reduce(local_v, v, m, MPI_DOUBLE, MPI_SUM, MASTER, comm);
}
