/* Example to show how to use Intel's FGMRES with preconditioner to solve the linear system Ax=b in MPI.
 * Based on Intel's example: solverc/source/fgmres_full_funct_c.c
 * For CS51501 HW3 Part b 
 * 
 * Please read Intel Reference Manual, Chapter 6 Sparse Solve Routine, FGMRES Interface Description for the detail information.
 */
#include <stdio.h>
#include "mkl.h"
#include "mpi.h"

#define MASTER 0               // taskid of first task 
#define RESTART 500
#define TOL 0.00000001
#define MAXIT 1000


void mpi_dgemv(const MKL_INT m, const MKL_INT local_m,const double *A,const double *u, double *v, double *local_u, double *local_v,int taskid, MPI_Comm comm);
void mpi_preconditioner_solver(const MKL_INT m, const MKL_INT local_m, const double *local_M, const double *u, double *v, double * local_u,int taskid, MPI_Comm comm);


int main (int argc, char *argv[])
{

	int	taskid;              // a task identifier 
	int	numtasks;              // number of tasks in partition
	MPI_Comm comm;
	int m; // size of the matrix
	int	local_m;                  // rows of matrix A sent to each worker 
	double *A, *b,*exact_x, *x;
	double *temp_1, *temp_2;
	double *local_A,  *local_v,*local_u;
	double *local_M; // M is the preconditioner in this example, which is the diagonal element of A;
	int i,j,k;


	MPI_Init(&argc,&argv);
	comm=MPI_COMM_WORLD;
	MPI_Comm_rank(comm,&taskid);
	MPI_Comm_size(comm,&numtasks);
	if(taskid==MASTER){ // initilization: A and b
		/* start modification 1: read A and b from mtx files in node 0 */
		m=64; // size of the matrix
		A=(double *)malloc(sizeof(double)*(m*m));
		// !!! A is in col-major
		for(j=0;j<m;j++)
			for (i=0;i<m;i++){
				if(i==j)
					*(A+j*m+i)=m*100.0;
				else
					*(A+j*m+i)=i+1.0;
			}
		exact_x=(double *)malloc(sizeof(double)*m);
		for (i=0;i<m;i++)
			*(exact_x+i)=1.0;
		b=(double *)malloc(sizeof(double)*m);
		// b=A*ones(n,1)
		cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1.0, A, m, exact_x, 1, 0.0, b, 1); 	
		/* end modification 1*/
	}

	MPI_Bcast(&m,1,MPI_INT, MASTER, comm); // send m from node MASTER to all other nodes.
	local_m=m/numtasks;
	local_A=(double *)malloc(sizeof(double)*(local_m*m));
	local_u=(double *)malloc(sizeof(double)*(local_m));
	local_v=(double *)malloc(sizeof(double)*m);
	//	partition A and send A_i to local_A on  node i
	MPI_Scatter(A,local_m*m, MPI_DOUBLE, local_A, local_m*m, MPI_DOUBLE, MASTER, comm);

	if(taskid==MASTER){
		free(A);
		free(exact_x);
		// do not free b, it wil be used  for GMRES
	}


	/* start modification 2: generate preconditioner M
	 * In this example, TA choose the diagonal elements of A as the preconditioner.
	 * In HW3 part b, you should generate L and U here.
	 */
	local_M=(double *)malloc(sizeof(double)*local_m);
	for(i=0;i<local_m;i++)
		*(local_M+i)=*(local_A+taskid*local_m+i*m+i);
	/* end  modification 2*/


	/*---------------------------------------------------------------------------
	 * GMRES: Allocate storage for the ?par parameters and the solution vectors
	 *---------------------------------------------------------------------------*/
	MKL_INT RCI_request;
	int RCI_flag;
	double dvar;
	int flag=0;

	MKL_INT ipar[128]; //specifies the integer set of data for the RCI FGMRES computations
	double dpar[128]; // specifies the double precision set of data
	double *tmp; //used to supply the double precision temporary space for theRCI FGMRES computations, specifically:
	double *computed_solution;
	double *residual;
	double *f;
	MKL_INT itercount, ierr=0;;
	MKL_INT  ivar;
	double b_2norm;
	char cvar='N';
	MKL_INT incx=1;
	if (taskid==MASTER){
		ipar[14]=RESTART; // restart iteration number
		int n_tmp = (2 * ipar[14] + 1) * m + ipar[14] * (ipar[14] + 9) / 2 + 1;
		tmp=(double *)malloc(sizeof(double)*n_tmp);
		computed_solution=(double *)malloc(sizeof(double)*m);
		residual=(double *)malloc(sizeof(double)*m);
		f=(double *)malloc(sizeof(double)*m);

		ivar=m;
		/*---------------------------------------------------------------------------
		 * Initialize the initial guess
		 *---------------------------------------------------------------------------*/
		for (i = 0; i < m; i++)
		{
			computed_solution[i] = 0.5;
		}

		b_2norm = cblas_dnrm2 (ivar, b, incx);
	//	printf("b_2norm=%f\n",b_2norm);
		/*---------------------------------------------------------------------------
		 * Initialize the solver
		 *---------------------------------------------------------------------------*/
		dfgmres_init (&ivar, computed_solution,b, &RCI_request, ipar, dpar, tmp);
		RCI_flag=RCI_request;
	}
	MPI_Bcast(&RCI_flag,1,MPI_INT, MASTER, comm);
	if (RCI_flag != 0)
		goto FAILED;

	if(taskid==MASTER){
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
		dfgmres_check (&ivar, computed_solution, b, &RCI_request, ipar, dpar,
				tmp);
		RCI_flag=RCI_request;
	}

	MPI_Bcast(&RCI_flag,1,MPI_INT, MASTER, comm);
	if (RCI_flag != 0)
		goto FAILED;

	if (taskid==MASTER){
		/*---------------------------------------------------------------------------
		 * Print the info about the RCI FGMRES method
		 *---------------------------------------------------------------------------*/
		printf ("Some info about the current run of RCI FGMRES method:\n\n");
		if (ipar[7])
		{
			printf ("As ipar[7]=%d, the automatic test for the maximal number of ", ipar[7]);
			printf ("iterations will be\nperformed\n");
		}
		else
		{
			printf ("As ipar[7]=%d, the automatic test for the maximal number of ", ipar[7]);
			printf ("iterations will be\nskipped\n");
		}
		printf ("+++\n");
		if (ipar[8])
		{
			printf ("As ipar[8]=%d, the automatic residual test will be performed\n", ipar[8]);
		}
		else
		{
			printf ("As ipar[8]=%d, the automatic residual test will be skipped\n", ipar[8]);
		}
		printf ("+++\n");
		if (ipar[9])
		{
			printf ("As ipar[9]=%d, the user-defined stopping test will be ", ipar[9]);
			printf ("requested via\nRCI_request=2\n");
		}
		else
		{
			printf ("As ipar[9]=%d, the user-defined stopping test will not be ", ipar[9]);
			printf ("requested, thus,\nRCI_request will not take the value 2\n");
		}
		printf ("+++\n");
		if (ipar[10])
		{
			printf ("As ipar[10]=%d, the Preconditioned FGMRES iterations will be ", ipar[10]);
			printf ("performed, thus,\nthe preconditioner action will be requested via ");
			printf ("RCI_request=3\n");
		}
		else
		{
			printf ("As ipar[10]=%d, the Preconditioned FGMRES iterations will not ", ipar[10]);
			printf ("be performed,\nthus, RCI_request will not take the value 3\n");
		}
		printf ("+++\n");
		if (ipar[11])
		{
			printf ("As ipar[11]=%d, the automatic test for the norm of the next ", ipar[11]);
			printf ("generated vector is\nnot equal to zero up to rounding and ");
			printf ("computational errors will be performed,\nthus, RCI_request will not ");
			printf ("take the value 4\n");
		}
		else
		{
			printf ("As ipar[11]=%d, the automatic test for the norm of the next ", ipar[11]);
			printf ("generated vector is\nnot equal to zero up to rounding and ");
			printf ("computational errors will be skipped,\nthus, the user-defined test ");
			printf ("will be requested via RCI_request=4\n");
		}
		printf ("+++\n\n");
	}
	/*---------------------------------------------------------------------------
	 * Compute the solution by RCI (P)FGMRES solver with preconditioning
	 * Reverse Communication starts here
	 *---------------------------------------------------------------------------*/
ONE:
	if(taskid==MASTER){
		dfgmres (&ivar, computed_solution,b, &RCI_request, ipar, dpar, tmp);
		RCI_flag=RCI_request;  
	}
	MPI_Bcast(&RCI_flag,1,MPI_INT, MASTER, comm); // send RCI_request from node MASTER to all other nodes.
	
	/*---------------------------------------------------------------------------
	 * If RCI_request=0, then the solution was found with the required precision
	 *---------------------------------------------------------------------------*/
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
		if (taskid==MASTER){
			temp_1=&tmp[ipar[21] - 1];
			temp_2=&tmp[ipar[22] - 1];
		}

		mpi_dgemv(m,local_m,local_A,temp_1, temp_2,local_u,local_v,taskid, comm);

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
		if (taskid==MASTER){
			ipar[12] = 1;
			/* Get the current FGMRES solution in the vector f */
			dfgmres_get (&ivar, computed_solution, f, &RCI_request, ipar, dpar, tmp, &itercount);
			temp_1=f;
			temp_2=residual;
		} 
		/* Compute the current true residual via mpi mat_vec multiplication */
		mpi_dgemv(m,local_m,local_A,temp_1,temp_2,local_u,local_v,taskid, comm);

		if(taskid==MASTER){
			dvar = -1.0E0;
			cblas_daxpy (ivar, dvar, b, incx, residual, incx);
			dvar = cblas_dnrm2 (ivar, residual, incx);
			printf("iteration %d, relative residual:%e\n",itercount, dvar);
		}

		MPI_Bcast(&dvar,1,MPI_DOUBLE, MASTER, comm); 
		if (dvar < TOL){
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
		if (taskid==MASTER){
			temp_1=&tmp[ipar[21] - 1];
			temp_2=&tmp[ipar[22] - 1];
		}
		/* start modification 3: solve L U temp_2 = temp_1   */
		mpi_preconditioner_solver(m,local_m,local_M,temp_1, temp_2,local_u,taskid,comm);
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
		if(taskid==MASTER)
			dvar=dpar[6];
		MPI_Bcast(&dvar,1,MPI_DOUBLE, MASTER, comm); 
		if (dvar <1.0E-12 ){
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
COMPLETE:if(taskid==MASTER){
			 ipar[12] = 0;
			 dfgmres_get (&ivar, computed_solution,b, &RCI_request, ipar, dpar, tmp, &itercount);
			 /*---------------------------------------------------------------------------
			  * Print solution vector: computed_solution[N] and the number of iterations: itercount
			  *---------------------------------------------------------------------------*/
			 printf ("The system has been solved in %d iterations \n", itercount);
			 printf ("The following solution has been obtained (first 4 elements): \n");
			 for (i = 0; i < 4; i++)
			 {
				 printf ("computed_solution[%d]=", i);
				 printf ("%e\n", computed_solution[i]);
			 }

			 /*-------------------------------------------------------------------------*/
			 /* Release internal MKL memory that might be used for computations         */
			 /* NOTE: It is important to call the routine below to avoid memory leaks   */
			 /* unless you disable MKL Memory Manager                                   */
			 /*-------------------------------------------------------------------------*/
			 MKL_Free_Buffers ();
			 temp_1=computed_solution;
			 temp_2=residual;
		 }
		// compute the relative residual
		 mpi_dgemv(m,local_m,local_A,temp_1,temp_2,local_u,local_v,taskid, comm);
		 if(taskid==MASTER){
			 dvar = -1.0E0;
			 cblas_daxpy (ivar, dvar, b, incx, residual, incx);
			 dvar = cblas_dnrm2 (ivar, residual, incx);
			 printf("relative residual:%e\n",dvar/b_2norm);

			 if(itercount<MAXIT && dvar<TOL)
				 flag=0; //success
			 else 
				 flag=1; //fail

		 }

		 MPI_Bcast(&flag,1,MPI_INT, MASTER, comm); 
		
		free(local_A);
		free(local_M);
		free(local_u);
		free(local_v);
		if(taskid==MASTER){
			free(tmp);
			free(b);
			free(computed_solution);
			free(residual);
		}
		
		if(flag==0){
			 MPI_Finalize();
			 return 0;
		 }
		 else{
			 MPI_Finalize();
			 return 1;
		 }
		 /* Release internal MKL memory that might be used for computations         */
		 /* NOTE: It is important to call the routine below to avoid memory leaks   */
		 /* unless you disable MKL Memory Manager                                   */
		 /*-------------------------------------------------------------------------*/
FAILED:
		 if(taskid==MASTER){
			 printf ("\nThis example FAILED as the solver has returned the ERROR code %d", RCI_request);
			 MKL_Free_Buffers ();
		 }
		free(local_A);
		free(local_M);
		free(local_u);
		free(local_v);
		if(taskid==MASTER){
			free(tmp);
			free(b);
			free(computed_solution);
			free(residual);
		}

		 MPI_Finalize();
		 return 1;





}


void mpi_dgemv(const MKL_INT m, const MKL_INT local_m, const double *local_A,const double *u, double *v, double *local_u, double *local_v,int taskid, MPI_Comm comm){
	// compute v=A*u in MPI
	CBLAS_LAYOUT    layout=CblasColMajor; //col major
	CBLAS_TRANSPOSE trans=CblasNoTrans; // no transfer

	MPI_Scatter(u,local_m, MPI_DOUBLE, local_u, local_m, MPI_DOUBLE, MASTER, comm   ); // send u_i from node MASTER to all other nodes.
	//	printf("scatter finish at taskid=%d\n",taskid);
	// compute A_i
	cblas_dgemv(layout, trans, m, local_m, 1.0, local_A, m, local_u, 1, 0.0, local_v, 1);  

	//  Apply a reduction operation on all nodes and place the result in vector v.
	MPI_Reduce( local_v, v, m, MPI_DOUBLE, MPI_SUM, MASTER,comm      );
}


void mpi_preconditioner_solver(const MKL_INT m, const MKL_INT local_m, const double *local_M, const double *u, double *v, double *local_u,int taskid, MPI_Comm comm){
	int i=0;
	//	printf("begin taskid=%d\n",taskid);
	MPI_Scatter(u,local_m, MPI_DOUBLE, local_u, local_m, MPI_DOUBLE, MASTER, comm   ); // send u_i from node MASTER to all other nodes.
	//	printf("taskid=%d\n",taskid);
	//compute Mi^(-1)*y_i at each node
	for (i=0;i<local_m;i++)
		*(local_u+i)/=*(local_M+i);

	// Apply a gather operation on all nodes 
	MPI_Gather(local_u, local_m, MPI_DOUBLE, v,local_m, MPI_DOUBLE,MASTER,comm);


}


