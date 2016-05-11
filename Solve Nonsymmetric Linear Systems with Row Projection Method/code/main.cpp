
static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
	-random_exact_sol : use a random exact solution vector\n\
	-view_exact_sol : write exact solution vector to stdout\n\
	-m <mesh_x> : number of mesh points in x-direction\n\
	-n <mesh_n> : number of mesh points in y-direction\n\n";

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <petscksp.h>
#include <time.h> 
#include <omp.h>
#include "mpi.h"
#include "mkl.h"
#include "petscmat.h"
#include "petscis.h"


#include "rcm.hpp"

// Wei's Library
#include "sparseLib.hpp"  
// Kai's Library
#include "lsCG.hpp"


/***************************************************
* Solve Ax = b <=> PAx = Pb
*              <=> PAP'y=Pb where x = P'y
*              <=> By = b1 where u = Pb, B is banded
* Use RCM to get B from A
*
* Permutate 2nd time: PBy = Pb1
*                <=>  Cy = b2
* 
* Set up Least Square Problem
*   Use CGLS method to solve that
*
* Build the CG framework that includes the LS
*
* Get the solution
*
**************************************************/


using namespace std;

const int outerSeg = 2;
const int innerSeg = 4;

typedef double FLOAT_TYPE;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
	int i, j, k, m, n, p, q;
	int nz, i0, i1, i2, ncols;
	int block_r, botom_r, Istart, Iend;
	int maxSupBand, maxSubBand;
	int upBand[outerSeg], lowBand[outerSeg];
	int id, core, local_r, dimPETSc;
	int fd, tmp0, tmp1;
	Vec x, b[3], u;         /* approx solution, RHS, exact solution */
	Mat A;                  /* linear system matrix */
	Mat Ap[outerSeg];       /* subMatrices A[i][j] */
	KSP	ksp;                /* linear solver context */
	PetscRandom	rctx;       /* random number generator context */
	PetscReal norm;         /* norm of solution error */
	IS  rowperm, colperm;   /* row and column permutations */
	PetscErrorCode ierr;
	PetscBool flg = PETSC_FALSE;
	PetscScalar	v, *vals, *coo_v;
    PetscScalar diff, diffin;
    PetscReal normAi[outerSeg];
	PetscInt *coo_I, *coo_J;
	PetscInt *perm, *permInv, *perm2nd, *mRow;
	MatOrderingType ordering = MATORDERINGRCM;
	PetscViewer view_out, view_in;
	PetscBool PetscPreLoad = PETSC_FALSE;
    PetscReal tolerance1 = 1.e-3;
	clock_t timestep[5];
    FILE *fp;
    #if defined(PETSC_USE_LOG)
	PetscLogStage stage[2];
    #endif

	PetscInitialize(&argc, &argv, (char*)0, help);
	MPI_Comm_rank(PETSC_COMM_WORLD, &id);
	MPI_Comm_size(PETSC_COMM_WORLD, &core);

	PetscLogStageRegister("Reverse Cuthill Mckee", &stage[0]);
	PetscLogStageRegister("Second Permutation", &stage[1]);

	if (id == 0) {
		printf("I will be starting the clock after a barrier\n");
		printf("Outer partition: %d, Inner blocks: %d \n", outerSeg, innerSeg);
		timestep[0] = clock();
	}

	/**************************************************************************************/
	/***************** Step 0: Reverse Cuthill Mckee **************************************/
	PetscLogStagePush(stage[0]);

	// Read the matrix with COO format
	fp = fopen(argv[1], "r");
	// Get matrix parameters
	getPara(fp, &m, &n, &nz, id, core);

	// Creat vector, b_0,1,2 represent original, 1st ordering, 2nd ordering RHS
	for (i = 0; i < 3; i++) {
		VecCreate(PETSC_COMM_WORLD, &b[i]);
		VecSetSizes(b[i], PETSC_DECIDE, m);
		VecSetFromOptions(b[i]);
	}
	VecDuplicate(b[0], &u);
	VecDuplicate(b[0], &x);

	// Set u, use a vector with all 1.0 by default
	if (flg) {
		PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
		PetscRandomSetFromOptions(rctx);
		VecSetRandom(u, rctx);
		PetscRandomDestroy(&rctx);
	}
	else {
		VecSet(u, 1.0);
	}

	// Create Matrix   :: A ::
	MatCreate(PETSC_COMM_WORLD, &A);
	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n);
	MatSetFromOptions(A);

	// set roughly thresholds for memory allocation
	memory_set(m, &dimPETSc);

	// Type in number of nonzeros per row (same for all rows)
	MatSeqAIJSetPreallocation(A, dimPETSc, NULL);
	MatMPIAIJSetPreallocation(A, dimPETSc / 4, NULL, dimPETSc, NULL);
	MatGetOwnershipRange(A, &Istart, &Iend);

	// Initialize memory for COO format
	PetscMalloc1(nz, &coo_I);
	PetscMalloc1(nz, &coo_J);
	PetscMalloc1(nz, &coo_v);

	// Load matrix file with COO format
	load_COO_FILE(fp, nz, coo_I, coo_J, coo_v);
	fclose(fp);

	// Initialize permutation vector
	perm = new int[m];
	permInv = new int[m];

	// Import permutation vector and set value to A
	// Use permutator in Matlab by default, flg : PETSC_FALSE
    // Calculation of rcm.cpp is included to show the RCM time. flg : PETSC_TRUE
	setValueA(argv[2], A, m, nz, core, id, coo_I, coo_J, coo_v, perm, permInv, &diffin, PETSC_TRUE);
	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

	// Generate RHS b[1], rather than b[0], since A is already banded.
	MatMult(A, u, b[1]);

	// Extract vals from b[1]
	PetscMalloc1(Iend - Istart, &vals);
	VecGetArray(b[1], &vals);

	// Get original b[0] derived from b[1]
	permuteVec(b[0], Istart, Iend, perm, vals);
	VecAssemblyBegin(b[0]);
	VecAssemblyEnd(b[0]);

	PetscFree(coo_I);
	PetscFree(coo_J);
	PetscFree(coo_v);

	if (id == 0) {
		timestep[1] = clock();
		diff = ((float)(timestep[1] - timestep[0])) / CLOCKS_PER_SEC;
		printf("Matrix has been loaded, time using %.2fs\n", diff - diffin);
        printf("Reverse Cuthill Mckee is complete, time using %.2fs\n", diffin);
	}
	/**************************************************************************************/
	/***************** Step 1: Make the 2nd reordering ************************************/

	PetscLogStagePop();
	PetscLogStagePush(stage[1]);

	// Determine the number of rows in A[i]
	mRow = getIndex(m, outerSeg, innerSeg);

	// Contruct matrices array Ap[4]
	for (i = 0; i < outerSeg; i++) {
		MatCreate(PETSC_COMM_WORLD, &Ap[i]);
		MatSetSizes(Ap[i], PETSC_DECIDE, PETSC_DECIDE, mRow[i], n);
		if (id == 0) printf("Submatrix %d has rows %d\n", i, mRow[i]);
		MatSetFromOptions(Ap[i]);
		// !!! Caution: This is a roughly estimate
		MatSeqAIJSetPreallocation(Ap[i], dimPETSc, NULL);
		MatMPIAIJSetPreallocation(Ap[i], dimPETSc / 4, NULL, dimPETSc, NULL);
	}

	for (i = 0; i < core; i++) {
		upBand[i] = 0;
		lowBand[i] = 0;
	}

	// Generate Ap[4] with independent parts
	SegMat(A, Ap, m, Istart, Iend, innerSeg, outerSeg, upBand, lowBand, id);
	printf("Node id = %d SupBandWidth = %d, SubBandWidth = %d\n", id, upBand[id], lowBand[id]);

	// Extract vals from b[1]

	VecGetArray(b[1], &vals);

	// Get the permutation vector globally
	perm2nd = permute2nd(m, innerSeg, outerSeg, mRow);

	// Permute vector just like what matrix did
	permuteVec(b[2], Istart, Iend, perm2nd, vals);
	VecAssemblyBegin(b[2]);
	VecAssemblyEnd(b[2]);

	//VecView(b[2],PETSC_VIEWER_STDOUT_WORLD);

	// Assemble matrix
	for (i = 0; i < outerSeg; i++) {
		MatAssemblyBegin(Ap[i], MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(Ap[i], MAT_FINAL_ASSEMBLY);
        // Compute the norm of each A[i]
        MatNorm(Ap[i], NORM_FROBENIUS, &normAi[i]);
        //MatView(Ap[i],PETSC_VIEWER_STDOUT_WORLD);
        if (id == 0)
            printf("A[%d] norm is %f\n", i, normAi[i]);
	}
	PetscLogStagePop();

	if (id == 0) {
		timestep[2] = clock();
		diff = ((float)(timestep[2] - timestep[1])) / CLOCKS_PER_SEC;
		printf("Permutation step is complete, time using %.2fs\n", diff);
	}

	/* End Wei's code, begin Nate's code */
    /**************************************************************************************/
	/***************** Step 2: Least Squares Problem Setup ********************************/
	p = outerSeg;
	Mat            B[p]; // Transpose of Ai          
	KSP            Apksp[p];
	PC             pc[p];
	PetscInt 	   P = p;
	PetscReal      tol = 1.e-8;
	PetscScalar    one = 1.0;
	Vec			   xp[p];

	Mat AtA[p];

	for (i = 0; i < p; i++) {
		VecCreate(PETSC_COMM_WORLD, &xp[i]);
		VecSetSizes(xp[i], PETSC_DECIDE, mRow[i]);
		VecSetFromOptions(xp[i]);

		PetscScalar q = .5;
		VecSet(xp[i], q);
	}

	if (id == 0) {
		timestep[3] = clock();
		diff = ((float)(timestep[3] - timestep[2])) / CLOCKS_PER_SEC;
		printf("Setting up least squares stuff is done, time using %.2fs\n", diff);
	}


	/* Let's try to get a preconditioner going for the least squares problem */
	// Need to form AtA[0] = A0'*A0 and AtA[1] = A0'*A0
    /**************************************************************************************/
	/******************Step 3: Set up for the CG scheme ***********************************/
	PetscInt       its;

	flg = PETSC_FALSE;
	// PetscOptionsGetBool(NULL,"-view_exact_sol",&flg,NULL);
	if (flg) { VecView(b[2], PETSC_VIEWER_STDOUT_WORLD);  }

	if (id == 0) printf("Ready to compute the right hand side c in a cheating way \n");

	/****************************** TO EDIT
	Form right hand side vector c. Really the way we should do this is to somehow perform:

	c = Tf = A'*(D+L)^{-1}D*(D+L)^{-1}f,

	but for simplicity for right now we won't do that.


	For now, I'm just going to cheat and form c by multiplying (I-Q)*ones(n,1)
	******************************* TO EDIT   ************************************/

	// Cheating way to get c
	Vec c;	// new right hand side vector
	Vec e;	// all ones vector
	VecDuplicate(u, &c); 
	VecDuplicate(u, &e); 
	VecSet(c, 1.0);  //set it to be just ones
	VecSet(e, 1.0);  //set it to be just ones

	//myleastsquare(B, c, Apksp, p, mRow, xp);
	QpkCGLS_FUNCTION(Ap, c, xp, tolerance1, p, n);
	VecAYPX(c, -1.0, e);
	//VecView(c,0);


	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	Conjugate Gradient Framework
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

	/* Declare variables I'm going to need for CG scheme */

	int r;
	Vec rk; 			// residual
	Vec pk; 			// search direction
	Vec Apk;			// vector obtained by multiplying A (=I-Q) by pk
	PetscReal Beta;		// scalar r(k+1)^T *r(k+1)/(rk^Trk)
	PetscReal alpha;	// another scalar, see below
	PetscReal adenom;	// the denominator of alpha
	PetscReal bnum;		// the numerator of beta
	PetscReal anum;		// the numerator of alpha
	PetscReal RelRes;	// the norm of relative residual, for convergence purposes

						//For checking the residual at the end:
	PetscReal normB;	// the norm of the right hand side vector, ends up being ||r0||_2
	PetscReal normRes;	// the residual norm: ||rk||_2

	Vec temp;			// save a version of pk

						/* Allocate space for the other vectors */
	VecDuplicate(u, &rk); 
	VecDuplicate(rk, &pk); 
	VecDuplicate(pk, &Apk); 
	//VecDuplicate(pk,&temp);

	/* Initialize first variables */

	k = 0;
	VecCopy(c, x);  					//initialize x0 to be c

	VecCopy(c, rk);  				// r0 := c

	//printf("Am I trying to compute least squares?\n");

	QpkCGLS_FUNCTION(Ap, rk, xp, tolerance1, p, n);

	//myleastsquare(B, rk, Apksp, p, mRow, xp);	// r0 := Q*r0 = Q*c;

	VecCopy(rk, pk); 				// p0 := r0 
	VecNorm(c, NORM_2, &normB);							// the norm of rhs: ||r0||_2

	/* Conjugate Gradient Loop */
	RelRes = 1;

	while (RelRes > tol && k < 100) {
        // Apk := A*pk
		MatMult(A, pk, Apk);
        // Apk_0 := pk;

		VecCopy(pk, Apk);
		
        //myleastsquare(B, Apk, Apksp, p, mRow, xp); 	// Apk_1 := Q*pk
		QpkCGLS_FUNCTION(Ap, Apk, xp, tolerance1, p, n);

        // Apk_2 = pk - Apk_1,
		VecAYPX(Apk, -1.0, pk);	

        // i.e. Apk = pk - Qpk
        // overall we have just performed Apk: = pk - Qpk = (I-Q)pk
        // pk has not changed, and Apk = (I-Q)pk;


        // adenom := (pk)'*(Apk)
		VecDot(pk, Apk, &adenom);		
        // anum := rk'*rk
		VecDot(rk, rk, &anum);		
        // set alpha
		alpha = anum / adenom;

        // x_{k+1} = alpha*p_k + 1*x_k
		VecAXPBY(x, alpha, 1, pk);	
        // r_{k+1} = -alpha*A*p_k + 1*r_k
		VecAXPBY(rk, -alpha, 1, Apk);

        // bnum := r_{k+1}'*r_{k+1}
		VecDot(rk, rk, &bnum);	

        // Beta = [r(k+1)'*r(k+1)]/(rk'*rk)
		Beta = bnum / anum;	

        // p_{k+1} = r_{k+1} + Beta*p_k
		VecAXPBY(pk, 1, Beta, rk);		
		k++;

		//Checking the residual:
        // the norm of rhs: ||rk||_2
		VecNorm(rk, NORM_2, &normRes);
        // relative residual for convergence
		RelRes = normRes / normB;

		if (id == 0) {
			printf("Residual at %d: %.10f \n", k, RelRes);
		}
	}
	MPI_Barrier(PETSC_COMM_WORLD);

	if (id == 0) {
		timestep[4] = clock();
		printf("Final Relative residual is %le \n", RelRes);
		printf("Total number of iterations: %d \n", k);

		diff = ((float)(timestep[4] - timestep[3])) / CLOCKS_PER_SEC;
		printf("CG acceleration is done, time using %.2fs\n", diff);
		printf("Total Time used for solving system: %.2fs \n", ((float)(timestep[4] - timestep[0])) / CLOCKS_PER_SEC);
	}
	//VecView(x, PETSC_VIEWER_STDOUT_WORLD);

	/* End Nate's code, begin Wei's code */
	VecDestroy(&b[1]);
	//MatView(A,PETSC_VIEWER_STDOUT_WORLD);
	//MatView(Ap[1],PETSC_VIEWER_STDOUT_WORLD);
	//delete[] vals[1];

	/**************************************************************************************/
	/**************************************************************************************/
	/**************************************************************************************/


	// When you get vector y, use the following to get u.
	// VecGetArray(y, &vals);
	// Get approximate solution u derived from y
	// permuteVec(u, Istart, Iend, perm, vals);
	// VecAssemblyBegin(u);
	// VecAssemblyEnd(u);

	MatDestroy(&A);
    VecDestroy(&rk);
    VecDestroy(&pk);
    VecDestroy(&Apk);
    for (i = 0; i < outerSeg; i++) {
        MatDestroy(&Ap[i]);
        VecDestroy(&xp[i]);
    }
    delete[] perm;
    delete[] permInv;
    delete[] perm2nd;
    delete[] mRow;

	PetscFinalize();
	return 0;
}

