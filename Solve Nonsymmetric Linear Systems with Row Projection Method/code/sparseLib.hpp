#include "rcm.hpp"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <ctime>
#include <cstring>


typedef double FLOAT_TYPE;


/*****************************************************************************************************************/
//
//  Modified:
//    24 April 2016
//
//  Author:
//
//    Wei Deng
//
/*****************************************************************************************************************/

// Inintialize the basic parameters
void getPara(FILE *fp, int *m, int *n, int *nz, int id, int core);

// Read the COO data
void load_COO_FILE(FILE *fp, int nz, PetscInt *coo_I, PetscInt *coo_J, PetscScalar *coo_v);

// Three ways to get the permutation vector
// The two from PETSc and rcm.cpp can't get a perfect band
// Use permutation vector in Matlab instead in the project
void calc_Permu_Vec(int m, int nz, PetscInt *coo_I, PetscInt *coo_J, PetscScalar *coo_v, int *perm, int *permInv);
void load_Permu_Vec(char *argv, int m, int *perm, int *permInv, int id);

// Permute the matrix and vector
void RCM_Permutation(int m, int nz, PetscInt *coo_I, PetscInt *coo_J, int *perm, int *I, int *J);

// Set value of A in parallel, sacrificing a little memory to achieve a much faster speed
void setValueA(char *argv, Mat A, int m, int nz, int core, int id, PetscInt *coo_I, PetscInt *coo_J,
            PetscScalar *coo_v, PetscInt *perm, PetscInt *permInv, PetscScalar *diff, PetscBool flg);

// Get the number of rows in each A[i]
int *getIndex(int m, const int outerSeg, const int innerSeg);
// Partition a banded matrix to some independent parts
void SegMat(Mat A, Mat *Ap, FLOAT_TYPE *v_ori, FLOAT_TYPE *v_tar, int m, int Istart, int Iend, 
            int innerSeg, int outerSeg, int *upBand, int *lowBand, int id);

// Get independent parts
int *permute2nd(int m, const int innerSeg, const int outerSeg, int *mRow);


/*****************************************************************************************************************/
/*****************************************************************************************************************/

// RCM in PETSc donesn't have perfect performance
void RCM_Permutation(int m, int nz, PetscInt *coo_I, PetscInt *coo_J, int *perm, int *I, int *J) {
    int i, j;
    for (i = 0; i < nz; i++) {
        I[i] = perm[coo_I[i]];
        J[i] = perm[coo_J[i]];
    }
}


// vector permutation
void permuteVec(Vec b, int Istart, int Iend, int *perm, PetscScalar *v) {
    int i, new_i;
    PetscScalar value;
    for (i = Istart; i < Iend; i++) {
        value = v[i - Istart];
        new_i = perm[i];
        VecSetValues(b, 1, &new_i, &value, ADD_VALUES);
    }
}

// Get the order and nnz of the matrix
void getPara(FILE *fp, int *m, int *n, int *nz, int id, int core) {
    // Process header with comments
    char buf[PETSC_MAX_PATH_LEN];
    do fgets(buf,PETSC_MAX_PATH_LEN-1,fp);
    while (buf[0] == '%');
    //Get the initial parameters
    sscanf(buf,"%d %d %d\n", m, n, nz);
    if (id == 0) {
        printf("------------------\n");
        printf("\n%d nodes available\n", core);
        printf("%d rows %d cols %d nnz\n", *m, *n, *nz);
        printf("------------------\n");
    }
}

void memory_set(int m, int *dimPETSc) {
    if (m < 1000)
        *dimPETSc = m / 2;
    else if (m < 40000)
        *dimPETSc = m / 20;
    else if (m < 60000)
        *dimPETSc = m / 50;
    else if (m < 100000)
        *dimPETSc = m / 200;
    else if (m < 1000000)
        *dimPETSc = m / 1000;
    else
        *dimPETSc = m / 4000;
}

// Load COO format data
void load_COO_FILE(FILE *fp, int nz, PetscInt *coo_I, PetscInt *coo_J, PetscScalar *coo_v) {
    int i, p, q;
    for (i = 0; i < nz; i++) {
        fscanf(fp,"%d %d %le\n",&p,&q,&coo_v[i]);
        coo_I[i] = p - 1;
        coo_J[i] = q - 1;
    }
}

// Method 1: Get Permutation vector from rcm.cpp
void calc_Permu_Vec(int m, int nz, PetscInt *coo_I, PetscInt *coo_J, PetscScalar *coo_v, int *perm, int *permInv) {
    int i, info;
    int *csr_I = new int[m + 1];
    int *csr_J = new int[nz];
    int *csc_I = new int[nz];
    int *csc_J = new int[m + 1];
    double *csr_v = new double[nz];
    double *csc_v = new double[nz];
    MKL_INT job0[6] = {1,1,0,0,nz,3};
    //MKL_INT job1[6] = {0,1,1,0,nz,3};
    // change COO format to CSR format
    mkl_dcsrcoo (job0, &m, csr_v, csr_J, csr_I, &nz, coo_v, coo_I, coo_J, &info);
    // change CSR format to CSC format
    //mkl_dcsrcsc (job1, &m, csr_v ,csr_J, csr_I, csc_v, csc_J, csc_I, &info);
    if (info != 0)
        printf("info = %d fails\n", info);
    genrcm (m, nz, csr_I, csr_J, perm);
    //genrcm (m, nz, csc_I, csc_J, perm);
    perm_inverse3 (m, perm, permInv);
    for (i = 0; i < m; i++) {
        perm[i] -= 1;
        permInv[i] -= 1;
    }
    delete[]  csr_I;
    delete[]  csr_J;
    delete[]  csc_I;
    delete[]  csc_J;
    delete[]  csr_v;
    delete[]  csc_v;
}

// Method 2: Load Permutation vector from Matlab
void load_Permu_Vec(char *argv, int m, int *perm, int *permInv, int id) {
    int i, p;
    FILE *fp;
    fp = fopen(argv, "r");
    for (i = 0; i < m; i++)
        fscanf(fp,"%d\n",&perm[i]);
    fclose(fp);
    // This function is 1-based index
    perm_inverse3 (m, perm, permInv);

    for (i = 0; i < m; i++) {
        perm[i] -= 1;
        permInv[i] -= 1;
    }
}

// Method 3: from PETSc
//ISCreateGeneral(PETSC_COMM_SELF,m,perm,PETSC_COPY_VALUES,&rowperm);
//ISSetPermutation(rowperm);
//ISView(rowperm,PETSC_VIEWER_STDOUT_SELF);
//MatGetOrdering(A,ordering,&rowperm,&colperm);
//MatPermute(A,rowperm,rowperm,&Ap);

// Set Value of A in parallel, despite sacrificing a little memory
// In total, it gets rid of the process of exporting and importing file
void setValueA(char *argv, Mat A, int m, int nz, int core, int id, PetscInt *coo_I, PetscInt *coo_J, 
            PetscScalar *coo_v, PetscInt *perm, PetscInt *permInv, PetscScalar *diff, PetscBool flg) {
    int i, prows;
    int *I = new int[nz];
    int *J = new int[nz];
    clock_t timestep[2];
    prows = ceil(double(nz) / core);
    //calc_Permu_Vec(m, nz, coo_I, coo_J, coo_v, perm, permInv);  
    if (!flg)
        load_Permu_Vec(argv, m, perm, permInv, id);
    else {
        timestep[0] = clock();
        // use this function just to count the running time of RCM
        calc_Permu_Vec(m, nz, coo_I, coo_J, coo_v, perm, permInv);
        timestep[1] = clock();
        // still use the matlab permutator
        load_Permu_Vec(argv, m, perm, permInv, id);
        *diff = ((float)(timestep[1] - timestep[0])) / CLOCKS_PER_SEC;
    }
    RCM_Permutation(m, nz, coo_I, coo_J, permInv, I, J);
    for (i = id * prows; i < MIN((id + 1) * prows, nz); i++) {
        MatSetValues(A, 1, &I[i], 1, &J[i], &coo_v[i],ADD_VALUES);
    }
    delete [] I;
    delete [] J;
}

// Determine the number of rows in A[i]
int *getIndex(int m, const int outerSeg, const int innerSeg) {
    int i, blocks, avgRow, b, j1;
    int *mRow = new int[outerSeg]();

    blocks = innerSeg * outerSeg;
    avgRow = ceil(float(m) / blocks);

    for (i = 0; i < m; i++) {
        b = i / avgRow;
        j1 = b % outerSeg;
        mRow[j1] += 1;
    }
    return mRow;
}

void SegMat(Mat A, Mat *Ap, int m, int Istart, int Iend, int innerSeg, int outerSeg, int *upBand, int *lowBand, int id) {
    int i, j1, j2, k, b, i0;
    int avgRow, lastRow, blocks;
    int ncols, i_in_sub;
    const PetscInt    *cols;
    const PetscScalar *vals;

    blocks = innerSeg * outerSeg;
    avgRow = ceil(float(m) / blocks);
    
    for (i = Istart; i < Iend; i++) {
        // j2 corresponds to the j2-th submatrix in B[i]
        b = i / avgRow;
        i0 = i % avgRow;
        // b-th block save to the j1-th node
        j1 = b % outerSeg;
        // b-th block save to the j2-th part in each node
        j2 = b / outerSeg;
        // the saving position in the submatrix
        i_in_sub = j2 * avgRow + i0;
        
        MatGetRow(A,i,&ncols,&cols, &vals);
        MatSetValues(Ap[j1], 1, &i_in_sub, ncols, cols, vals,ADD_VALUES);
        for (k = 0; k < ncols; k++) {
            upBand[id] = MAX(upBand[id], cols[k] - i);
            lowBand[id] = MAX(lowBand[id], i - cols[k]);
        }
        MatRestoreRow(A,i0,&ncols,&cols,&vals);
    }
}

int *permute2nd(int m, const int innerSeg, const int outerSeg, int *mRow) {
    int i, i0, i_new, tmp;
    int b, j, j1, j2, k, *perm;
    int blocks, avgRow, lastRow;

    perm = new int[m];
    blocks = innerSeg * outerSeg;
    avgRow = ceil(float(m) / blocks);
    // Initialize the identify permutator
    for (i = 0; i < m; i++) {
        // b-th block
        b = i / avgRow;
        // A[i0]
        i0 = i % avgRow;
        // Ap[j1]
        j1 = b % outerSeg;
        // Ap[j1][j2]
        j2 = b / outerSeg;
        
        // the first Ap[0-j1] can't all have full elements
        i_new = 0;
        for (j = 0; j < j1; j++)
            i_new += mRow[j];
        i_new += j2 * avgRow + i0;
        perm[i] = i_new;
        //printf("i = %d | i0 = %d | j1 = %d | j2 = %d | newROw= %d\n", i,i0,j1,j2, perm[i]);
    }
    return perm;
}



void calc_ILUT(int m, int nz, PetscInt *coo_I, PetscInt *coo_J, PetscScalar *coo_v, 
    int *ibilut, int *jbilut, FLOAT_TYPE *bilut) {
    int i, info;
    int *csr_I = new int[m + 1];
    int *csr_J = new int[nz];
    /* Maximum fill-in, which is half of the preconditioner bandwidth */
    int maxfil = 20;
    int ierr;

    int ipar[128];
    FLOAT_TYPE dpar[128];
    ipar[0] = 6;
    ipar[5] = 1;
    ipar[6] = 1;
    ipar[30] = 1;
    //dpar[30] = 1.0e-16;
    dpar[30] = 1.0e-2;
    dpar[31] = 1.0e-10;

    FLOAT_TYPE *csr_v = new FLOAT_TYPE[nz];
    FLOAT_TYPE tol = 1e-3;

    MKL_INT job0[6] = {2,1,0,0,nz,3};
    
    mkl_dcsrcoo (job0, &m, csr_v, csr_J, csr_I, &nz, coo_v, coo_I, coo_J, &info);


    for (i = 0; i < m + 1; i++)
        printf("indexing %d %d %d %f\n",i,coo_I[i],csr_I[i], csr_v[i]);
    
    //one-based indexing of the array parameters.
    if (info != 0)
        printf("info = %d fails\n", info);
    printf("---------\n");
    dcsrilut (&m, csr_v, csr_I,csr_J, bilut, ibilut, jbilut, &tol, &maxfil, ipar, dpar, &ierr);
    printf("ierr = %d\n", ierr);
}
