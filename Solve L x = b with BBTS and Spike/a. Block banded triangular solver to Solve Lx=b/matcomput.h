
float **readBandedMatrix(FILE *f, MM_typecode *matcode, int *row, int *col, int *nz, int wid);
float **readMatrix(FILE *f, MM_typecode *matcode, int *row, int *col);
float *readVector(FILE *f, MM_typecode *matcode, int *row);
void fileWrite(FILE *f, int *row, float *C, MM_typecode *matcode);
void threadReport(int *row, int tid, int i);
float *DTS_mat_vec_mul(float **m, float *f, int *row, int col_start, int col_end);
float **DTS_mat_mat_mul(float **m1, float **m2, int *row, int col_1, int col_2);
float **DTS();

// Basic Matrix Calculation
float **matTrans(float **A, int dim);
float **mat_mat_mul(float **A, float **B, int dim);
float **matSplit(float **L, int i0, int j0, int m, int n, char tag);
float *vecSplit(float *f, int i0, int m);
void matMerge(float ***L, int v, int wid, float **M);
void vecMerge(float **f, int v, int wid, float *val);



// Matrix Algorithm
float *cSweep(float **L, float *f, int *row, int chunk, bool tag);
void vecCombine(float **L, float *f, int dim, int x, int wid, int r, int chunk);
void matCombine(float **L, float **C, int dim, int x, int wid, int r, int chunk);
float *BBTS(float **L, float *f, int dim, int wid, int chunk, int core, int tid);
