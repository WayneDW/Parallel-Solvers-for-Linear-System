#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <math.h>

// Function
void print2D(double **L, int m, int n);

// Initialization
void readArg(int argc, char *argv[], FILE **f);
void fileWrite(FILE *file, double *val, int row);
void threadReport(int *row, int tid, int i);

double *readVector(FILE *f, int *row);
double **matInit(int m, int n);
double *readMat(FILE *file, int *dim);
double **readBandedMat(FILE *file, int wid, int *col);

// Basic Matrix Calculation
double *Trans(double *A, int m, int n);
double **matTrans(double **A, int m, int n);
double **matSplit(double **L, int i0, int j0, int m, int n, char tag);
double *vecSplit(double *f, int i0, int m);
void getLU(double *vec, int dim, double *L, double *U);
void matMerge(double ***L, int v, int wid, double **M);
void vecMerge(double **f, int v, int wid, double *val);
double *Seg(double *vec, int dim, int n, int i0, char type);
void data_seg(double **L, double *f, double **L_hat[4], double **R_hat[4], 
    double *f_hat[4], int col, int cores, int WID);
void matLU(double *A, int m, int tag, int chunk, int id, int dim, int *step);

// Matrix Algorithm
void vecCombine(double **L, double *f, int dim, int x, int wid, int r, int chunk, int nthreads);
void vecCombine_Mat(double **L, double **f, int dim, int x, int wid, int r, int chunk, int nthreads);
void matCombine(double **L, double **C, int dim, int x, int wid, int r, int chunk, int nthreads);
double *cSweep(double **L, double *f, int row, int chunk, bool tag);
double **cSweep_Mat(double **L, double **f, int wid, int row, int chunk);
double *BBTS(double **L, double *f, int dim, int wid, int chunk, int nthreads, int tid);
double **BBTS_Mat(double **L, double **fvec, int dim, int wid, int chunk, int nthreads, int taskid);


void print2D(double **L, int m, int n) {
    int i, j;
    printf("\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++)
            printf("%.2f ", L[i][j]);
        printf("\n");
    }
}

double **matInit(int m, int n) {
    int i;
    double **L;
    L = new double *[m];
    for (i = 0; i < m; i++)
        L[i] = new double[n]();
    return L;
}

void readArg(int argc, char *argv[], FILE **f) {
	int i;
	if (argc < 4) {
		fprintf(stderr, "Usage: %s missing input data!\n", argv[0]);
		exit(1);
	}
	else {
		for (i = 0; i < argc - 2; i++)
			f[i] = fopen(argv[i + 1], "r");
		f[argc - 2] = fopen(argv[argc - 1], "w");
	}
}

double *readMat(FILE *file, int *dim) {
    int i, m, n, len, row;
    double value, *L;
    for (i = 0; i < 2; i++)
        fscanf(file, "%*[^\n]%*c");
    fscanf(file, "%d %d", &row, dim);
    L = new double[row * *dim]();
    for (i = 0; i < row * *dim; i++)
        fscanf(file, "%lg\n", &L[i]);
    return L;
}

double **readBandedMat(FILE *file, int wid, int col[1]) {
	int i, m, n, row, len;
	double value, **L;
	// ignore the first 2 rows
	for (i = 0; i < 2; i++)
		fscanf(file, "%*[^\n]%*c");
	// get the indexes from this file
	fscanf(file, "%d %d %d", &row, col, &len);
	L = new double *[wid];
	for (i = 0; i < wid; i++)
		L[i] = new double[col[0]];
	for (i = 0; i < len; i++) {
		fscanf(file, "%d %d  %f\n", &m, &n, &value);
		if (m == n)
			continue;
		L[m - n - 1][n - 1] = value;
		//L[m - 1][wid - m + n] = value;
	}
	if (file != stdin)
		fclose(file);
	return L;
}

double *readVector(FILE *file, int *row) {
	int i, col;
	double *f;
	for (i = 0; i < 2; i++)
		fscanf(file, "%*[^\n]%*c");
	fscanf(file, "%d %d", row, &col);
	f = new double[*row];
	for (i = 0; i < *row; i++) {
		fscanf(file, "%lg", &f[i]);
	}
	if (file != stdin)
		fclose(file);
	return f;
}

void fileWrite(FILE *file, double *val, int row) {
	int i;
	char const *s1 = "%%MatrixMarket matrix array real general";
	char const *s2 = "%vector b";
	fprintf(file, "%s\n", s1);
	fprintf(file, "%s\n", s2);
	fprintf(file, "%d %d\n", row, 1);
	for (i = 0; i < row; i++)
		fprintf(file, "%.20lg\n", val[i]);
	fclose(file);
}

void threadReport(int *row, int tid, int i) {
	int limit = *row / 100 + 1;
	if (i%limit == 0) {
		if (tid < 10)
			printf(" Core 0%d: %d %% complete!\n", tid, i / limit);
		else
			printf(" Core %d: %d %% complete!\n", tid, i / limit);
	}
}


void matLU(double *A, int m, int tag, int chunk, int id, int dim, int *step) {
    int i, j;
    double beta, norm, tmp;
    norm = 0;
    // beta is the squre root of alpha
    beta = 1e-3;
    //#pragma omp parallel for reduction(+:norm)
    for (i = tag; i < m; i++)
        norm += fabs(A[tag + i * m]);
    if (fabs(A[tag + tag * m]) <= beta * beta * norm) {
        if (A[tag + tag * m] > 0)
            A[tag + tag * m] += beta * norm;
        else
            A[tag + tag * m] -= beta * norm;

        printf("Node %d Boosting row-col [%d] in step %d\n", id, tag + id * dim + 1, *step);
        *step += 1;
    }
    #pragma omp parallel shared(A, m, tag, norm, tmp) private(i, j) 
    {
        #pragma omp for schedule (static, chunk)
        for (i = tag + 1; i < m; i++) {
            for (j = tag; j < m; j++) {
                if (j == tag)
                    A[j + i * m] /= A[tag + tag * m];
                else
                    A[j + i * m] -= A[tag + i * m] * A[j + tag * m];
            }
        }
    }
    if (tag < m - 2)
        matLU(A, m, tag + 1, chunk, id, dim, step);
}

void data_seg(double **L, double *f, double **L_hat[4], double **R_hat[4], double *f_hat[4], int col, int cores, int WID) {
	int i, j, k;
	int subcol = col / cores;
	for (i = 0; i < cores; i++) {
		L_hat[i] = new double*[WID]();
		for (j = 0; j < WID; j++)
			L_hat[i][j] = new double[subcol]();
		if (i != 0) {
            R_hat[i] = new double*[subcol]();
			//R_hat[i] = new double*[WID]();
			for (j = 0; j < subcol; j++)
				R_hat[i][j] = new double[WID]();
		}
	}
	for (i = 0; i < cores; i++) {
		for (j = 0; j < WID; j++) {
			for (k = 0; k < subcol; k++) {
				if (j + k < subcol - 1) {
					L_hat[i][j][k] = L[j][k + i * subcol];
				}
				else if (i != cores - 1) {
					R_hat[i + 1][j + k - subcol + 1][k - subcol + WID] = L[j][k + i * subcol];
				}
			}
		}
	}
	for (i = 0; i < cores; i++)
		f_hat[i] = vecSplit(f, i * subcol, subcol);
}

double **matTrans(double **A, int m, int n) {
	int i, j;
	double **val;
	val = new double *[n];
	for (i = 0; i < n; i++) {
		val[i] = new double[m];
		for (j = 0; j < m; j++)
			val[i][j] = A[j][i];
	}
	return val;
}

// type: true for vec, false for square matrix split
double *Seg(double *vec, int dim, int n, int i0, char type) {
	int i, j, m, tmp;
	double *val;
	if (type == true) {
		val = new double[n];
		for (i = 0; i < n; i++)
			val[i] = vec[i0 + i];
	}
	else {
		val = new double[n * n];
		for (i = 0; i < n; i++) {
			tmp = i0 + i * dim;
			for (j = 0; j < n; j++)
				val[j + n * i] = vec[tmp + j];
		}
	}
	return val;
}

void getLU(double *vec, int dim, double *L, double *U) {
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

double *Trans(double *A, int m, int n) {
    int i, j, k;
    double *val;
    val = new double [n * m];
    for (i = 0; i < m * n; i++) {
        j = i / m;
        k = i % m;
        val[k * m + j] = A[i];
    }
    return val;
}

// Given the size and location to generate the submatrix.
double **matSplit(double **L, int i0, int j0, int m, int n, char tag) {
	int i, j;
	double **val;
	val = new double *[m]();
	for (i = 0; i < m; i++) {
		val[i] = new double[n]();
		if (tag == true) {
			for (j = 0; j < n; j++)
				if (i == j)
					val[i][j] = 1;
				else if (i > j)
					val[i][j] = L[i - j - 1][j + j0];
		}
		else {
			for (j = 0; j < n; j++)
				if (i <= j)
					val[i][j] = L[m - 1 + i - j][j + j0];
		}
	}
	return val;
}

// Given the size and location to generate the subvector.
double *vecSplit(double *f, int i0, int m) {
	int i;
	double *val;
	val = new double[m]();
	for (i = 0; i < m; i++)
		val[i] = f[i + i0];
	return val;
}

void matMerge(double ***L, int v, int wid, double **M) {
	int i, j, k;
	int num = exp2(v) - 1;
	for (k = 0; k < num; k++)
		for (i = 0; i < wid; i++)
			for (j = 0; j < wid; j++)
				M[i + k * wid][j] = L[k][i][j];
}

void vecMerge(double **f, int v, int wid, double *val) {
	int i, k;
	int num = exp2(v);
	for (k = 0; k < num; k++)
		for (i = 0; i < wid; i++)
			val[i + k * wid] = f[k][i];
}

void vecCombine(double **L, double *f, int dim, int x, int wid, int r, int chunk, int nthreads) {
	int i, j;
	double *v;
	int m0 = 2 * r * x * wid;
	v = new double[dim]();
	#pragma omp parallel shared(L, f, v, m0, x, wid, nthreads) private(i, j)
	{
		#pragma omp for schedule (static, chunk)
		for (i = 0; i < wid * x; i++)
			for (j = 0; j < wid; j++)
				v[i + m0 + wid * x] += L[i + m0][j] * f[j - wid + m0 + wid * x];
		#pragma omp for schedule (static, chunk)
		for (i = 0; i < 2 * x * wid; i++)
			f[i + m0] -= v[i + m0];
	}
	free(v);
}

void vecCombine_Mat(double **L, double **f, int dim, int x, int wid, int r, int chunk, int nthreads) {
    int i, j, k;
    double **v;
    int m0 = 2 * r * x * wid;
    v = new double*[wid]();
    for (i = 0; i < wid; i++)
        v[i] = new double[dim]();
    #pragma omp parallel shared(L, f, v, m0, x, wid, nthreads) private(i, j, k)
    {
        #pragma omp for schedule (static, chunk)
        for (k = 0; k < wid; k++)
            for (i = 0; i < wid * x; i++)
                for (j = 0; j < wid; j++)
                    v[k][i + m0 + wid * x] += L[i + m0][j] * f[k][j - wid + m0 + wid * x];
        #pragma omp for schedule (static, chunk)
        for (k = 0; k < wid; k++)
            for (i = 0; i < 2 * x * wid; i++)
                f[k][i + m0] -= v[k][i + m0];
    }
    for (i = 0; i < wid; i++)
        delete[] v[i];
    delete[] v;
}



void matCombine(double **L, double **C, int dim, int x, int wid, int r, int chunk, int nthreads) {
	int i, j, k, tmp;
	int m0 = 2 * (r - 1) * x * wid;
	#pragma omp parallel shared(L, C, m0, r, x, wid, nthreads) private(i, j, k)
	{
		#pragma omp for schedule (static, chunk)
		for (i = 0; i < wid * x; i++)
			for (j = 0; j < wid; j++)
				C[i + m0][j] = L[i + (2 * r - 1) * wid * x][j];
		#pragma omp for schedule (static, chunk)
		for (i = 0; i < wid * x; i++)
			for (j = 0; j < wid; j++) {
				tmp = 0;
				for (k = 0; k < wid; k++)
					tmp -= L[i + (2 * r) * wid * x][k] * L[k + m0 + 2 * wid * x - wid][j];
				C[i + m0 + wid * x][j] = tmp;
			}
	}
}

// CSweep: Column-sweep method for unit lower triangular system
// tag means whether implement through parallel
double *cSweep(double **L, double *f, int row, int chunk, bool tag) {
	int i, j, n, tid, nthreads;
	double *val;
	val = new double[row];
	for (i = 0; i < row; i++) {
		val[i] = f[i];
		if (tag) {
    		#pragma omp parallel shared(L,f,i,nthreads) private(j,tid) 
	    	{
		    	#pragma omp for schedule (static, chunk)
			    for (j = i + 1; j < row; j++)
				    f[j] -= f[i] * L[j][i];
		    }
		} else {
			for (j = i + 1; j < row; j++)
				f[j] -= f[i] * L[j][i];
		}
	}
	return val;
}


double **cSweep_Mat(double **L, double **f, int wid, int row, int chunk) {
    int i, j, k, n, tid, nthreads;
    double **val;
    val = new double*[wid];
    for (i = 0; i < wid; i++)
        val[i] = new double[row]();
    #pragma omp parallel shared(L, f, i, val, nthreads) private(j, k, tid)
    {
        #pragma omp for schedule (static, chunk)
        for (k = 0; k < wid; k++) {
            for (i = 0; i < row; i++) {
                val[k][i] = f[k][i];
                for (j = i + 1; j < row; j++)
                    f[k][j] -= f[k][i] * L[j][i];
            }
        }
    }
    return val;
}



double *BBTS(double **L, double *f, int dim, int wid, int chunk, int nthreads) {
	int i, j, k, v, x, num, tid;
	double **fpar, *val, **G[2];
	// Dim0|1: ori|inverse, dim2: order; Dim3/4: Col/Row
	double ***Gs[2], ***Rs[2], ***Ls[2];
	num = dim / wid;
	v = log2(num);
	fpar = new double*[num]();
	val = new double[dim]();
	double  tic = omp_get_wtime();
	for (i = 0; i < 2; i++) {
		Ls[i] = new double**[num]();
		Gs[i] = new double**[num]();
		Rs[i] = new double**[num]();
		G[i] = new double*[dim - wid]();
		for (j = 0; j < dim - wid; j++)
			G[i][j] = new double[wid]();
	}
	for (i = 0; i < num; i++) {
		fpar[i] = vecSplit(f, i * wid, wid);
		Ls[0][i] = matSplit(L, 0, i * wid, wid, wid, true);
		if (i != num - 1) {
			Rs[0][i] = matSplit(L, 0, i * wid, wid, wid, false);
			Rs[1][i] = matTrans(Rs[0][i], wid, wid);
			Gs[1][i] = new double*[wid]();
		}
	}
	// Step 1
	fpar[0] = cSweep(Ls[0][0], fpar[0], wid, chunk, true);
	// Step 2-4
	#pragma omp parallel shared(fpar, Ls, Rs, Gs, nthreads, chunk) private(tid, i, j, k)
	{
		tid = omp_get_thread_num();
		#pragma omp for schedule (static, chunk)
		for (i = 1; i < num; i++) {
			fpar[i] = cSweep(Ls[0][i], fpar[i], wid, chunk, false);
			for (j = 0; j < wid; j++)
				Gs[1][i - 1][j] = cSweep(Ls[0][i], Rs[1][i - 1][j], wid, chunk, false);
			Gs[0][i - 1] = matTrans(Gs[1][i - 1], wid, wid);
		}
	}
	// Step 5-10
	matMerge(Gs[0], v, wid, G[0]);
	vecMerge(fpar, v, wid, val);
	for (k = 1; k < v; k++) {
		// Step 6 update f with the last half part
		x = exp2(k - 1);
		vecCombine(G[(k - 1) % 2], val, dim, x, wid, 0, chunk, nthreads);
		for (j = 1; j < exp2(v - k); j++) {
			vecCombine(G[(k - 1) % 2], val, dim, x, wid, j, chunk, nthreads);
			matCombine(G[(k - 1) % 2], G[k % 2], dim, x, wid, j, chunk, nthreads);
		}
	}
	vecCombine(G[(v - 1) % 2], val, dim, exp2(v - 1), wid, 0, chunk, nthreads);
	return val;
}


double **BBTS_Mat(double **L, double **fvec, int dim, int wid, int chunk, int nthreads, int taskid) {
	int i, j, k, v, x, num, tid;
	double **fpar[wid], **val, **G[2];
	// Dim0|1: ori|inverse, dim2: order; Dim3/4: Col/Row
	double ***Gs[2], ***Rs[2], ***Ls[2];
	num = dim / wid;
	v = log2(num);
    val = new double*[wid];
	for (i = 0; i < wid; i++) {
		val[i] = new double[dim]();
		fpar[i] = new double*[num]();
	}

	for (i = 0; i < 2; i++) {
		Ls[i] = new double**[num]();
		Gs[i] = new double**[num]();
		Rs[i] = new double**[num]();
		G[i] = new double*[dim - wid]();
		for (j = 0; j < dim - wid; j++)
			G[i][j] = new double[wid]();
	}

    for (i = 0; i < wid; i++)
        for (j = 0; j < num; j++)
            fpar[i][j] = vecSplit(fvec[i], j * wid, wid);
	for (i = 0; i < num; i++) {
		Ls[0][i] = matSplit(L, 0, i * wid, wid, wid, true);
		if (i != num - 1) {
			Rs[0][i] = matSplit(L, 0, i * wid, wid, wid, false);
			Rs[1][i] = matTrans(Rs[0][i], wid, wid);
			Gs[1][i] = new double*[wid]();
		}
	}
    // Step 1 - 4
    for (i = 1; i < num; i++)
        Gs[1][i - 1] = cSweep_Mat(Ls[0][i], Rs[1][i - 1], wid, wid, chunk);
	#pragma omp parallel shared(fpar, Ls, Rs, Gs, nthreads, chunk) private(tid, i, j, k)
	{
		tid = omp_get_thread_num();
        #pragma omp for schedule (static, chunk)
        for (i = 1; i < num; i++)
            Gs[0][i - 1] = matTrans(Gs[1][i - 1], wid, wid);
		#pragma omp for schedule (static, chunk)
		for (i = 0; i < wid; i++)
			for (j = 0; j < num; j++)
				fpar[i][j] = cSweep(Ls[0][j], fpar[i][j], wid, chunk, false);
	}
	// Step 5-10
	matMerge(Gs[0], v, wid, G[0]);
	for (i = 0; i < wid; i++)
		vecMerge(fpar[i], v, wid, val[i]);
	for (k = 1; k < v; k++) {
		// Step 6 update f with the last half part
		x = exp2(k - 1);
        vecCombine_Mat(G[(k - 1) % 2], val, dim, x, wid, 0, chunk, nthreads);
		for (j = 1; j < exp2(v - k); j++) {
            vecCombine_Mat(G[(k - 1) % 2], val, dim, x, wid, j, chunk, nthreads);
			matCombine(G[(k - 1) % 2], G[k % 2], dim, x, wid, j, chunk, nthreads);
		}
	}
    vecCombine_Mat(G[(v - 1) % 2], val, dim, exp2(v - 1), wid, 0, chunk, nthreads);
	return val;
}

