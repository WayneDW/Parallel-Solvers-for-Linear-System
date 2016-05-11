#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <math.h>

#include "mmio.h"
#include "matcomput.h"


float **readBandedMatrix(FILE *f, MM_typecode *matcode, int *row, int *col, int *nz, int wid) {
	int i, j, ret_code, m, n;
	float **val, value;
	if (mm_read_banner(f, matcode) != 0) {
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}
	if ((ret_code = mm_read_mtx_crd_size(f, row, col, nz)) != 0)
		exit(1);
	val = (float **)malloc(wid * sizeof(double));
    for (i = 0; i < wid; i++)
		val[i] = (float *)malloc(*col * sizeof(double));
    for (i = 0; i < *nz; i++) {
		    fscanf(f, "%d %d  %f\n", &m, &n, &value);
            if(m == n)
                continue;
            val[m - n - 1][n - 1] = value;
        }
	if (f != stdin)
		fclose(f);
	return val;
}


float **readMatrix(FILE *f, MM_typecode *matcode, int *row, int *col) {
	int i, j, ret_code;
	float **val;
	if (mm_read_banner(f, matcode) != 0) {
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}
	if ((ret_code = mm_read_mtx_array_size(f, row, col)) != 0)
		exit(1);
	val = (float **)malloc(*row * sizeof(double));
	for (i = 0; i < *row; i++)
		val[i] = (float *)malloc(*col * sizeof(double));
	for (j = 0; j < *col; j++)
		for (i = 0; i < *row; i++)
			fscanf(f, "%f", &val[i][j]);
	if (f != stdin)
		fclose(f);
	return val;
}

float *readVector(FILE *f, MM_typecode *matcode, int *row) {
	int i, col, ret_code;
	float *val;
	if (mm_read_banner(f, matcode) != 0) {
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}
	if ((ret_code = mm_read_mtx_array_size(f, row, &col)) != 0)
		exit(1);
	val = (float *)malloc(*row * sizeof(double));

	for (i = 0; i < *row; i++)
		fscanf(f, "%f", &val[i]);
	if (f != stdin)
		fclose(f);
	return val;
}

void fileWrite(FILE *f, int *row, float *C, MM_typecode *matcode) {
	int i, j;
	mm_write_banner(f, *matcode);
    fprintf(f, "%vector x\n");
	mm_write_mtx_array_size(f, *row, 1);
	for (i = 0; i < *row; i++)
		fprintf(f, "%f\n", C[i]);
	fclose(f);
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

float **matTrans(float **A, int dim) {
	int i, j;
	float **val;
	val = (float **)malloc(dim * sizeof(double));
	for (i = 0; i < dim; i++) {
		val[i] = (float *)malloc(dim * sizeof(double));
		for (j = 0; j < dim; j++)
			val[i][j] = A[j][i];
	}
	return val;
}

// CSweep: Column-sweep method for unit lower triangular system
// tag means whether implement through parallel
float *cSweep(float **L, float *f, int *row, int chunk, bool tag) {
	int i, j, n, tid, nthreads;
	float *val;
	val = (float *)malloc(*row * sizeof(double));
	for (i = 0; i < *row; i++) {
		val[i] = f[i];
		if (tag) {
            #pragma omp parallel shared(L,f,i,nthreads) private(j,tid) 
            {
                #pragma omp for schedule (static, chunk)
	    		for (j = i + 1; j < *row; j++)
		    		f[j] -= f[i] * L[j][i];
            }
		}
		else {
			for (j = i + 1; j < *row; j++)
				f[j] -= f[i] * L[j][i];
		}
	}
	return val;
}

// Given the size and location to generate the submatrix.
float **matSplit(float **L, int i0, int j0, int m, int n, char tag) {
	int i, j;
	float **val;
	val = (float **)calloc(m, sizeof(double));
	for (i = 0; i < m; i++) {
		val[i] = (float *)calloc(n, sizeof(double));
        if(tag == true) {
            for (j = 0; j < n; j++)
                if(i == j)
                    val[i][j] = 1;
                else if(i > j)
                    val[i][j] = L[i - j - 1][j + j0];
        }
        else {
            for (j = 0; j < n; j++)
                if(i <= j)
                    val[i][j] = L[m - 1 + i - j][j + j0];
        }
	}
	return val;
}

// Given the size and location to generate the subvector.
float *vecSplit(float *f, int i0, int m) {
	int i;
	float *val;
	val = (float *)calloc(m, sizeof(double));
	for (i = 0; i < m; i++)
		val[i] = f[i + i0];
	return val;
}

void matMerge(float ***L, int v, int wid, float **M) {
	int i, j, k;
	int num = exp2(v) - 1;
	for (k = 0; k < num; k++)
		for (i = 0; i < wid; i++)
			for (j = 0; j < wid; j++)
				M[i + k * wid][j] = L[k][i][j];
}

void vecMerge(float **f, int v, int wid, float *val) {
	int i, k;
	int num = exp2(v);
	for (k = 0; k < num; k++)
		for (i = 0; i < wid; i++)
			val[i + k * wid] = f[k][i];
}

void vecCombine(float **L, float *f, int dim, int x, int wid, int r, int chunk) {
	int i, j;
    float *v;
    int m0 = 2 * r * x * wid;
    v = (float *)calloc(dim, sizeof(double));
    #pragma omp parallel shared(L, f, v, m0, x, wid) private(i, j)
    {
        #pragma omp for schedule (static, chunk)
        for (i = 0; i < wid * x; i++)
            for (j = 0; j < wid; j++)
                    v[i + m0 + wid * x] += L[i + (2 * r) * wid * x][j] * f[j + (2 * r) * wid * x];
        #pragma omp for schedule (static, chunk)
	    for (i = 0; i < 2 * x * wid; i++)
		    f[i + m0] -=  v[i + m0];
    }
    free(v);
}

void matCombine(float **L, float **C, int dim, int x, int wid, int r, int chunk) {
    int i, j, k;
    int m0 = 2 * (r - 1) * x * wid;
    #pragma omp parallel shared(L, C, m0, r, x, wid) private(i, j, k)
    {
        #pragma omp for schedule (static, chunk)
        for (i = 0; i < wid * x; i++)
            for (j = 0; j < wid; j++)
                C[i + m0][j] = L[i + (2 * r - 1) * wid * x][j];
        #pragma omp for schedule (static, chunk)
        for (i = 0; i < wid * x; i++)
            for (k = 0; k < wid; k++)
                for (j = 0; j < wid; j++)
                    C[i + m0 + wid * x][j] -= L[i + (2 * r) * wid * x][k] * L[k + (2 * r - 1) * wid * x][j];
    }
}

float *BBTS(float **L, float *f, int dim, int wid, int chunk, int core, int tid) {
	int i, j, k, v, x, num;
	float **fpar, *val, **G[2];
	// Dim0|1: ori|inverse, dim2: order; Dim3/4: Col/Row
	float ***Gs[2], ***Rs[2], ***Ls[2];
    num = dim / wid;
    v = log2(num);
	fpar = (float **)calloc(num, sizeof(double));
    val = (float *)calloc(dim, sizeof(double));
	for (i = 0; i < 2; i++) {
		Ls[i] = (float ***)calloc(num, sizeof(double));
		Gs[i] = (float ***)calloc(num, sizeof(double));
		Rs[i] = (float ***)calloc(num, sizeof(double));
        G[i] = (float **)calloc(dim - wid, sizeof(double));
        for (j = 0; j < dim - wid; j++)
            G[i][j] = (float *)calloc(wid, sizeof(double));
	}
	for (i = 0; i < num; i++) {
		fpar[i] = vecSplit(f, i * wid, wid);
		Ls[0][i] = matSplit(L, 0, i * wid, wid, wid, true);
		if (i != num - 1) {
			Rs[0][i] = matSplit(L, 0, i * wid, wid, wid, false);
			Rs[1][i] = matTrans(Rs[0][i], wid);
			for (j = 0; j < wid; j++) {
				Gs[1][i] = (float **)calloc(wid, sizeof(double));
			}
		}
	}
	// Step 1
	fpar[0] = cSweep(Ls[0][0], fpar[0], &wid, chunk, true);
	// Step 2-4
	#pragma omp parallel shared(fpar, Ls, Rs, Gs, core, chunk) private(tid, i, j, k)
	{
		#pragma omp for schedule (static, chunk)
		for (i = 1; i < num; i++) {
			fpar[i] = cSweep(Ls[0][i], fpar[i], &wid, chunk, false);
			for (j = 0; j < wid; j++)
				Gs[1][i - 1][j] = cSweep(Ls[0][i], Rs[1][i - 1][j], &wid, chunk, false);
			Gs[0][i - 1] = matTrans(Gs[1][i - 1], wid);
		}
	}
	// Step 5-10
	matMerge(Gs[0], v, wid, G[0]);
	vecMerge(fpar, v, wid, val);
	for (k = 1; k < v; k++) {
		// Step 6 update f with the last half part
		x = exp2(k - 1);
		vecCombine(G[(k - 1) % 2], val, dim, x, wid, 0, chunk);
		for (j = 1; j < exp2(v - k); j++) {
			vecCombine(G[(k - 1) % 2], val, dim, x, wid, j, chunk);
            matCombine(G[(k - 1) % 2], G[k % 2], dim, x, wid, j,chunk);
		}
	}
    vecCombine(G[(v - 1) % 2], val, dim, exp2(v - 1), wid, 0, chunk);
    return val;
}
