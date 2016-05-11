#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <math.h>

#include "mmio.h"
#include "matcomput.h"

const int WID = 128;

int main(int argc, char *argv[])
{
	int ret_code, chunk, nthreads, tid;
	int i, j, row[1], col[1], len[1];
	float **A, *b, *x;
    double *MM;
	FILE *f1, *f2, *f3;
	MM_typecode matcode;
	chunk = 16;

    /* Input File Judgement */
	if (argc < 4) {
		fprintf(stderr, "Usage: %s missing input data!\n", argv[0]);
		exit(1);
	}
	else {
		if ((f1 = fopen(argv[1], "r")) == NULL)
			exit(1);
        if ((f2 = fopen(argv[2], "r")) == NULL)
            exit(1);
        if ((f3 = fopen(argv[3], "w")) == NULL)
            exit(1);
	}
	/* Read Input File */
    printf("Reading...\n");
	A = readBandedMatrix(f1, &matcode, row, col, len, WID);
    b = readVector(f2, &matcode, row);
    printf("Runing...\n");
    // Algorithm 3.3: Block Banded Triangular solver
    double  tic = omp_get_wtime();
    x = BBTS(A, b, *row, WID, chunk, nthreads, tid);
    double  toc = omp_get_wtime();
    printf("Time = %.3f seconds\n", toc - tic);
    fileWrite(f3, row, x, &matcode);
	return 0;
}
