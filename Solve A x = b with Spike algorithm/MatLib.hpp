

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))



void readArg(int argc, char *argv[], FILE **f);
void readVector(FILE *file, double *F, int dim, int id);
void vecSplit(double *f, int i0, int m, double *val);

void chooseRows(double *A, int row, int col, int p, int q, double *Top, double *Mid, double *Bot, int id);

// For Band Matrix
void bandWidth(FILE *file, int *dim, int *len, int *supDiag, int *subDiag);
void readBigBand(FILE *file, double *A, double *B, double *C, int id, int dim, int len, int supDiag, int subDiag, int core);

void fileWrite(FILE *file, double *val, int row);


// Determine the number of the subDiagonals and supDiagonals
void bandWidth(FILE *file, int *dim, int *len, int *supDiag, int *subDiag) {
    int i, m, n, tmp;
    double value;
    for (i = 0; i < 2; i++)
        fscanf(file, "%*[^\n]%*c");
    fscanf(file, "%d %d %d", dim, &tmp, len);
    if (*dim != tmp) {
        printf("This is not a squared matrix!");
        exit(1);
    }
    *subDiag = 0;
    *supDiag = 0;
    for (i = 0; i < *len; i++) {
        fscanf(file, "%d %d  %lg\n", &m, &n, &value);
        *supDiag = MAX(*supDiag, n - m);
        *subDiag = MAX(*subDiag, m - n);
    }
    fclose(file);
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

void readVector(FILE *file, double *F, int dim, int id) {
    int i, col;
    double *f;
    for (i = 0; i < 3 + id * dim; i++)
        fscanf(file, "%*[^\n]%*c");
    for (i = 0; i < dim; i++)
        fscanf(file, "%lg", &F[i]);
    if (file != stdin)
        fclose(file);
}


// For row * col Mat A, split into 3 parts from row p and q
void chooseRows(double *A, int row, int col, int p, int q, double *Top, double *Mid, double *Bot, int id) {
    int i, j;
    for (j = 0; j < col; j++) {
        for (i = 0; i < p; i++)
            Top[i + p * j] = A[i + row * j];
        for (i = p; i < q; i++)
            Mid[i - p + (q - p) * j] = A[i + row * j];      
        for (i = q; i < row; i++) {
            Bot[i - q + (row - q) * j] = A[i + row * j];
        }
    }
}

void vecSplit(double *f, int i0, int m, double *val) {
    int i;
    for (i = 0; i < m; i++)
        val[i] = f[i + i0];
}




void readBigBand(FILE *file, double *A, double *B, double *C, int id, int dim, int len, int supDiag, int subDiag, int core) {
    int i, j, i0, m, n, tmp, row;
    double value;
    // ignore the first 2 rows
    for (i = 0; i < 3; i++)
        fscanf(file, "%*[^\n]%*c");
    
    row = 2 * supDiag + subDiag + 1;
    
    for (i = 0; i < len; i++) {
        fscanf(file, "%d %d  %lg\n", &m, &n, &value);
        m -= id * dim;
        n -= id * dim;
        // Quit if it exceeds the limit for each node
        //if ((n > dim + supDiag) || (n < -subDiag))
        if (n > dim + supDiag)
            break;
        // Band Mat
        //if (id == 1)    printf("m= %d n= %d v = %f\n", m, n, value);
        if ((m - n <= subDiag) && (n - m <= supDiag)) {
            if ((m >= MAX(1, n - supDiag)) && (m <= MIN(dim, n + subDiag)) && (n > 0) && (n <= dim))
                A[m - n + supDiag + (n - 1) * row + subDiag] = value;
            else if ((id != core - 1) && (m >= dim - supDiag) && (m <= dim) && (n > dim)) {
                // The first i0 rows are 0
                i0 = dim - supDiag;
                B[m - dim + supDiag - 1 + dim * (n - dim - 1) + i0] = value;
            }
            else if ((id != 0) && (m > 0) && (m <= subDiag) && (n <= 0) && (n > -subDiag))
                // The last dim - subDiag rows are 0
                C[m - 1 + dim * (n + subDiag - 1)] = value;
        }
    }
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
