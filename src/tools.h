// ARCHITECTURE
#include "arch.h"

// Standard headers.
#include <stdio.h>
#include <stdlib.h>
#ifdef WIN
#define _USE_MATH_DEFINES
#endif
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <assert.h>
#include <omp.h>
#include <string.h>

// System headers.
#include <sys/types.h>
#include <sys/stat.h>
#ifdef WIN
#include <io.h>
#include <direct.h>
#else
#include <unistd.h>
#endif

// Intel MKL
#include "mkl.h"

// Libconfig for parameter parsing.
#include <libconfig.h>

// Indexing macro: requires that NzTotal be defined in scope.
#define IDX(i, j) ((i) * NzTotal + (j))

// Polar indexing macro.
#define P_IDX(i, j) ((i) * NthTotal + (j))

// MIN/MAX macros.
#define MIN(X, Y) ((X) < (Y)) ? (X) : (Y)
#define MAX(X, Y) ((X) > (Y)) ? (X) : (Y)

// ABS macro.
#define ABS(X) ((X) < 0) ? -(X) : (X)

// CSR matrix index base.
#define BASE 1

#define RESCALE 1.0

/* Macro for array sum z = alpha * x + beta * y: for alpha, beta scalars; z, x, y arrays. */
#define ARRAY_SUM(Z, ALPHA, X, BETA, Y) array_sum((Z), (ALPHA), (X), (BETA), (Y), dim)
void array_sum(double *z, const double alpha, double *x, const double beta, double *y, const MKL_INT dim);

/* Macro for coupled array sum for regularization. */
void coupled_du(double *du, double *u, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT ghost, const double dr, const double mu);

// Safe allocation macros.
#define SAFE_MALLOC(n) safe_malloc((n), __FILE__, __LINE__)
void *safe_malloc(const size_t n, const char *file, const MKL_INT line);

// Safe deallocation macros.
#define SAFE_FREE(x) safe_free((x), __FILE__, __LINE__)
void safe_free(void *x, const char *file, const MKL_INT line);

// CSR matrix type.
typedef struct csr_matrices
{
	double *a;
	MKL_INT *ia;
	MKL_INT *ja;
	MKL_INT nrows;
	MKL_INT ncols;
	MKL_INT nnz;
	MKL_INT analysis_phase;

} csr_matrix;

// Forward declarations.
// 
// Write simple ASCII 1D file.
void write_single_file_1d(double *u, const char *fname, const MKL_INT dim);
// Write simple integer ASCII 1D file.
void write_single_integer_file_1d(const MKL_INT *u, const char *fname, const MKL_INT dim);
// Write simple ASCII 2D file.
void write_single_file_2d(double *u, const char *fname, const MKL_INT NrTotal, const MKL_INT NzTotal);
// Write ASCII 2D file with iterations.
void write_iterated_file_2d(double **u, const char *fname, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT iterations, const MKL_INT gnum);
// Write simple ASCII 2D polar file.
void write_single_file_2d_polar(double *u, const char *fname, const MKL_INT NrrTotal, const MKL_INT NthTotal);
// Read simple ASCII 1D file.
void read_single_file_1d(double *u, const char *fname, const MKL_INT dim, const char *source_file, const MKL_INT source_line);
// Read simple ASCII 2D file.
void read_single_file_2d(double *u, const char *fname, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT NrTotalInitial, const MKL_INT NzTotalInitial, const char *source_file, const MKL_INT source_line);
// Create CSR matrix.
void csr_allocate(csr_matrix *A, const MKL_INT nrows, const MKL_INT ncols, const MKL_INT nnz);
// Destroy CSR matrix.
void csr_deallocate(csr_matrix *A);
// CSR matrix print.
void csr_print(csr_matrix *A, const char *vA, const char *iA, const char *jA);
