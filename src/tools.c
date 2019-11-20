// Headers.
#include "arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <omp.h>
#include <string.h>

// System headers.
#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN
#include <unistd.h>
#endif

// MKL header.
#include "mkl.h"

// Indexing macro: requires that NzTotal be defined in scope.
#define IDX(i, j) ((i) * NzTotal + (j))

// CSR matrix index base.
#define BASE 1

/* Macro for array sum z = alpha * x + beta * y: for alpha, beta scalars; z, x, y arrays. */
#define ARRAY_SUM(Z, ALPHA, X, BETA, Y) array_sum((Z), (ALPHA), (X), (BETA), (Y), dim)
void array_sum(double *z, const double alpha, const double *x, const double beta, const double *y, const MKL_INT dim)
{
	MKL_INT i;
	#pragma omp parallel shared(z) 
	{
		#pragma omp for schedule(guided)
		for (i = 0; i < dim; i++) 
			z[i] = alpha * x[i] + beta * y[i];
	} 
	return;
}

// Safe allocation macros.
#define SAFE_MALLOC(n) safe_malloc((n), __FILE__, __LINE__)
void *safe_malloc(const size_t n, const char *file, const MKL_INT line)
{
	void *x;
	
	x = malloc(n);

	if (!x)
	{
		fprintf(stderr, "CRITICAL ERROR! Failed memory allocation on file %s:%ld.\n", file, line);
		exit(1);
	}
	else
	{
		return x;
	}
}

// Safe deallocation macros.
#define SAFE_FREE(x) safe_free((x), __FILE__, __LINE__)
void safe_free(void *x, const char *file, const MKL_INT line)
{
	if (!x)
	{
		fprintf(stderr, "WARNING! Memory block attempting to free already points to NULL on file %s:%ld.\n", file, line);
	}
	else
	{
		free(x);
		x = NULL;
	}
	return;
}

// CSR matrix type.
typedef struct csr_matrices
{
	// Nonzeros array.
	double *a;
	// Row pointer.
	MKL_INT *ia;
	// Column pointer.
	MKL_INT *ja;
	// Number of rows.
	MKL_INT nrows;
	// Number of columns.
	MKL_INT ncols;
	// Number of nonzeros.
	MKL_INT nnz;
	// Flag to point if analysis phase has been done for this matrix.
	MKL_INT analysis_phase;

} csr_matrix;

// Write simple ASCII 1D file.
void write_single_file_1d(const double *u, const char *fname, const MKL_INT dim)
{
	// Auxiliary integers.
	MKL_INT i;

	// Open file.
	FILE *fp = fopen(fname, "w");

	// Loop over r and write values.
	for (i = 0; i < dim; i++)
	{
		fprintf(fp, "%9.18E\n", u[i]);
	}

	// Close file.
	fclose(fp);

	return;
}

// Write simple ASCII 2D file.
void write_single_file_2d(const double *u, const char *fname, const MKL_INT NrTotal, const MKL_INT NzTotal)
{
	// Auxiliary integers.
	MKL_INT i, j;

	// Open file.
	FILE *fp = fopen(fname, "w");

	// Loop over r and write values.
	for (i = 0; i < NrTotal; i++)
	{
		for (j = 0; j < NzTotal; j++)
		{
			fprintf(fp, (j < NzTotal - 1) ? "%9.18E\t" : "%9.18E\n", u[IDX(i, j)]);
		}
	}

	// Close file.
	fclose(fp);

	return;
}

// Read simple ASCII 1D file.
void read_single_file_1d(double *u, const char *fname, const MKL_INT dim, const char *source_file, const MKL_INT source_line)
{
	// Auxiliary integer.
	MKL_INT i, err;

	// Open file. 
	FILE *fp = fopen(fname, "r");

	// Loop over values. 
	for (i = 0; i < dim; i++)
	{
		err = fscanf(fp, "%lE", u + i);

		// Check that read was successfull.
		if (!err)
		{
			fprintf(stderr, "CRITICAL ERROR READING FILE \"%s\"! at %s.%ld.\n", fname, source_file, source_line);
			exit(1);
		}
	}

	// Close file.
	fclose(fp);

	// All done.
	return;
}

// Read simple ASCII 2D file.
void read_single_file_2d(double *u, const char *fname, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT NrTotalInitial, const MKL_INT NzTotalInitial, const char *source_file, const MKL_INT source_line)
{
	// Auxiliary integers.
	MKL_INT i, j, err;

	// Open file.
	FILE *fp = fopen(fname, "r");

	// Loop over r and read values.
	for (i = 0; i < NrTotalInitial; i++)
	{
		for (j = 0; j < NzTotalInitial; j++)
		{
			err = fscanf(fp, "%lE", u + IDX(i, j));

			// Check that read was successfull.
			if (!err)
			{
				fprintf(stderr, "CRITICAL ERROR READING FILE \"%s\"! at %s.%ld.\n", fname, source_file, source_line);
			}
		}
	}

	// Fill other values.
	for (i = 0; i < NrTotalInitial; i++)
	{
		for (j = NzTotalInitial; j < NzTotal; j++)
		{
			u[IDX(i, j)] = u[IDX(i, NzTotalInitial - 1)];
		}
	}

	for (j = 0; j < NzTotalInitial; j++)
	{
		for (i = NrTotalInitial; i < NrTotal; i++)
		{
			u[IDX(i, j)] = u[IDX(NrTotalInitial - 1, j)];
		}
	}

	for (i = NrTotalInitial; i < NrTotal; i++)
	{
		for (j = NzTotalInitial; j < NzTotal; j++)
		{
			u[IDX(i, j)] = 0.5 * (u[IDX(i, NzTotalInitial - 1)] + u[IDX(NrTotalInitial - 1, j)]);
		}
	}

	// Close file.
	fclose(fp);

	// All done.
	return;
}

// Create CSR matrix.
void csr_allocate(csr_matrix *A, const MKL_INT nrows, const MKL_INT ncols, const MKL_INT nnz)
{
	// Set integer parameters.
	A->nrows = nrows;
	A->ncols = ncols;
	A->nnz = nnz;
	// Allocate pointers.
	A->a  = (double *)SAFE_MALLOC(nnz * sizeof(double));
	A->ja = (MKL_INT 	*)SAFE_MALLOC(nnz * sizeof(MKL_INT));
	A->ia = (MKL_INT	*)SAFE_MALLOC((nrows + 1) * sizeof(MKL_INT));

	// No analysis phase.
	A->analysis_phase = 0;

	return;
}

// Destroy CSR matrix.
void csr_deallocate(csr_matrix *A)
{
	// Set integers to zero.
	A->nrows = 0;
	A->ncols = 0;
	A->nnz = 0;
	// Deallocate memory.
	SAFE_FREE(A->a);
	SAFE_FREE(A->ia);
	SAFE_FREE(A->ja);

	// No analysis phase.
	A->analysis_phase = 0;

	return;
}

// CSR matrix print.
void csr_print(csr_matrix *A, const char *vA, const char *iA, const char *jA)
{
	printf("Printing CSR matrix...\n");
	// Open three files corresponding to each array.
	FILE *fvA = fopen(vA, "w");
	FILE *fiA = fopen(iA, "w");
	FILE *fjA = fopen(jA, "w");

	// Auxiliary integers.
	MKL_INT k = 0;

	// Loop over number of nonzeros and write A.a and A.ja.
	for (k = 0; k < A->nnz; k++)
	{
		fprintf(fvA, "%9.18E\n", A->a[k]);
		fprintf(fjA, "%ld\n", A->ja[k]);
	}

	// Loop over A.nrows + 1 and write A.ia.
	for (k = 0; k < A->nrows + 1; k++)
		fprintf(fiA, "%ld\n", A->ia[k]);

	// Close files.
	fclose(fvA);
	fclose(fiA);
	fclose(fjA);

	printf("Done CSR printing.\n");

	return;
}
