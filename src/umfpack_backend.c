// UMFPACK (SuiteSparse) backend for the sparse direct solver interface.
//
// Mirrors the PARDISO backend's semantics:
//   - solver_start(n)   : record dimension, set UMFPACK defaults
//   - solver_solve      : symbolic (once) + numeric + solve
//   - solver_repeated_solve : solve only, reusing the current factorization
//   - solver_solve_low_rank  : falls back to solver_solve (no low-rank update)
//   - solver_diff_gen        : no-op
//
// The Jacobian is stored in csr_matrix as 1-based CSR (ia = row pointers,
// ja = column indices); UMFPACK wants CSC, so it is converted once per
// sparsity pattern and its values refreshed on each factorization.
#include "tools.h"
#include "solver.h"

#include <suitesparse/umfpack.h>

static MKL_INT sys_n = 0;

static void *Sym = NULL; // symbolic factorization
static void *Num = NULL; // numeric factorization
static int64_t *CSC_Ap = NULL; // CSC column pointers (n+1)
static int64_t *CSC_Ai = NULL; // CSC row indices (nnz)
static double *CSC_Ax = NULL; // CSC values (nnz)
static double Control[UMFPACK_CONTROL];
static double Info[UMFPACK_INFO];

// Convert a 1-based CSR matrix to 0-based CSC, allocating the CSC arrays.
static void csr_to_csc(const csr_matrix *A)
{
    int64_t n = A->nrows;
    int64_t nnz = A->nnz;
    int64_t *counts = (int64_t *)SAFE_MALLOC(sizeof(int64_t) * n);
    int64_t *next = (int64_t *)SAFE_MALLOC(sizeof(int64_t) * n);
    int64_t r, k;

    CSC_Ap = (int64_t *)SAFE_MALLOC(sizeof(int64_t) * (n + 1));
    CSC_Ai = (int64_t *)SAFE_MALLOC(sizeof(int64_t) * nnz);
    CSC_Ax = (double *)SAFE_MALLOC(sizeof(double) * nnz);

    for (r = 0; r < n; ++r)
        counts[r] = 0;

    // Count entries per column.
    for (k = 0; k < nnz; ++k)
        counts[A->ja[k] - 1]++;

    CSC_Ap[0] = 0;
    for (r = 0; r < n; ++r)
        CSC_Ap[r + 1] = CSC_Ap[r] + counts[r];

    for (r = 0; r < n; ++r)
        next[r] = CSC_Ap[r];

    // Scatter entries into columns.
    for (r = 0; r < n; ++r)
    {
        for (k = A->ia[r] - 1; k < A->ia[r + 1] - 1; ++k)
        {
            int64_t col = A->ja[k] - 1;
            int64_t pos = next[col]++;
            CSC_Ai[pos] = r;
            CSC_Ax[pos] = A->a[k];
        }
    }

    SAFE_FREE(counts);
    SAFE_FREE(next);
}

// Refresh CSC values from the (possibly updated) CSR values, same pattern.
static void refresh_csc_values(const csr_matrix *A)
{
    int64_t n = A->nrows;
    int64_t *next = (int64_t *)SAFE_MALLOC(sizeof(int64_t) * n);
    int64_t r, k;

    for (r = 0; r < n; ++r)
        next[r] = CSC_Ap[r];

    for (r = 0; r < n; ++r)
    {
        for (k = A->ia[r] - 1; k < A->ia[r + 1] - 1; ++k)
        {
            int64_t col = A->ja[k] - 1;
            CSC_Ax[next[col]++] = A->a[k];
        }
    }

    SAFE_FREE(next);
}

void solver_start(const MKL_INT n)
{
    sys_n = n;
    Sym = NULL;
    Num = NULL;
    CSC_Ap = NULL;
    CSC_Ai = NULL;
    CSC_Ax = NULL;
    umfpack_dl_defaults(Control);
}

void solver_stop(void)
{
    if (Num)
        umfpack_dl_free_numeric(&Num);
    if (Sym)
        umfpack_dl_free_symbolic(&Sym);
    if (CSC_Ax)
        SAFE_FREE(CSC_Ax);
    if (CSC_Ai)
        SAFE_FREE(CSC_Ai);
    if (CSC_Ap)
        SAFE_FREE(CSC_Ap);
    sys_n = 0;
}

void solver_solve(double *u, csr_matrix *A, double *f)
{
    int status;

    // Symbolic factorization once (sparsity pattern is constant).
    if (!A->analysis_phase)
    {
        csr_to_csc(A);
        status = umfpack_dl_symbolic(sys_n, sys_n, CSC_Ap, CSC_Ai, CSC_Ax, &Sym, Control, Info);
        if (status != UMFPACK_OK)
        {
            printf("UMFPACK symbolic factorization failed: %d\n", status);
            exit(1);
        }
        A->analysis_phase = 1;
    }

    refresh_csc_values(A);

    if (Num)
        umfpack_dl_free_numeric(&Num);

    status = umfpack_dl_numeric(CSC_Ap, CSC_Ai, CSC_Ax, Sym, &Num, Control, Info);
    if (status != UMFPACK_OK)
    {
        printf("UMFPACK numeric factorization failed: %d\n", status);
        exit(2);
    }

    status = umfpack_dl_solve(UMFPACK_A, CSC_Ap, CSC_Ai, CSC_Ax, u, f, Num, Control, Info);
    if (status != UMFPACK_OK)
    {
        printf("UMFPACK solve failed: %d\n", status);
        exit(3);
    }
}

void solver_repeated_solve(double *u, csr_matrix *A, double *f)
{
    int status;

    (void)A;
    if (!Num)
    {
        printf("UMFPACK: repeated solve requested before factorization.\n");
        exit(3);
    }

    status = umfpack_dl_solve(UMFPACK_A, CSC_Ap, CSC_Ai, CSC_Ax, u, f, Num, Control, Info);
    if (status != UMFPACK_OK)
    {
        printf("UMFPACK solve failed: %d\n", status);
        exit(3);
    }
}

void solver_solve_low_rank(double *u, csr_matrix *A, double *f)
{
    // No low-rank update in UMFPACK; plain factor + solve.
    solver_solve(u, A, f);
}

void solver_diff_gen(rb_context *ctx)
{
    // No low-rank bookkeeping needed.
    (void)ctx;
}
