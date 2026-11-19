#include "tools.h"
#include "pardiso_param.h"
#include "pardiso.h"
#include "exit_codes.h"
#include "profile.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Single-precision solve mode (ROTBOSON_PRECISION=32).
//
// With iparm[28]=1 MKL PARDISO keeps all internal data (and the LU factors)
// in float, halving factor memory; the CSR index arrays stay int64. The
// caller-side matrix/RHS/solution stay double (Newton, assembly, output);
// only the arrays handed to pardiso_64 are converted. Precision mode is
// stored in the handle and can only be set during phase 11, so the flag is
// read once in solver_start (pardiso_start.c).
//
// Accuracy note: a plain fp32 solve carries ~1e-7 relative error. Newton-
// level iterative refinement in double (Stage 2 of the M3 fp32 plan,
// data/summaries/profile-baseline.md) recovers near-double accuracy; these
// helpers only do the Stage-1 plumbing.
// ---------------------------------------------------------------------------
static float *fp32_a = NULL;      // float copy of CSR values
static float *fp32_b = NULL;      // float copy of RHS
static float *fp32_x = NULL;      // float solution buffer
static MKL_INT fp32_a_cap = 0;    // capacity of fp32_a (floats)
static MKL_INT fp32_vec_cap = 0;  // capacity of fp32_b / fp32_x

// Stage 2 IR buffers (see fp32_ir_refine below).
static double *ir_r = NULL;       // double residual buffer
static double *ir_du_bak = NULL;  // tentative-update rollback buffer
static MKL_INT ir_cap = 0;

static int fp32_enabled(void)
{
    const char *env = getenv("ROTBOSON_PRECISION");
    return env && strcmp(env, "32") == 0;
}

// Lazily grow the fp32 buffers; returns 0 on allocation failure.
static int fp32_ensure(MKL_INT nnz, MKL_INT n)
{
    if (fp32_a_cap < nnz)
    {
        float *p = (float *)realloc(fp32_a, (size_t)nnz * sizeof(float));
        if (!p) return 0;
        fp32_a = p;
        fp32_a_cap = nnz;
    }
    if (fp32_vec_cap < n)
    {
        float *b = (float *)realloc(fp32_b, (size_t)n * sizeof(float));
        if (!b) return 0;
        fp32_b = b;
        float *x = (float *)realloc(fp32_x, (size_t)n * sizeof(float));
        if (!x) return 0;
        fp32_x = x;
        fp32_vec_cap = n;
    }
    return 1;
}

// Release the fp32 buffers (called from solver_stop).
void solver_fp32_cleanup(void)
{
    free(fp32_a); fp32_a = NULL; fp32_a_cap = 0;
    free(fp32_b); fp32_b = NULL;
    free(fp32_x); fp32_x = NULL; fp32_vec_cap = 0;
    free(ir_r); ir_r = NULL;
    free(ir_du_bak); ir_du_bak = NULL; ir_cap = 0;
}

// Convert the double CSR values and RHS into the fp32 buffers, and return
// the pointers to hand to pardiso_64. Per the MKL C prototype the pointers
// are typed double*; when fp32 is enabled they actually point at float
// arrays (iparm[28]=1 makes PARDISO interpret them as single precision).
static void fp32_convert(csr_matrix *A, double *f, double **a_out, double **b_out,
                         double **x_out)
{
    MKL_INT nnz = A->nnz;
    for (MKL_INT k = 0; k < nnz; ++k)
        fp32_a[k] = (float)A->a[k];
    for (MKL_INT k = 0; k < n; ++k)
        fp32_b[k] = (float)f[k];
    *a_out = (double *)fp32_a;
    *b_out = (double *)fp32_b;
    *x_out = (double *)fp32_x;
}

// ---------------------------------------------------------------------------
// Stage 2: double-residual iterative refinement (IR) for the fp32+direct
// path (M3 fp32 plan; design and gates in data/summaries/profile-baseline.md).
//
// A plain fp32 solve carries ~1e-7 relative error; near folds that is enough
// to turn Newton convergence marginal (measured: the post-regrid fold step
// stalls ~50 min in fp32-CGS, cf. 20 s double-CGS). IR recovers double-grade
// solves at ~one extra back-substitution per step:
//
//   r = f - A·du      (double matvec against the untouched double Jacobian)
//   A·delta = r       (extra phase-33 call reusing the SAME fp32 factors)
//   du += delta       (double accumulate; accept only if ||r|| strictly drops)
//
// Each accepted step multiplies the solve error by ~kappa·u_f; for the
// well-conditioned bulk that is ~1e-7 -> ~1e-12 in two steps. Stagnation
// (kappa·u_f ~ 1, deep fold) degrades gracefully to the Stage-1 result.
// Default ROTBOSON_FP32_IR=2; 0 disables. Only active with fp32 + direct
// (CGS already inverts to 1e-6 in double and has its own pathology there).
// ---------------------------------------------------------------------------
static int fp32_direct_enabled(void)
{
    return fp32_enabled() && getenv("ROTBOSON_PARDISO_DIRECT") != NULL;
}

static int fp32_ir_steps(void)
{
    const char *env = getenv("ROTBOSON_FP32_IR");
    int steps = 2;
    if (env && *env)
    {
        steps = atoi(env);
        if (steps < 0)
            steps = 0;
        if (steps > 4)
            steps = 4;
    }
    return steps;
}

static int ir_ensure(MKL_INT dim)
{
    if (ir_cap >= dim)
        return 1;
    double *r = (double *)realloc(ir_r, (size_t)dim * sizeof(double));
    if (!r)
        return 0;
    ir_r = r;
    double *bak = (double *)realloc(ir_du_bak, (size_t)dim * sizeof(double));
    if (!bak)
        return 0;
    ir_du_bak = bak;
    ir_cap = dim;
    return 1;
}

// r = A x in double (1-based CSR, BASE convention; matches pardiso indexing).
static void csr_matvec_double(const csr_matrix *A, const double *x, double *r)
{
#pragma omp parallel for schedule(static)
    for (MKL_INT i = 0; i < n; ++i)
    {
        double s = 0.0;
        MKL_INT p0 = A->ia[i] - BASE;
        MKL_INT p1 = A->ia[i + 1] - BASE;
        for (MKL_INT p = p0; p < p1; ++p)
            s += A->a[p] * x[A->ja[p] - BASE];
        r[i] = s;
    }
}

// Refine du (approximate solution of A du = f held in double) in place.
// a_f: float values array already handed to PARDISO (typed double* per the
// prototype). Residual ladder is printed when profiling is enabled.
static void fp32_ir_refine(csr_matrix *A, double *f, double *du, double *a_f, int R)
{
    if (R <= 0 || !ir_ensure(n))
        return;

    double bn = cblas_dnrm2(n, f, 1);

    csr_matvec_double(A, du, ir_r);
    for (MKL_INT k = 0; k < n; ++k)
        ir_r[k] = f[k] - ir_r[k];
    double rn = cblas_dnrm2(n, ir_r, 1);
    double r0 = rn;
    int steps = 0;

    prof_begin(PROF_IR);
    for (int it = 0; it < R; ++it)
    {
        // Already at the fp32-to-double noise floor.
        if (rn <= 1e-14 * (bn > 1.0 ? bn : 1.0))
            break;

        // Correction solve A delta = r, reusing the existing fp32 factors.
        for (MKL_INT k = 0; k < n; ++k)
            fp32_b[k] = (float)ir_r[k];
        phase = 33;
        pardiso_64(pt, &maxfct, &mnum, &mtype, &phase, &n, a_f, A->ia, A->ja, perm, &nrhs, iparm,
                   &msglvl, (double *)fp32_b, (double *)fp32_x, &error);
        if (error != 0)
            break;

        // Tentative update, re-check the residual in double; revert if no gain.
        for (MKL_INT k = 0; k < n; ++k)
        {
            ir_du_bak[k] = du[k];
            du[k] += (double)fp32_x[k];
        }
        csr_matvec_double(A, du, ir_r);
        for (MKL_INT k = 0; k < n; ++k)
            ir_r[k] = f[k] - ir_r[k];
        double rn_new = cblas_dnrm2(n, ir_r, 1);
        if (rn_new >= rn)
        {
            for (MKL_INT k = 0; k < n; ++k)
                du[k] = ir_du_bak[k];
            break;
        }
        rn = rn_new;
        steps++;
    }
    prof_end(PROF_IR);

    if (prof_enabled())
        printf("PARDISO IR DIAGNOSTICS: ||b||=%.3e ||r0||=%.3e ||r_final||=%.3e ir_steps=%d\n", bn,
               r0, rn, steps);
}

void solver_solve(double *u, // Solution array.
                  csr_matrix *A, // Matrix system to solve: Au = f.
                  double *f) // RHS array.
{
    // Stage 1 fp32 plumbing: convert the matrix values and RHS once per call
    // and hand float arrays to pardiso_64 (iparm[28]=1, set in solver_start).
    double *sa, *sb, *sx;
    if (fp32_enabled())
    {
        if (!fp32_ensure(A->nnz, n))
        {
            printf("ERROR: fp32 buffer allocation failed.\n");
            exit(RB_EXIT_SOLVER);
        }
        fp32_convert(A, f, &sa, &sb, &sx);
    }
    else
    {
        sa = A->a;
        sb = f;
        sx = u;
    }

    // Reordering and symbolic factorization.
    if (!(A->analysis_phase))
    {
        prof_begin(PROF_ANALYSIS);
        phase = 11;
        pardiso_64(pt, &maxfct, &mnum, &mtype, &phase, &n, sa, A->ia, A->ja, perm, &nrhs, iparm,
                   &msglvl, &ddum, &ddum, &error);

        if (error != 0)
        {
            printf("ERROR during symbolic factorization: %lld.\n", error);
            for (MKL_INT k = 0; k < 64; ++k)
                printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
            exit(RB_EXIT_SOLVER);
        }

        printf("PARDISO MEMORY DIAGNOSTICS\n");
        printf("iparm[14] : % 9lld kB : Peak memory on symbolic factorization.\n", iparm[14]);
        printf("iparm[15] : % 9lld kB : Permament memory on symbolic factorization.\n", iparm[15]);
        printf("iparm[16] : % 9lld kB : Peak memory on numerical factorization and solution.\n",
               iparm[16]);
        printf("iparm[62] : % 9lld kB : Minimum OOC memory for numerical factorization and "
               "solution.\n",
               iparm[62]);
        printf("Total peak memory consumption for IC  : %09lld kB : MAX(iparm[14], iparm[15] + "
               "iparm[16]).\n",
               MAX(iparm[14], iparm[15] + iparm[16]));
        printf("Total peak memory consumption for OOC : %09lld kB : MAX(iparm[14], iparm[15] + "
               "iparm[62]).\n",
               MAX(iparm[14], iparm[15] + iparm[62]));
        printf("\n");

        // Analysis phase has been completed.
        A->analysis_phase = 1;

        // iparm[3]=61 turns every phase-33 solve into an LU-preconditioned CGS
        // iteration loop (stopping criterion 1e-6) instead of a direct
        // back-substitution, and phase-22 then REUSES the previous LU as
        // preconditioner instead of refactorizing.
        //
        // Measured (data/summaries/profile-baseline.md):
        //   - direct mode wins on solves-per-factor-heavy problems: master
        //     seed 58 s -> 42 s (-28%), solutions bit-identical;
        //   - direct mode LOSES on continuation campaigns: full l=1 runbook
        //     campaign 2778 s -> 3825 s (+38%), because near-fold Newton
        //     matrices are near-singular and direct refactorization blows up
        //     (56 s/factor at N=256 fold steps vs ~20 s total per step in CGS
        //     mode with a stale-LU preconditioner).
        // Default remains CGS (proven by the M2 catalogue); opt into direct
        // mode (exact solves, no stale-LU reuse) with ROTBOSON_PARDISO_DIRECT=1.
        if (getenv("ROTBOSON_PARDISO_DIRECT") == NULL)
            iparm[3] = 61;

        prof_end(PROF_ANALYSIS);
    }

    // Numerical factorization.
    prof_begin(PROF_FACTOR);
    phase = 22;
    pardiso_64(pt, &maxfct, &mnum, &mtype, &phase, &n, sa, A->ia, A->ja, perm, &nrhs, iparm,
               &msglvl, &ddum, &ddum, &error);

    if (error != 0)
    {
        printf("ERROR during numerical factorization: %lld.\n", error);
        for (MKL_INT k = 0; k < 64; ++k)
            printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
        exit(RB_EXIT_SOLVER);
    }
    prof_end(PROF_FACTOR);

    // Factor diagnostics are filled by the numerical factorization phase.
    printf("PARDISO FACTOR DIAGNOSTICS\n");
    printf("iparm[17] : % 9lld : Number of nonzeros in factors.\n", iparm[18 - 1]);
    printf("iparm[18] : % 9lld : Number of factorization MFLOPS.\n", iparm[19 - 1]);
    printf("\n");

#ifdef VERBOSE
    printf("PARDISO: Factorization completed.\n");
#endif

    // Back substitution and iterative refinement.
    prof_begin(PROF_SOLVE);
    phase = 33;
    pardiso_64(pt, &maxfct, &mnum, &mtype, &phase, &n, sa, A->ia, A->ja, perm, &nrhs, iparm,
               &msglvl, sb, sx, &error);

    if (error != 0)
    {
        printf("ERROR during solution: %lld,\n", error);
        for (MKL_INT k = 0; k < 64; ++k)
            printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
        exit(RB_EXIT_SOLVER);
    }
    if (fp32_enabled())
        for (MKL_INT k = 0; k < n; ++k)
            u[k] = (double)fp32_x[k];
    prof_end(PROF_SOLVE);

    // Stage 2: double-residual refinement of the fp32+direct solve.
    if (fp32_direct_enabled())
        fp32_ir_refine(A, f, u, sa, fp32_ir_steps());

    // CGS diagnostics (iparm[19] is set by the solution phase when iparm[3] > 0).
    printf("PARDISO SOLVE DIAGNOSTICS\n");
    printf("iparm[19]: % 9lld : CGS/CG diagnostics (iterations*10 - error, <0 failed).\n", iparm[19]);
    printf("\n");

    // Return.
    return;
}

void solver_solve_low_rank(double *u, // Solution array.
                           csr_matrix *A, // Matrix system to solve: Au = f.
                           double *f) // RHS array.
{
    // If using low-rank, calls are different.
    // Notice in particular that diff is used instead of perm array.
    if (!(A->analysis_phase))
    {
        // Modify parameters according to Low-Rank update.
        iparm[24 - 1] = 10;
        // No permutation.
        iparm[5 - 1] = 0;
        // No CGS.
        iparm[4 - 1] = 0;
        // Additional values.
        iparm[6 - 1] = 0;
        iparm[12 - 1] = 0;
        iparm[28 - 1] = 0;
        iparm[31 - 1] = 0;
        iparm[36 - 1] = 0;
        iparm[37 - 1] = 0;
        iparm[56 - 1] = 0;
        // Ensure we are using IC.
        iparm[60 - 1] = 0;

        // Reordering and symbolic factorization.
        phase = 11;
        pardiso_64(pt, &maxfct, &mnum, &mtype, &phase, &n, A->a, A->ia, A->ja, perm, &nrhs, iparm,
                   &msglvl, &ddum, &ddum, &error);

        if (error != 0)
        {
            printf("ERROR during symbolic factorization: %lld.\n", error);
            for (MKL_INT k = 0; k < 64; ++k)
                printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
            exit(RB_EXIT_SOLVER);
        }

        printf("PARDISO MEMORY DIAGNOSTICS\n");
        printf("iparm[14] : % 9lld kB : Peak memory on symbolic factorization.\n", iparm[14]);
        printf("iparm[15] : % 9lld kB : Permament memory on symbolic factorization.\n", iparm[15]);
        printf("iparm[16] : % 9lld kB : Peak memory on numerical factorization and solution.\n",
               iparm[16]);
        printf("iparm[62] : % 9lld kB : Minimum OOC memory for numerical factorization and "
               "solution.\n",
               iparm[62]);
        printf("Total peak memory consumption for IC  : %09lld kB : MAX(iparm[14], iparm[15] + "
               "iparm[16]).\n",
               MAX(iparm[14], iparm[15] + iparm[16]));
        printf("Total peak memory consumption for OOC : %09lld kB : MAX(iparm[14], iparm[15] + "
               "iparm[62]).\n",
               MAX(iparm[14], iparm[15] + iparm[62]));
        printf("\n");

        printf("PARDISO FACTOR DIAGNOSTICS\n");
        printf("iparm[17] : % 9lld : Number of nonzeros in factors.\n", iparm[18 - 1]);
        printf("iparm[18] : % 9lld : Number of factorization MFLOPS.\n", iparm[19 - 1]);
        printf("\n");

        // Now that we have done the analysis phase, set the matrix handle.
        A->analysis_phase = 1;
    }

    // Numerical factorization.
    phase = 22;
    pardiso_64(pt, &maxfct, &mnum, &mtype, &phase, &n, A->a, A->ia, A->ja, diff, &nrhs, iparm,
               &msglvl, &ddum, &ddum, &error);

    if (error != 0)
    {
        printf("ERROR during numerical factorization: %lld.\n", error);
        for (MKL_INT k = 0; k < 64; ++k)
            printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
        exit(RB_EXIT_SOLVER);
    }

#ifdef VERBOSE
    printf("PARDISO: Factorization completed.\n");
#endif

    // Back substitution and iterative refinement.
    phase = 33;
    pardiso_64(pt, &maxfct, &mnum, &mtype, &phase, &n, A->a, A->ia, A->ja, diff, &nrhs, iparm,
               &msglvl, f, u, &error);

    if (error != 0)
    {
        printf("ERROR during solution: %lld,\n", error);
        for (MKL_INT k = 0; k < 64; ++k)
            printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
        exit(RB_EXIT_SOLVER);
    }

    // Turn Low-Rank for future updates.
    iparm[39 - 1] = 1;

    // Return.
    return;
}

void solver_repeated_solve(double *u, // Solution array.
                           csr_matrix *A, // Matrix system to solve: Au = f.
                           double *f) // RHS array.
{
    // Stage 1 fp32 plumbing (same conversion as solver_solve).
    double *sa, *sb, *sx;
    if (fp32_enabled())
    {
        if (!fp32_ensure(A->nnz, n))
        {
            printf("ERROR: fp32 buffer allocation failed.\n");
            exit(RB_EXIT_SOLVER);
        }
        fp32_convert(A, f, &sa, &sb, &sx);
    }
    else
    {
        sa = A->a;
        sb = f;
        sx = u;
    }

    /*
    // Numerical factorization.
    phase = 22;
    pardiso_64(pt, &maxfct, &mnum, &mtype, &phase,
        &n, A->a, A->ia, A->ja, perm, &nrhs,
        iparm, &msglvl, &ddum, &ddum, &error);

    if (error != 0)
    {
        printf("ERROR during numerical factorization: %lld.\n", error);
        exit(RB_EXIT_SOLVER);
    }

#ifdef VERBOSE
    printf("PARDISO: Factorization completed.\n");
#endif
    */

    // Back substitution and iterative refinement.
    prof_begin(PROF_SOLVE);
    phase = 33;
    pardiso_64(pt, &maxfct, &mnum, &mtype, &phase, &n, sa, A->ia, A->ja, perm, &nrhs, iparm,
               &msglvl, sb, sx, &error);

    if (error != 0)
    {
        printf("ERROR during solution: %lld,\n", error);
        for (MKL_INT k = 0; k < 64; ++k)
            printf("iparm(%lld) = %lld\n", k + 1, iparm[k]);
        exit(RB_EXIT_SOLVER);
    }
    if (fp32_enabled())
        for (MKL_INT k = 0; k < n; ++k)
            u[k] = (double)fp32_x[k];
    prof_end(PROF_SOLVE);

    // Stage 2: double-residual refinement of the fp32+direct solve.
    if (fp32_direct_enabled())
        fp32_ir_refine(A, f, u, sa, fp32_ir_steps());

    // Return.
    return;
}
