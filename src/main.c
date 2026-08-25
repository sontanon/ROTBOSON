#include "tools.h"
#include "context.h"

#include "parser.h"
#include "log.h"
#include "output.h"
#include "initial.h"
#include "rhs.h"
#include "solver.h"
#include "omega_calc.h"
#include "csr.h"
#include "nleq_err.h"
#include "nleq_res.h"
#include "newton.h"
#include "vector_algebra.h"
#include "cart_to_pol.h"
#include "analysis.h"

// ---------------------------------------------------------------------------
// Driver helpers. main() below is a thin driver; each stage is a named unit
// so the responsibilities are visible (banner / params / solver / analysis /
// sweep) instead of one ~700-line function.
// ---------------------------------------------------------------------------

static void print_banner(void)
{
    rb_log(RB_LOG_INFO,
           "******************************************************\n"
           "******************************************************\n"
           "***                                                \n"
           "***                    ROTBOSON                    \n"
           "***                                                \n"
           "***          Global Newton Method Version          \n"
           "***                                                \n"
           "***        Author: Santiago Ontanon Sanchez        \n"
           "***                                                \n"
           "***              ICN UNAM, Mexico City             \n"
           "***                                                \n"
           "***                                                \n"
           "***             First Revision: 01/08/2019         \n"
           "***                                                \n"
           "***             Last  Revision: 24/09/2020         \n"
           "***                                                \n"
           "******************************************************\n");
}

static void print_parameters(const rb_context *ctx)
{
    rb_log(RB_LOG_INFO,
           "******************************************************\n"
           "***                                                \n"
           "***           Generating Rotating Boson           \n"
           "***            Star Initial Data For NR.           \n"
           "***                                                \n"
           "***           GRID:                                \n"
           "***            dr          = %-12.10E          \n"
           "***            dz          = %-12.10E          \n"
           "***            dim         = %-7lld               \n"
           "***            NrInterior  = %-7lld               \n"
           "***            NzInterior  = %-7lld               \n"
           "***            order       = %lld                     \n"
           "***            ghost       = %lld                     \n"
           "***                                                \n"
           "***           SCALAR FIELD:                        \n"
           "***            l           = %-7lld               \n"
           "***            m           = %-12.10E          \n",
           ctx->dr, ctx->dz, ctx->dim, ctx->NrInterior, ctx->NzInterior, ctx->order, ctx->ghost,
           ctx->l, ctx->m);
    if (ctx->fixedPhi)
    {
        rb_log(RB_LOG_INFO,
               "***            Scalar Field is Fixed at:           \n"
               "***            r(fixedPhi) = %-12.10E          \n"
               "***            z(fixedPhi) = %-12.10E          \n",
               ctx->dr * (ctx->fixedPhiR - 0.5), ctx->dz * (ctx->fixedPhiZ - 0.5));
    }
    else if (ctx->fixedOmega)
    {
        rb_log(RB_LOG_INFO, "***            Initial Omega is Fixed.             \n");
    }
    rb_log(RB_LOG_INFO,
           "***                                                \n"
           "***           INITIAL DATA:                        \n"
           "***            readInitialData = %lld     \n",
           ctx->readInitialData);
    if (ctx->readInitialData)
    {
        rb_log(RB_LOG_INFO,
               "***            log_alpha_i = %-18s     \n"
               "***            beta_i      = %-18s     \n"
               "***            log_h_i     = %-18s     \n"
               "***            log_a_i     = %-18s     \n"
               "***            psi_i       = %-18s     \n"
               "***            lambda_i    = %-18s     \n"
               "***            w_i         = %-18s     \n",
               ctx->log_alpha_i, ctx->beta_i, ctx->log_h_i, ctx->log_a_i, ctx->psi_i,
               ctx->lambda_i, ctx->w_i);
    }
    else
    {
        rb_log(RB_LOG_INFO,
               "***            psi0        = %-12.10E          \n"
               "***            sigmaR      = %-12.10E          \n"
               "***            sigmaZ      = %-12.10E          \n"
               "***            rExt        = %-12.10E          \n",
               ctx->psi0, ctx->sigmaR, ctx->sigmaZ, ctx->rExt);
    }
    if (!ctx->w_i)
    {
        rb_log(RB_LOG_INFO, "***            w0          = %-12.10E          \n", ctx->w0);
    }
    rb_log(RB_LOG_INFO,
           "***                                                \n"
           "***           SOLVER:                              \n"
           "***            solverType    = %-18s  \n"
           "***            epsilon       = %-12.10E        \n"
           "***            maxNewtonIter = %-4lld                \n"
           "***            lambda0       = %-12.10E        \n"
           "***            lambdaMin     = %-12.10E        \n"
           "***            useLowRank    = %lld       \n"
           "***                                                \n"
           "******************************************************\n",
           (ctx->solverType == 1) ? "Error" : "Residual", ctx->epsilon, ctx->maxNewtonIter,
           ctx->lambda0, ctx->lambdaMin, ctx->useLowRank);
}

static void configure_openmp(void)
{
#pragma omp parallel
    {
#pragma omp master
        {
            // Determine OMP threads.
            rb_log(RB_LOG_INFO,
                   "******************************************************\n"
                   "***                                                \n"
                   "***            Maximum OMP threads = %d             \n"
                   "***            Currently running on %d              \n"
                   "***                                                \n"
                   "******************************************************\n",
                   omp_get_max_threads(), omp_get_num_threads());
            mkl_set_dynamic(0);
            mkl_set_num_threads(omp_get_num_threads());
        }
    }
}

// Newton orchestration: pick the error-/residual-/classic-Newton driver and
// run it, then record errCode. Returns the iteration index k (negative on
// failure, matching the solver's convention).

// Trial-iteration caps for the nleq_err/nleq_res inner loops.
#define MAX_TRIAL_A_ITERATIONS 8
#define MAX_TRIAL_B_ITERATIONS 8
static MKL_INT run_newton(rb_context *ctx, solution_writer *w, MKL_INT *errCode, double **u,
                          double **f, double **du, double **du_bar, double *norm_f,
                          double *norm_du, double *norm_du_bar, double *lambda, double *Theta,
                          double *mu, double *lambda_prime, double *mu_prime, csr_matrix *J,
                          rb_linear_solve_fn linear_solve_1, rb_linear_solve_fn linear_solve_2)
{
    MKL_INT k = 0;

    // Set initial damping factor lambda[0].
    lambda[0] = ctx->lambda0;

    /* MAIN ALGORITHM: NEWTON SOLVER */
    if (ctx->maxNewtonIter > 0)
    {
        switch (ctx->solverType)
        {
        // Error-based algorithm.
        case 1:
            k = nleq_err(ctx, errCode, u, f, lambda, du, du_bar, norm_du, norm_du_bar, Theta, mu,
                         lambda_prime, mu_prime, J, ctx->epsilon, ctx->maxNewtonIter,
                         MAX_TRIAL_A_ITERATIONS, MAX_TRIAL_B_ITERATIONS, ctx->lambdaMin,
                         ctx->localSolver, rhs, csr_gen_jacobian, norm2_all_variables,
                         dot_all_variables, linear_solve_1, linear_solve_2);
            break;
        // Residual-based algorithm.
        case 2:
            norm_f[0] = norm2_interior_all_variables(ctx, u[0]);
            k = nleq_res(ctx, errCode, u, f, lambda, du, norm_f, Theta, mu, lambda_prime, mu_prime,
                         J, ctx->epsilon, ctx->maxNewtonIter, MAX_TRIAL_A_ITERATIONS,
                         MAX_TRIAL_B_ITERATIONS, ctx->lambdaMin, ctx->localSolver, rhs,
                         csr_gen_jacobian, norm2_all_variables, dot_all_variables, linear_solve_1,
                         linear_solve_2);
            break;
        // Classic error-based Newton.
        case 3:
            norm_f[0] = norm2_interior_all_variables(ctx, u[0]);
            k = newton(ctx, errCode, u, f, lambda, du, norm_du, Theta, J, ctx->epsilon,
                       ctx->maxNewtonIter, rhs, csr_gen_jacobian, norm2_all_variables,
                       linear_solve_1);
            break;
        }

        // Write errCode to file.
        solution_writer_write_int_1d(w, "error_code", errCode, 1);

        // Check for convergence.
        if (*errCode != 0)
        {
            rb_log(RB_LOG_WARN,
                   "******************************************************\n"
                   "***                                                \n"
                   "***    Warning! Did not converge: Error Code = %lld  \n"
                   "***    Will output anyway. Do not trust results!   \n"
                   "***                                                \n"
                   "******************************************************\n",
                   *errCode);
            k = -k;
        }
    }
    else
    {
        rb_log(RB_LOG_WARN,
               "******************************************************\n"
               "***                                                \n"
               "***    Warning! User did not specify any Newton Iterations.  \n"
               "***                                                \n"
               "******************************************************\n");
        k = 0;
    }

    return k;
}

// Analysis phase: interpolate to spherical coordinates, write polar fields,
// and compute Komar masses / angular momenta / virial identities plus
// rr(phi_max).
static void run_analysis(rb_context *ctx, solution_writer *w, double **u, double *r, double *z,
                         MKL_INT k, double w_val)
{
    // Interpolate. Memory is allocated inside this subroutine.
    ex_cart_to_pol(&ctx->i_u, &ctx->i_rr, &ctx->i_th, r, z, u[k], ctx->Dr_u, ctx->Dz_u, ctx->Drz_u,
                   GNUM, ctx->dr, ctx->dz, ctx->NrInterior, ctx->NzInterior, ctx->ghost,
                   &ctx->NrrTotal, &ctx->NthTotal, &ctx->p_dim, &ctx->drr, &ctx->dth, &ctx->rr_inf);

    // Write spherical fields to file.
    solution_writer_write_2d_polar(w, "sph_rr", ctx->i_rr, ctx->NrrTotal, ctx->NthTotal);
    solution_writer_write_2d_polar(w, "sph_th", ctx->i_th, ctx->NrrTotal, ctx->NthTotal);
    solution_writer_write_2d_polar(w, "sph_log_alpha_f", ctx->i_u, ctx->NrrTotal,
                                   ctx->NthTotal);
    solution_writer_write_2d_polar(w, "sph_beta_f", ctx->i_u + ctx->p_dim, ctx->NrrTotal,
                                   ctx->NthTotal);
    solution_writer_write_2d_polar(w, "sph_log_h_f", ctx->i_u + 2 * ctx->p_dim, ctx->NrrTotal,
                                   ctx->NthTotal);
    solution_writer_write_2d_polar(w, "sph_log_a_f", ctx->i_u + 3 * ctx->p_dim, ctx->NrrTotal,
                                   ctx->NthTotal);
    solution_writer_write_2d_polar(w, "sph_psi_f", ctx->i_u + 4 * ctx->p_dim, ctx->NrrTotal,
                                   ctx->NthTotal);
    solution_writer_write_2d_polar(w, "sph_lambda_f", ctx->i_u + 5 * ctx->p_dim, ctx->NrrTotal,
                                   ctx->NthTotal);

    // Do analysis.
    ex_analysis(w, &ctx->M_KOMAR, &ctx->J_KOMAR, &ctx->GRV2, &ctx->GRV3, ctx->i_u, ctx->i_rr,
                ctx->i_th, w_val, ctx->m, ctx->l, ctx->ghost, ctx->order, ctx->NrrTotal,
                ctx->NthTotal, ctx->p_dim, ctx->drr, ctx->dth, ctx->rr_inf);

    // Calculate rr(phi_max).
    ex_phi_analysis(w, &ctx->phi_max, &ctx->rr_phi_max, &ctx->hwl_res, ctx->i_u, ctx->i_rr,
                    ctx->i_th, ctx->l, ctx->ghost, ctx->order, ctx->NrrTotal, ctx->NthTotal,
                    ctx->p_dim, ctx->drr, ctx->dth, ctx->rr_inf);

    // Clean analysis spherical variables.
    SAFE_FREE(ctx->i_rr);
    SAFE_FREE(ctx->i_th);
    SAFE_FREE(ctx->i_u);
}

// Sweep/continuation: sanity-check the converged solution, then either stop
// (return 0) or seed the next solve from the current solution with predicted
// scale factors (return 1).
static int sweep_advance(rb_context *ctx, double **u, MKL_INT k, double w, MKL_INT errCode,
                         csr_matrix *J)
{
    MKL_INT counter_i = 0;

    // Peaks and predicted scale factors (recomputed each step).
    double peak_next[GNUM + 1] = {0.0};
    double peak_prev[GNUM + 1] = {0.0};
    double next_scale[GNUM + 1] = {0.0};

    if (!((errCode == 0) && (ctx->sweep > 0)))
    {
        if (ctx->sweep > 0)
        {
            rb_log(RB_LOG_WARN,
                   "******************************************************\n"
                   "***                                                \n"
                   "***   Sweep cannot continue because errCode = %lld !\n"
                   "***                                                \n"
                   "******************************************************\n",
                   errCode);
        }
        return 0;
    }

    // Sanity checks on this resolution.
    if (w <= ctx->w_min || w >= ctx->w_max)
    {
        rb_log(RB_LOG_WARN,
               "******************************************************\n"
               "***                                                \n"
               "***   Sweep cannot continue because w is out of range (%.5E, %.5E) !\n"
               "***                                                \n"
               "******************************************************\n",
               ctx->w_min, ctx->w_max);
        return 0;
    }
    if (ctx->rr_phi_max < ctx->rr_phi_max_minimum)
    {
        rb_log(RB_LOG_WARN,
               "******************************************************\n"
               "***                                                \n"
               "***   Sweep cannot continue because rr(max(phi)) < min(rr(max(phi))) = %.5E !\n"
               "***                                                \n"
               "******************************************************\n",
               ctx->rr_phi_max_minimum);
        return 0;
    }
    if (ctx->rr_phi_max > ctx->rr_phi_max_maximum)
    {
        rb_log(RB_LOG_WARN,
               "******************************************************\n"
               "***                                                \n"
               "***   Sweep cannot continue because rr(max(phi)) > max(rr(max(phi))) = %.5E !\n"
               "***                                                \n"
               "******************************************************\n",
               ctx->rr_phi_max_maximum);
        return 0;
    }
    if (ctx->hwl_res < ctx->hwl_min)
    {
        rb_log(RB_LOG_WARN,
               "******************************************************\n"
               "***                                                \n"
               "***   Sweep cannot continue because N(HWL) < MIN(N(HWL)) = %lld !\n"
               "***   In other words, scalar field has not enough resolution. Try with more "
               "resolution or decrease hwl_min.\n"
               "***                                                \n"
               "******************************************************\n",
               ctx->hwl_min);
        return 0;
    }
    if (ctx->hwl_res > ctx->hwl_max)
    {
        rb_log(RB_LOG_WARN,
               "******************************************************\n"
               "***                                                \n"
               "***   Sweep cannot continue because N(HWL) > MAX(N(HWL)) = %lld !\n"
               "***   In other words, scalar field is too scattered. Try with less resolution or "
               "increase hwl_max.\n"
               "***                                                \n"
               "******************************************************\n",
               ctx->hwl_max);
        return 0;
    }

    // Predict scale factors from the peak values.
    for (counter_i = 0; counter_i < GNUM; ++counter_i)
    {
        peak_prev[counter_i] = ctx->u_seed[counter_i * ctx->dim +
                                           cblas_idamax(ctx->dim, u[k] + counter_i * ctx->dim, 1)];
        peak_next[counter_i] =
            u[k][counter_i * ctx->dim + cblas_idamax(ctx->dim, u[k] + counter_i * ctx->dim, 1)];
    }
    next_scale[4] = peak_next[4] / peak_prev[4];
    next_scale[0] = 1.0 + next_scale[4] * (1.0 - peak_prev[0] / peak_next[0]);
    next_scale[1] = 1.0 + next_scale[4] * (1.0 - peak_prev[1] / peak_next[1]);
    next_scale[2] = 1.0 + next_scale[4] * (1.0 - peak_prev[2] / peak_next[2]);
    next_scale[3] = 1.0 + next_scale[4] * (1.0 - peak_prev[3] / peak_next[3]);
    next_scale[5] = 1.0 + next_scale[4] * (1.0 - peak_prev[5] / peak_next[5]);

    for (counter_i = 0; counter_i < GNUM; ++counter_i)
        rb_log(RB_LOG_INFO,
               "**** Variable %lld peak = % -.5E, previous peak = % -.5E : predicted scale factor "
               "= %.5E\n",
               counter_i, peak_next[counter_i], peak_prev[counter_i], next_scale[counter_i]);

    // Omega prediction.
    peak_prev[GNUM] = omega_calc(ctx->u_seed[GNUM * ctx->dim], ctx->m);
    peak_next[GNUM] = omega_calc(u[k][GNUM * ctx->dim], ctx->m);

    next_scale[GNUM] = 1.0 + next_scale[4] * (1.0 - peak_prev[GNUM] / peak_next[GNUM]);

    rb_log(RB_LOG_INFO, "**** scaled w = %.5E, w = %.5E, scale_u6 = %.5E\n", next_scale[GNUM] * w, w,
           next_scale[GNUM]);

    // Transfer to initial data.
#pragma omp parallel shared(u)
    {
#pragma omp for schedule(dynamic, 1)
        for (counter_i = 0; counter_i < GNUM * ctx->dim + 1; ++counter_i)
        {
            u[0][counter_i] = -ctx->scale_next * ctx->u_seed[counter_i];
            u[0][counter_i] += (1.0 + ctx->scale_next) * u[k][counter_i];
            ctx->u_seed[counter_i] = u[k][counter_i];
        }
    }
    if (ctx->w_step != 0.0)
    {
        u[0][ctx->w_idx] = inverse_omega_calc(w + ctx->w_step, ctx->m);
    }
    // Set initial omega.
    ctx->w0 = omega_calc(u[0][ctx->w_idx], ctx->m);

    // Set analysis phase to 0 again.
    J->analysis_phase = 0;

    // Set new lambda0 to one since convergence has improved.
    ctx->lambda0 = 1.0;

    rb_log(RB_LOG_INFO,
           "******************************************************\n"
           "***                                                \n"
           "***   Setting initial data to last solution, scaling, and continuing...\n"
           "***                                                \n"
           "***                                                \n"
           "******************************************************\n");

    return 1;
}

int main(int argc, char *argv[])
{
    // Integer counter.
    MKL_INT i = 0, j = 0;

    // Stop index.
    MKL_INT k = 0;

    // Error code.
    MKL_INT errCode = 1;

    print_banner();

    // File name is in argv[1]. Check that we have at least one argument.
    if (argc < 2)
    {
        rb_log(RB_LOG_ERROR,
               "***                                                \n"
               "***           Usage: ./ROTBOSON file.toml          \n"
               "***                                                \n"
               "***            Missing parameter file.             \n"
               "***                                                \n"
               "******************************************************\n"
               "******************************************************\n");
        return EXIT_FAILURE;
    }

    // Runtime context (replaces the old param.h globals).
    rb_context ctx;
    rb_context_init(&ctx);

    // Local alias for the IDX macro, which indexes by row-major stride NzTotal.
    const MKL_INT NzTotal = ctx.NzTotal;

    // Parse the parameter file into ctx.
    parser(&ctx, argv[1]);

    print_parameters(&ctx);
    configure_openmp();

    // Allocate memory.
    rb_log(RB_LOG_INFO,
           "******************************************************\n"
           "***                                                \n"
           "***               Allocating memory...             \n"
           "***                                                \n");

    // Allocate pointer to double pointers.
    double **u = (double **)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double *));
    double **f = (double **)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double *));
    double **du = (double **)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double *));
    double **du_bar = (double **)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double *));

    // Allocate memory.
    for (i = 0; i < ctx.maxNewtonIter + 1; i++)
    {
        u[i] = (double *)SAFE_MALLOC((GNUM * ctx.dim + 1) * sizeof(double));
        f[i] = (double *)SAFE_MALLOC((GNUM * ctx.dim + 1) * sizeof(double));
        du[i] = (double *)SAFE_MALLOC((GNUM * ctx.dim + 1) * sizeof(double));
        du_bar[i] = (double *)SAFE_MALLOC((GNUM * ctx.dim + 1) * sizeof(double));
    }

    // Also include grids.
    double *r = (double *)SAFE_MALLOC(ctx.dim * sizeof(double));
    double *z = (double *)SAFE_MALLOC(ctx.dim * sizeof(double));

    // Initial data seed.
    ctx.u_seed = (double *)SAFE_MALLOC((GNUM * ctx.dim + 1) * sizeof(double));

    // Since these grids never change, fill them once and for all.
    // Fill coordinate grids.
    double aux_r;
#pragma omp parallel shared(r, z) private(i, j, aux_r)
    {
#pragma omp for schedule(dynamic, 1)
        for (i = 0; i < ctx.NrTotal; i++)
        {
            // Calculate rho value.
            aux_r = ((double)(i - ctx.ghost) + 0.5) * ctx.dr;
            // Loop over z points.
            for (j = 0; j < ctx.NzTotal; j++)
            {
                r[IDX(i, j)] = aux_r;
                z[IDX(i, j)] = ((double)(j - ctx.ghost) + 0.5) * ctx.dz;
            }
        }
    }

    // Auxiliary derivative buffers.
    ctx.Dr_u = (double *)SAFE_MALLOC((GNUM * ctx.dim + 1) * sizeof(double));
    ctx.Dz_u = (double *)SAFE_MALLOC((GNUM * ctx.dim + 1) * sizeof(double));
    ctx.Drr_u = (double *)SAFE_MALLOC((GNUM * ctx.dim + 1) * sizeof(double));
    ctx.Dzz_u = (double *)SAFE_MALLOC((GNUM * ctx.dim + 1) * sizeof(double));
    ctx.Drz_u = (double *)SAFE_MALLOC((GNUM * ctx.dim + 1) * sizeof(double));

    // Auxiliary variables.
    ctx.u_aux = (double *)SAFE_MALLOC(2 * ctx.dim * sizeof(double));
    ctx.Dr_u_aux = (double *)SAFE_MALLOC(2 * ctx.dim * sizeof(double));

    // Newton output parameters.
    double *norm_f = (double *)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double));
    double *norm_du = (double *)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double));
    double *norm_du_bar = (double *)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double));
    double *lambda = (double *)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double));
    double *Theta = (double *)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double));
    double *mu = (double *)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double));
    double *lambda_prime = (double *)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double));
    double *mu_prime = (double *)SAFE_MALLOC((ctx.maxNewtonIter + 1) * sizeof(double));

    // Initial guess norms.
    double f_norms[GNUM];

    // Final omega.
    double w = 0.0;

    rb_log(RB_LOG_INFO,
           "***               Finished allocation!             \n"
           "***                                                \n"
           "******************************************************\n");

    // Allocate PARDISO memory.
    rb_log(RB_LOG_INFO,
           "******************************************************\n"
           "***                                                \n"
           "***           Allocating PARDISO memory...         \n"
           "***                                                \n");

    // Initialize PARDISO memory and parameters.
    // Square matrix dimension is (GNUM * dim + 1).
    solver_start(GNUM * ctx.dim + 1);

    // Allocate CSR matrix.
    csr_matrix J;
    MKL_INT nnz = nnz_jacobian(&ctx);
    csr_allocate(&J, GNUM * ctx.dim + 1, GNUM * ctx.dim + 1, nnz);

    rb_log(RB_LOG_INFO,
           "***                                                \n"
           "***            Allocated CSR matrix with:          \n"
           "***             Rows      = %-6lld                 \n"
           "***             Columns   = %-6lld                 \n"
           "***             Non-zeros = %-12lld           \n"
           "***                                                \n"
           "***           Finished PARDISO allocation!         \n"
           "***                                                \n"
           "******************************************************\n",
           J.nrows, J.ncols, J.nnz);

    // LOW RANK UPDATE and linear solver subroutines.
    rb_linear_solve_fn linear_solve_1;
    if (ctx.useLowRank)
    {
        linear_solve_1 = solver_solve_low_rank;
        solver_diff_gen(&ctx);
    }
    else
        linear_solve_1 = solver_solve;

    // Repeated solver.
    rb_linear_solve_fn linear_solve_2 = solver_repeated_solve;

    rb_log(RB_LOG_INFO,
           "******************************************************\n"
           "***                                                \n"
           "***          Setting initial guess and RHS.        \n");

    // Set initial guess.
    initial_guess(&ctx, u[0]);

    // Loop over sweep.
    do
    {
        // Open the solution writer for this sweep step (path-aware; no chdir).
        solution_writer *sw = solution_writer_open(ctx.initial_dirname, &ctx, argv[1]);
        if (!sw)
        {
            rb_log(RB_LOG_ERROR, "OUTPUT: cannot open solution writer for \"%s\".\n",
                   ctx.initial_dirname);
            return EXIT_FAILURE;
        }

        // Print main variables.
        solution_writer_write_2d(sw, "log_alpha_i", u[0], ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "beta_i", u[0] + ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "log_h_i", u[0] + 2 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "log_a_i", u[0] + 3 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "psi_i", u[0] + 4 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "lambda_i", u[0] + 5 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_1d(sw, "w_i", &ctx.w0, 1);

        // Also print r, z grids.
        solution_writer_write_2d(sw, "r", r, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "z", z, ctx.NrTotal, ctx.NzTotal);

        // And initial "seed".
        solution_writer_write_2d(sw, "log_alpha_seed", ctx.u_seed, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "beta_seed", ctx.u_seed + ctx.dim, ctx.NrTotal,
                                 ctx.NzTotal);
        solution_writer_write_2d(sw, "log_h_seed", ctx.u_seed + 2 * ctx.dim, ctx.NrTotal,
                                 ctx.NzTotal);
        solution_writer_write_2d(sw, "log_a_seed", ctx.u_seed + 3 * ctx.dim, ctx.NrTotal,
                                 ctx.NzTotal);
        solution_writer_write_2d(sw, "psi_seed", ctx.u_seed + 4 * ctx.dim, ctx.NrTotal,
                                 ctx.NzTotal);
        solution_writer_write_2d(sw, "lambda_seed", ctx.u_seed + 5 * ctx.dim, ctx.NrTotal,
                                 ctx.NzTotal);
        w = omega_calc(ctx.u_seed[GNUM * ctx.dim], ctx.m);
        solution_writer_write_1d(sw, "w_seed", &w, 1);

        // First calculate initial RHS.
        rhs(&ctx, f[0], u[0]);

        // Print initial RHS.
        solution_writer_write_2d(sw, "f0_i", f[0], ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "f1_i", f[0] + ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "f2_i", f[0] + 2 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "f3_i", f[0] + 3 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "f4_i", f[0] + 4 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "f5_i", f[0] + 5 * ctx.dim, ctx.NrTotal, ctx.NzTotal);

        // Calculate 2-norms.
        f_norms[0] = norm2(&ctx, f[0]);
        f_norms[1] = norm2(&ctx, f[0] + ctx.dim);
        f_norms[2] = norm2(&ctx, f[0] + 2 * ctx.dim);
        f_norms[3] = norm2(&ctx, f[0] + 3 * ctx.dim);
        f_norms[4] = norm2(&ctx, f[0] + 4 * ctx.dim);
        f_norms[5] = norm2(&ctx, f[0] + 5 * ctx.dim);

        rb_log(RB_LOG_INFO,
               "***                                                \n"
               "***        INITIAL GUESS:                          \n"
               "***           || f0 ||   = %-12.10E           \n"
               "***           || f1 ||   = %-12.10E           \n"
               "***           || f2 ||   = %-12.10E           \n"
               "***           || f3 ||   = %-12.10E           \n"
               "***           || f4 ||   = %-12.10E           \n"
               "***           || f5 ||   = %-12.10E           \n"
               "***                                                \n"
               "***                                                \n"
               "******************************************************\n",
               f_norms[0], f_norms[1], f_norms[2], f_norms[3], f_norms[4], f_norms[5]);

        // Newton solve.
        k = run_newton(&ctx, sw, &errCode, u, f, du, du_bar, norm_f, norm_du, norm_du_bar, lambda,
                       Theta, mu, lambda_prime, mu_prime, &J, linear_solve_1, linear_solve_2);

        // Get omega.
        w = omega_calc(u[k][ctx.w_idx], ctx.m);

        // Print final solutions.
        solution_writer_write_2d(sw, "log_alpha_f", u[k], ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "beta_f", u[k] + ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "log_h_f", u[k] + 2 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "log_a_f", u[k] + 3 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "psi_f", u[k] + 4 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "lambda_f", u[k] + 5 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_1d(sw, "w_f", &w, 1);

        // Print final update.
        if (k > 0)
        {
            solution_writer_write_2d(sw, "du0_f", du[k - 1], ctx.NrTotal, ctx.NzTotal);
            solution_writer_write_2d(sw, "du1_f", du[k - 1] + ctx.dim, ctx.NrTotal, ctx.NzTotal);
            solution_writer_write_2d(sw, "du2_f", du[k - 1] + 2 * ctx.dim, ctx.NrTotal,
                                     ctx.NzTotal);
            solution_writer_write_2d(sw, "du3_f", du[k - 1] + 3 * ctx.dim, ctx.NrTotal,
                                     ctx.NzTotal);
            solution_writer_write_2d(sw, "du4_f", du[k - 1] + 4 * ctx.dim, ctx.NrTotal,
                                     ctx.NzTotal);
            solution_writer_write_2d(sw, "du5_f", du[k - 1] + 5 * ctx.dim, ctx.NrTotal,
                                     ctx.NzTotal);
        }

        // Print final RHS.
        solution_writer_write_2d(sw, "f0_f", f[k], ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "f1_f", f[k] + ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "f2_f", f[k] + 2 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "f3_f", f[k] + 3 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "f4_f", f[k] + 4 * ctx.dim, ctx.NrTotal, ctx.NzTotal);
        solution_writer_write_2d(sw, "f5_f", f[k] + 5 * ctx.dim, ctx.NrTotal, ctx.NzTotal);

        // Also print Newton parameters.
        switch (ctx.solverType)
        {
        case 1:
            solution_writer_write_1d(sw, "norm_du", norm_du, k);
            solution_writer_write_1d(sw, "norm_du_bar", norm_du_bar, k);
            break;
        case 2:
            solution_writer_write_1d(sw, "norm_f", norm_f, k);
            break;
        }

        solution_writer_write_1d(sw, "lambda", lambda, k);
        solution_writer_write_1d(sw, "Theta", Theta, k);
        solution_writer_write_1d(sw, "mu", mu, k);
        solution_writer_write_1d(sw, "lambda_prime", lambda_prime, k);
        solution_writer_write_1d(sw, "mu_prime", mu_prime, k);

        // Print final iteration's RHS's norms.
        f_norms[0] = norm2(&ctx, f[k]);
        f_norms[1] = norm2(&ctx, f[k] + ctx.dim);
        f_norms[2] = norm2(&ctx, f[k] + 2 * ctx.dim);
        f_norms[3] = norm2(&ctx, f[k] + 3 * ctx.dim);
        f_norms[4] = norm2(&ctx, f[k] + 4 * ctx.dim);
        f_norms[5] = norm2(&ctx, f[k] + 5 * ctx.dim);
        rb_log(RB_LOG_INFO,
               "***                                                \n"
               "***        FINAL ITERATION:                        \n"
               "***           || f0 ||   = %-12.10E           \n"
               "***           || f1 ||   = %-12.10E           \n"
               "***           || f2 ||   = %-12.10E           \n"
               "***           || f3 ||   = %-12.10E           \n"
               "***           || f4 ||   = %-12.10E           \n"
               "***           || f5 ||   = %-12.10E           \n"
               "***                                                \n"
               "***                                                \n"
               "******************************************************\n",
               f_norms[0], f_norms[1], f_norms[2], f_norms[3], f_norms[4], f_norms[5]);

        // Also print omega.
        rb_log(RB_LOG_INFO,
               "******************************************************\n"
               "***                                                \n"
               "***           FINAL OMEGA:                         \n"
               "***            w          = %-12.10E            \n"
               "***                                                \n"
               "******************************************************\n",
               w);

        // ANALYSIS PHASE.
        run_analysis(&ctx, sw, u, r, z, k, w);

        // Close the writer (flushes HDF5 attributes and closes the file).
        solution_writer_close(sw);

        // Rename directory to include w.
        snprintf(ctx.final_dirname, MAX_STR_LEN, "l=%lld,w=%.5E,dr=%.5E,N=%04lld", ctx.l, w, ctx.dr,
                 ctx.NrInterior);
        rename(ctx.initial_dirname, ctx.final_dirname);

        // Sweep continuation if sanity checks pass.
        if (!sweep_advance(&ctx, u, k, w, errCode, &J))
            break;
    } while (ctx.sweep > 0);

    // Clear memory.
    rb_log(RB_LOG_INFO,
           "******************************************************\n"
           "***                                                \n"
           "***              Deallocating memory...            \n"
           "***                                                \n");

    solver_stop();
    csr_deallocate(&J);

    // Free main variables with full maxNewtonIter size by looping inside them.
    for (i = 0; i < ctx.maxNewtonIter + 1; i++)
    {
        SAFE_FREE(u[i]);
        SAFE_FREE(f[i]);
        SAFE_FREE(du[i]);
        SAFE_FREE(du_bar[i]);
    }
    // Once all clear, free top pointer.
    SAFE_FREE(u);
    SAFE_FREE(f);
    SAFE_FREE(du);
    SAFE_FREE(du_bar);

    // Coordinate grids.
    SAFE_FREE(r);
    SAFE_FREE(z);

    // Derivatives.
    SAFE_FREE(ctx.Dr_u);
    SAFE_FREE(ctx.Dz_u);
    SAFE_FREE(ctx.Drr_u);
    SAFE_FREE(ctx.Dzz_u);
    SAFE_FREE(ctx.Drz_u);

    // Auxiliary variables.
    SAFE_FREE(ctx.u_aux);
    SAFE_FREE(ctx.Dr_u_aux);

    // Newton variables.
    SAFE_FREE(norm_f);
    SAFE_FREE(norm_du);
    SAFE_FREE(norm_du_bar);
    SAFE_FREE(lambda);
    SAFE_FREE(Theta);
    SAFE_FREE(mu);
    SAFE_FREE(lambda_prime);
    SAFE_FREE(mu_prime);

    // Initial data seed.
    SAFE_FREE(ctx.u_seed);

    rb_log(RB_LOG_INFO,
           "***              Finished deallocation!            \n"
           "***                                                \n"
           "******************************************************\n");

    // Print final message.
    rb_log(RB_LOG_INFO,
           "******************************************************\n"
           "***                                                \n"
           "***           All done! Have a nice day!           \n"
           "***                                                \n"
           "******************************************************\n"
           "******************************************************\n");

    // All done.
    return 0;
}
