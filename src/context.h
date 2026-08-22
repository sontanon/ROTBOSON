// ROTBOSON runtime context.
//
// This struct holds every quantity that used to live as a file-scope global in
// param.h (the "#ifdef MAIN_FILE + extern" pattern). It is owned by main() and
// passed explicitly to the routines that need it, so data flow is visible and
// the numeric kernels are no longer implicitly coupled to global state.
//
// Design notes:
//   - "Params" (grid, field, solver, sweep) are set once by the parser and
//     read-only afterwards.
//   - "State" (derivative buffers, spherical interpolation arrays, analysis
//     outputs, seed, output directory names) is mutated across the solve.
//   - MKL_INT is used for integer parameters to match the MKL ILP64 backend;
//     see PLAN.md Phase 6 for the eventual de-MKL-ification.
#ifndef ROTBOSON_CONTEXT_H
#define ROTBOSON_CONTEXT_H

// tools.h provides MKL_INT, csr_matrix, and the safe-allocation helpers.
#include "tools.h"

// Number of grid variables (lapse, shift, h, a, scalar field, regularization).
#define GNUM 6

// String length for paths and directory names.
#define MAX_STR_LEN 256

typedef struct rb_context
{
	// -- GRID ---------------------------------------------------------------
	double dr;
	double dz;
	MKL_INT NrInterior;
	MKL_INT NzInterior;
	MKL_INT NrTotal;
	MKL_INT NzTotal;
	MKL_INT dim;
	MKL_INT ghost;
	MKL_INT order;

	// -- SCALAR FIELD -------------------------------------------------------
	MKL_INT l;
	double m;
	double psi0;
	double sigmaR;
	double sigmaZ;
	double rExt;
	double w0;
	MKL_INT w_idx;
	MKL_INT fixedPhi;
	MKL_INT fixedPhiR;
	MKL_INT fixedPhiZ;
	MKL_INT fixedOmega;

	// -- INITIAL DATA -------------------------------------------------------
	MKL_INT readInitialData;
	char *log_alpha_i;
	char *beta_i;
	char *log_h_i;
	char *log_a_i;
	char *psi_i;
	char *lambda_i;
	char *w_i;
	MKL_INT NrTotalInitial;
	MKL_INT NzTotalInitial;
	MKL_INT ghost_i;
	MKL_INT order_i;
	double dr_i;
	double dz_i;

	// -- SCALE INITIAL DATA -------------------------------------------------
	double scale_u0;
	double scale_u1;
	double scale_u2;
	double scale_u3;
	double scale_u4;
	double scale_u5;
	double scale_u6;
	double *u_seed;

	// -- NEXT SCALE ADVANCE -------------------------------------------------
	double scale_next;

	// -- SOLVER PARAMETERS --------------------------------------------------
	MKL_INT solverType;
	MKL_INT localSolver;
	double epsilon;
	MKL_INT maxNewtonIter;
	double lambda0;
	double lambdaMin;
	MKL_INT useLowRank;

	// -- INITIAL GUESS CHECK ------------------------------------------------
	MKL_INT max_initial_guess_checks;
	double norm_f0_target;

	// -- AUXILIARY ARRAYS FOR DERIVATIVES -----------------------------------
	double *Dr_u;
	double *Dz_u;
	double *Drr_u;
	double *Dzz_u;
	double *Drz_u;

	// -- AUXILIARY VARIABLES -------------------------------------------------
	double *u_aux;
	double *Dr_u_aux;

	// -- SPHERICAL PARAMETERS FOR ANALYSIS ----------------------------------
	MKL_INT NrrTotal;
	MKL_INT NthTotal;
	MKL_INT p_dim;
	double drr;
	double dth;
	double rr_inf;

	// -- OUTPUT --------------------------------------------------------------
	char work_dirname[MAX_STR_LEN];
	char initial_dirname[MAX_STR_LEN];
	char final_dirname[MAX_STR_LEN];

	// -- SWEEP CONTROL -------------------------------------------------------
	MKL_INT sweep;
	double rr_phi_max_minimum;
	double rr_phi_max_maximum;
	MKL_INT hwl_min;
	MKL_INT hwl_max;
	double w_max;
	double w_min;
	double w_step;

	// -- ANALYSIS ------------------------------------------------------------
	double *i_rr;
	double *i_th;
	double *i_u;
	double M_KOMAR;
	double J_KOMAR;
	double GRV2;
	double GRV3;
	double phi_max;
	double rr_phi_max;
	MKL_INT hwl_res;
} rb_context;

// Set every field to its pre-parse default (the values param.h used to
// hard-code). The parser then overwrites whichever keys are present.
void rb_context_init(rb_context *ctx);

// ---------------------------------------------------------------------------
// Nonlinear-solver callback types. The context is threaded explicitly instead
// of relying on globals, so the solver cores are testable in isolation.
// ---------------------------------------------------------------------------
typedef void (*rb_rhs_fn)(rb_context *ctx, double *f, double *u);
typedef void (*rb_jacobian_fn)(rb_context *ctx, csr_matrix A, double *u, const MKL_INT print);
typedef double (*rb_norm_fn)(const rb_context *ctx, double *x);
typedef double (*rb_dot_fn)(const rb_context *ctx, double *x, double *y);
typedef void (*rb_linear_solve_fn)(double *u, csr_matrix *A, double *f);

#endif /* ROTBOSON_CONTEXT_H */
