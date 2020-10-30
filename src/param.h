#include <libconfig.h>

// String length.
#define MAX_STR_LEN 256

// Number of grid variables.
#define GNUM 6

#ifdef MAIN_FILE
/* CONFIG FILE */
config_t cfg;

/* GRID */
double dr      		= 0.0625;
double dz      		= 0.0625;
MKL_INT NrInterior 	= 128;
MKL_INT NzInterior 	= 128;
MKL_INT NrTotal		= 130;
MKL_INT NzTotal		= 130;
MKL_INT dim       	= 16900;
MKL_INT ghost     	= 1;
MKL_INT order 		= 2;

/* SCALAR FIELD PARAMETERS */
MKL_INT l		= 1;	
double m 		= 1.0;
double psi0 		= 0.2;
double sigmaR		= 4.0;
double sigmaZ		= 4.0;
double rExt		= 16.0;
double w0 		= 0.7;
MKL_INT w_idx 		= 135200;
MKL_INT fixedPhi    	= 1;
MKL_INT fixedPhiR 	= 4;
MKL_INT fixedPhiZ 	= 4;
MKL_INT fixedOmega  	= 0;

/* INITIAL DATA */
MKL_INT readInitialData = 0;
const char *log_alpha_i = NULL;
const char *beta_i 	= NULL;
const char *log_h_i 	= NULL;
const char *log_a_i 	= NULL;
const char *psi_i 	= NULL;
const char *lambda_i 	= NULL;
const char *w_i 	= NULL;
MKL_INT NrTotalInitial 	= 0;
MKL_INT NzTotalInitial 	= 0;
MKL_INT ghost_i		= 1;
MKL_INT order_i		= 2;
double dr_i		= 1.0;
double dz_i		= 1.0;

/* SCALE INITIAL DATA */
double scale_u0 = 1.0;
double scale_u1 = 1.0;
double scale_u2 = 1.0;
double scale_u3 = 1.0;
double scale_u4 = 1.0;
double scale_u5 = 1.0;
double scale_u6 = 1.0;
double *u_seed = NULL;

/* NEXT SCALE ADVANCE */
double scale_next = 1.0;

/* SOLVER PARAMETERS */
MKL_INT solverType	= 1;
MKL_INT localSolver	= 1;
double epsilon		= 1E-5;
MKL_INT maxNewtonIter 	= 10;
double lambda0 		= 1.0E-3;
double lambdaMin 	= 1.0E-8;
MKL_INT useLowRank	= 0;

/* INITIAL GUESS CHECK */
MKL_INT max_initial_guess_checks = 8;
double norm_f0_target = 1.0E-05;

/* AUXILIARY ARRAYS FOR DERIVATIVES. */
double *Dr_u;
double *Dz_u;
double *Drr_u;
double *Dzz_u;
double *Drz_u;

/* AUXILIARY VARIABLES. */
double *u_aux;
double *Dr_u_aux;

/* SPHERICAL PARAMETERS FOR ANALYSIS */
MKL_INT NrrTotal;
MKL_INT NthTotal;
MKL_INT p_dim;
double drr;
double dth;
double rr_inf;

/* OUTPUT */
char    work_dirname[MAX_STR_LEN] = { 0 };
char initial_dirname[MAX_STR_LEN] = { 0 };
char   final_dirname[MAX_STR_LEN] = { 0 };

/* SWEEP CONTROL */
double rr_phi_max_minimum = 1.0;
#else
/* CONFIG FILE */
extern config_t cfg;

/* GRID */
extern double dr;
extern double dz;
extern MKL_INT NrInterior;
extern MKL_INT NzInterior;
extern MKL_INT NrTotal;
extern MKL_INT NzTotal;
extern MKL_INT dim;
extern MKL_INT ghost;
extern MKL_INT order;

/* SCALAR FIELD PARAMETERS */
extern MKL_INT l;
extern double m;
extern double psi0;
extern double sigmaR;
extern double sigmaZ;
extern double rExt;
extern double w0;
extern MKL_INT w_idx;
extern MKL_INT fixedPhi;
extern MKL_INT fixedPhiR;
extern MKL_INT fixedPhiZ;
extern MKL_INT fixedOmega;

/* INITIAL DATA */
extern MKL_INT readInitialData;
extern const char *log_alpha_i;
extern const char *beta_i;
extern const char *log_h_i;
extern const char *log_a_i;
extern const char *psi_i;
extern const char *lambda_i;
extern const char *w_i;
extern MKL_INT NrTotalInitial;
extern MKL_INT NzTotalInitial;
extern MKL_INT ghost_i;
extern MKL_INT order_i;
extern double dr_i;
extern double dz_i;

/* SCALE INITIAL DATA */
extern double scale_u0;
extern double scale_u1;
extern double scale_u2;
extern double scale_u3;
extern double scale_u4;
extern double scale_u5;
extern double scale_u6;
extern double *u_seed;

/* NEXT SCALE ADVANCE */
extern double scale_next;

/* SOLVER PARAMETERS */
extern MKL_INT solverType;
extern MKL_INT localSolver; 
extern double epsilon; 
extern MKL_INT maxNewtonIter;
extern double lambda0;
extern double lambdaMin;
extern MKL_INT useLowRank;

/* AUXILIARY ARRAYS FOR DERIVATIVES. */
extern double *Dr_u;
extern double *Dz_u;
extern double *Drr_u;
extern double *Dzz_u;
extern double *Drz_u;

/* INITIAL GUESS CHECK */
extern MKL_INT max_initial_guess_checks;
extern double norm_f0_target;

/* AUXILIARY VARIABLES. */
extern double *u_aux;
extern double *Dr_u_aux;

/* SPHERICAL PARAMETERS FOR ANALYSIS */
extern MKL_INT NrrTotal;
extern MKL_INT NthTotal;
extern MKL_INT p_dim;
extern double drr;
extern double dth;
extern double rr_inf;

/* OUTPUT */
extern char    work_dirname[MAX_STR_LEN];
extern char   final_dirname[MAX_STR_LEN];
extern char initial_dirname[MAX_STR_LEN];

/* SWEEP CONTROL */
extern double rr_phi_max_minimum;
#endif
