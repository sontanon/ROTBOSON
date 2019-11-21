#include <libconfig.h>

#ifdef MAIN_FILE
/* CONFIG FILE */
config_t cfg;

/* GRID */
double dr      	= 0.0625;
double dz      	= 0.0625;
MKL_INT NrInterior 	= 128;
MKL_INT NzInterior 	= 128
MKL_INT NrTotal	= 130;
MKL_INT NzTotal	= 130;
MKL_INT dim       	= 16900;
MKL_INT ghost     	= 1;
MKL_INT order 	= 2;

/* SCALAR FIELD PARAMETERS */
MKL_INT l		= 1;	
double m 	= 1.0;
double psi0 	= 0.2;
double sigmaR	= 4.0;
double sigmaZ	= 4.0;
double rExt	= 16.0;
double w0 	= 0.7;
MKL_INT fixedPhiR 	= 1;
MKL_INT fixedPhiZ 	= 1;

/* INITIAL DATA */
MKL_INT readInitialData = 0;
const char *log_alpha_i = NULL;
const char *beta_i = NULL;
const char *log_h_i = NULL;
const char *log_a_i = NULL;
const char *psi_i = NULL;
const char *w_i = NULL;
MKL_INT NrTotalInitial = 0;
MKL_INT NzTotalInitial = 0;

/* SOLVER PARAMETERS */
MKL_INT solverType		= 1;
MKL_INT localSolver		= 1;
double epsilon		= 1E-5;
MKL_INT maxNewtonIter 	= 100;
double lambda0 		= 1.0E-3;
double lambdaMin 	= 1.0E-10;
MKL_INT useLowRank		= 0;

/* BOUNDARY TYPES */
MKL_INT alphaBoundOrder	= 2;
MKL_INT betaBoundOrder	= 2;
MKL_INT hBoundOrder		= 2;
MKL_INT aBoundOrder		= 2;
MKL_INT phiBoundOrder	= 2;

/* AUXILIARY ARRAYS FOR DERIVATIVES. */
double *Dr_u;
double *Dz_u;
double *Drr_u;
double *Dzz_u;
double *Drz_u;

/* OUTPUT */
const char *dirname = "test";
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
extern MKL_INT fixedPhiR;
extern MKL_INT fixedPhiZ;

/* INITIAL DATA */
extern MKL_INT readInitialData;
extern const char *log_alpha_i;
extern const char *beta_i;
extern const char *log_h_i;
extern const char *log_a_i;
extern const char *psi_i;
extern const char *w_i;
extern MKL_INT NrTotalInitial;
extern MKL_INT NzTotalInitial;

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

/* BOUNDARY TYPES */
extern MKL_INT alphaBoundOrder;
extern MKL_INT betaBoundOrder;
extern MKL_INT hBoundOrder;
extern MKL_INT aBoundOrder;
extern MKL_INT phiBoundOrder;

/* OUTPUT */
extern const char *dirname;
#endif
