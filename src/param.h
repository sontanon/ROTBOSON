#include <libconfig.h>

#ifdef MAIN_FILE
/* CONFIG FILE */
config_t cfg;

/* GRID */
double dr      	= 0.0625;
double dz      	= 0.0625;
int NrInterior 	= 128;
int NzInterior 	= 128;
int NrTotal	= 130;
int NzTotal	= 130;
int dim        	= 16900;
int ghost      	= 1;
int order 	= 2;

/* SCALAR FIELD PARAMETERS */
int l		= 1;	
double m 	= 1.0;
double psi0 	= 0.2;
double sigmaR	= 4.0;
double sigmaZ	= 4.0;
double rExt	= 16.0;
double w0 	= 0.7;
int fixedPhiR 	= 1;
int fixedPhiZ 	= 1;

/* INITIAL DATA */
int readInitialData = 0;
const char *log_alpha_i = NULL;
const char *beta_i = NULL;
const char *log_h_i = NULL;
const char *log_a_i = NULL;
const char *psi_i = NULL;
const char *w_i = NULL;
int NrTotalInitial = 0;
int NzTotalInitial = 0;

/* SOLVER PARAMETERS */
int solverType		= 1;
int localSolver		= 1;
double epsilon		= 1E-5;
int maxNewtonIter 	= 100;
double lambda0 		= 1.0E-3;
double lambdaMin 	= 1.0E-10;
int useLowRank		= 0;

/* BOUNDARY TYPES */
int alphaBoundOrder	= 2;
int betaBoundOrder	= 2;
int hBoundOrder		= 2;
int aBoundOrder		= 2;
int phiBoundOrder	= 2;

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
extern int NrInterior;
extern int NzInterior;
extern int NrTotal;
extern int NzTotal;
extern int dim;
extern int ghost;
extern int order;

/* SCALAR FIELD PARAMETERS */
extern int l;
extern double m;
extern double psi0;
extern double sigmaR;
extern double sigmaZ;
extern double rExt;
extern double w0;
extern int fixedPhiR;
extern int fixedPhiZ;

/* INITIAL DATA */
extern int readInitialData;
extern const char *log_alpha_i;
extern const char *beta_i;
extern const char *log_h_i;
extern const char *log_a_i;
extern const char *psi_i;
extern const char *w_i;
extern int NrTotalInitial;
extern int NzTotalInitial;

/* SOLVER PARAMETERS */
extern int solverType;
extern int localSolver; 
extern double epsilon; 
extern int maxNewtonIter;
extern double lambda0;
extern double lambdaMin;
extern int useLowRank;

/* AUXILIARY ARRAYS FOR DERIVATIVES. */
extern double *Dr_u;
extern double *Dz_u;
extern double *Drr_u;
extern double *Dzz_u;
extern double *Drz_u;

/* BOUNDARY TYPES */
extern int alphaBoundOrder;
extern int betaBoundOrder;
extern int hBoundOrder;
extern int aBoundOrder;
extern int phiBoundOrder;

/* OUTPUT */
extern const char *dirname;
#endif
