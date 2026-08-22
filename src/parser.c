#include "tools.h"
#include "param.h"
#include "toml.h"

#include <stdarg.h>

// Parameter range bounds. Kept at the top so they are easy to audit and extend.
#define MAX_DR 1.0
#define MIN_DR 0.001

#define MAX_NRINTERIOR 9999999LL
#define MIN_NRINTERIOR 16LL

#define MAX_L 6LL
#define MIN_L 1LL

#define MAX_M 1.0E+3
#define MIN_M 1.0E-3

#define MAX_PSI0 1.0E+5
#define MIN_PSI0 1.0E-25

#define MAX_SIGMA 1.0E+3
#define MIN_SIGMA 1.0E-3

#define MAX_R_EXT 1.0E+3
#define MIN_R_EXT 1.0E-3

#define MAX_W0 1.0
#define MIN_W0 0.0

#define MAX_MAXITER 100000LL
#define MIN_MAXITER 0LL

#define MIN_WEIGHT 1.0E-26

#define MAX_EPS 1.0E-1
#define MIN_EPS 1.0E-16

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

static void die(const char *fmt, ...)
{
	va_list ap;
	fprintf(stderr, "PARSER: ERROR! ");
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	exit(EXIT_FAILURE);
}

static void warn(const char *fmt, ...)
{
	va_list ap;
	fprintf(stderr, "PARSER: WARNING! ");
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}

// ---------------------------------------------------------------------------
// Strict key validation: every key in the file must be known.
// ---------------------------------------------------------------------------

static const char *const KNOWN_KEYS[] = {
	// Grid.
	"dr", "dz", "NrInterior", "NzInterior", "order",
	// Scalar field.
	"l", "m", "fixedPhi", "fixedPhiR", "fixedPhiZ", "fixedOmega",
	// Initial data (file paths + grid).
	"readInitialData", "log_alpha_i", "beta_i", "log_h_i", "log_a_i",
	"psi_i", "lambda_i", "w_i",
	"NrTotalInitial", "NzTotalInitial", "order_i", "ghost_i", "dr_i", "dz_i",
	// Scale initial data.
	"scale_u0", "scale_u1", "scale_u2", "scale_u3", "scale_u4", "scale_u5", "scale_u6",
	// Analytic initial guess.
	"psi0", "sigmaR", "sigmaZ", "rExt",
	// Initial frequency.
	"w0",
	// Solver.
	"solverType", "localSolver", "epsilon", "maxNewtonIter",
	"lambda0", "lambdaMin", "useLowRank",
	// Initial guess check.
	"max_initial_guess_checks", "norm_f0_target",
	// Sweep control.
	"rr_phi_max_minimum", "rr_phi_max_maximum", "sweep",
	"hwl_min", "hwl_max", "w_max", "w_min", "w_step",
	// Next-scale advancement.
	"scale_next",
	NULL
};

static int is_known_key(const char *key)
{
	for (int i = 0; KNOWN_KEYS[i] != NULL; i++)
	{
		if (strcmp(KNOWN_KEYS[i], key) == 0)
			return 1;
	}
	return 0;
}

static void reject_unknown_keys(toml_table_t *tab)
{
	// The parameter format is a single flat table; nested tables/arrays are
	// not part of it and are rejected outright.
	if (toml_table_ntab(tab) != 0 || toml_table_narr(tab) != 0)
		die("nested tables/arrays are not supported in the flat parameter format.\n");

	int n = toml_table_nkval(tab);
	for (int i = 0; i < n; i++)
	{
		const char *key = toml_key_in(tab, i);
		if (!is_known_key(key))
			die("unknown parameter key \"%s\".\n", key);
	}
}

// ---------------------------------------------------------------------------
// Typed lookups. Return 1 when the key is present and stored, 0 when absent.
// A present key with the wrong TOML type is a hard error (not silently
// ignored, as libconfig did).
// ---------------------------------------------------------------------------

static int lookup_int(toml_table_t *tab, const char *key, MKL_INT *out)
{
	if (!toml_key_exists(tab, key))
		return 0;
	toml_datum_t d = toml_int_in(tab, key);
	if (!d.ok)
		die("\"%s\" must be an integer.\n", key);
	*out = (MKL_INT)d.u.i;
	return 1;
}

static int lookup_double(toml_table_t *tab, const char *key, double *out)
{
	if (!toml_key_exists(tab, key))
		return 0;
	toml_datum_t d = toml_double_in(tab, key);
	if (d.ok)
	{
		*out = d.u.d;
		return 1;
	}
	// Accept integer literals where a float is expected.
	d = toml_int_in(tab, key);
	if (d.ok)
	{
		*out = (double)d.u.i;
		return 1;
	}
	die("\"%s\" must be a number.\n", key);
	return 0;
}

static int lookup_string(toml_table_t *tab, const char *key, const char **out)
{
	if (!toml_key_exists(tab, key))
		return 0;
	toml_datum_t d = toml_string_in(tab, key);
	if (!d.ok)
		die("\"%s\" must be a string.\n", key);
	*out = strdup(d.u.s);
	free(d.u.s);
	return 1;
}

// ---------------------------------------------------------------------------
// Range validation.
// ---------------------------------------------------------------------------

static void check_in_range(const char *key, double v, double lo, double hi)
{
	if (v < lo || v > hi)
		die("%s = %3.5E is not in range [%3.5E, %3.5E].\n", key, v, lo, hi);
}

static void check_int_in_range(const char *key, MKL_INT v, MKL_INT lo, MKL_INT hi)
{
	if (v < lo || v > hi)
		die("%s = %lld is not in range [%lld, %lld].\n", key, v, lo, hi);
}

static void check_is_int(const char *key, MKL_INT v, MKL_INT a, MKL_INT b)
{
	if (v != a && v != b)
		die("%s = %lld is not supported. Allowed values are %lld or %lld.\n", key, v, a, b);
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

void parser(const char *fname)
{
	FILE *fp = fopen(fname, "r");
	if (!fp)
		die("could not open parameter file \"%s\".\n", fname);

	char errbuf[256];
	toml_table_t *tab = toml_parse_file(fp, errbuf, sizeof(errbuf));
	fclose(fp);
	if (!tab)
		die("could not parse \"%s\":\n%s\n", fname, errbuf);

	reject_unknown_keys(tab);

	// -- GRID --------------------------------------------------------------
	if (lookup_double(tab, "dr", &dr))
		check_in_range("dr", dr, MIN_DR, MAX_DR);
	else
		warn("missing \"dr\". Using default, dr = %3.5E.\n", dr);

	if (lookup_double(tab, "dz", &dz))
		check_in_range("dz", dz, MIN_DR, MAX_DR);
	else
		warn("missing \"dz\". Using default, dz = %3.5E.\n", dz);

	if (lookup_int(tab, "NrInterior", &NrInterior))
		check_int_in_range("NrInterior", NrInterior, MIN_NRINTERIOR, MAX_NRINTERIOR);
	else
		warn("missing \"NrInterior\". Using default, NrInterior = %lld.\n", NrInterior);

	if (lookup_int(tab, "NzInterior", &NzInterior))
		check_int_in_range("NzInterior", NzInterior, MIN_NRINTERIOR, MAX_NRINTERIOR);
	else
		warn("missing \"NzInterior\". Using default, NzInterior = %lld.\n", NzInterior);

	if (lookup_int(tab, "order", &order))
		check_is_int("order", order, 2, 4);
	else
		warn("missing \"order\". Using default, order = %lld.\n", order);

	// Ghost zones follow from the FD order.
	if (order == 2)
		ghost = 1;
	else
		ghost = 2;

	// Derived grid sizes.
	NrTotal = NrInterior + 2 * ghost;
	NzTotal = NzInterior + 2 * ghost;
	dim = NrTotal * NzTotal;
	w_idx = GNUM * dim;

	// -- SCALAR FIELD ------------------------------------------------------
	if (lookup_int(tab, "l", &l))
		check_int_in_range("l", l, MIN_L, MAX_L);
	else
		warn("missing \"l\". Using default, l = %lld.\n", l);

	if (lookup_double(tab, "m", &m))
		check_in_range("m", m, MIN_M, MAX_M);
	else
		warn("missing \"m\". Using default, m = %3.5E.\n", m);

	if (lookup_int(tab, "fixedPhi", &fixedPhi))
		check_is_int("fixedPhi", fixedPhi, 0, 1);

	if (lookup_int(tab, "fixedOmega", &fixedOmega))
		check_is_int("fixedOmega", fixedOmega, 0, 1);

	// Exactly one of phi or omega must be held fixed.
	if (!fixedOmega && !fixedPhi)
		die("fixedPhi = %lld and fixedOmega = %lld. One quantity must be held fixed.\n", fixedPhi, fixedOmega);
	if (fixedOmega && fixedPhi)
		die("fixedPhi = %lld and fixedOmega = %lld. Only one variable can be fixed.\n", fixedPhi, fixedOmega);

	if (fixedPhi)
	{
		if (lookup_int(tab, "fixedPhiR", &fixedPhiR))
			check_int_in_range("fixedPhiR", fixedPhiR, 1, NrInterior);
		else
			warn("missing \"fixedPhiR\". Using default, fixedPhiR = %lld.\n", fixedPhiR);

		if (lookup_int(tab, "fixedPhiZ", &fixedPhiZ))
			check_int_in_range("fixedPhiZ", fixedPhiZ, 1, NzInterior);
		else
			warn("missing \"fixedPhiZ\". Using default, fixedPhiZ = %lld.\n", fixedPhiZ);
	}

	// -- INITIAL DATA ------------------------------------------------------
	if (lookup_int(tab, "readInitialData", &readInitialData))
	{
		if (readInitialData != 0 && readInitialData != 1 && readInitialData != 2 && readInitialData != 3)
			die("readInitialData = %lld is not supported. Allowed values are 0, 1, 2 or 3.\n", readInitialData);
	}
	else
		warn("missing \"readInitialData\". Using default, readInitialData = %lld.\n", readInitialData);

	switch (readInitialData)
	{
	// Interpolation from a different grid (size and/or resolution).
	case 3:
		if (!lookup_string(tab, "log_alpha_i", &log_alpha_i))
			die("readInitialData = 3 requires \"log_alpha_i\".\n");
		if (!lookup_string(tab, "beta_i", &beta_i))
			die("readInitialData = 3 requires \"beta_i\".\n");
		if (!lookup_string(tab, "log_h_i", &log_h_i))
			die("readInitialData = 3 requires \"log_h_i\".\n");
		if (!lookup_string(tab, "log_a_i", &log_a_i))
			die("readInitialData = 3 requires \"log_a_i\".\n");
		if (!lookup_string(tab, "psi_i", &psi_i))
			die("readInitialData = 3 requires \"psi_i\".\n");
		if (!lookup_string(tab, "lambda_i", &lambda_i))
			warn("readInitialData = 3 expects \"lambda_i\" (optional).\n");

		if (lookup_int(tab, "NrTotalInitial", &NrTotalInitial))
			check_int_in_range("NrTotalInitial", NrTotalInitial, MIN_NRINTERIOR, MAX_NRINTERIOR);
		else
			die("readInitialData = 3 requires \"NrTotalInitial\".\n");

		if (lookup_int(tab, "NzTotalInitial", &NzTotalInitial))
			check_int_in_range("NzTotalInitial", NzTotalInitial, MIN_NRINTERIOR, MAX_NRINTERIOR);
		else
			die("readInitialData = 3 requires \"NzTotalInitial\".\n");

		if (lookup_int(tab, "order_i", &order_i))
			check_is_int("order_i", order_i, 2, 4);
		else
			die("readInitialData = 3 requires \"order_i\".\n");

		if (lookup_int(tab, "ghost_i", &ghost_i))
			check_is_int("ghost_i", ghost_i, 1, 2);
		else
			die("readInitialData = 3 requires \"ghost_i\".\n");

		if (lookup_double(tab, "dr_i", &dr_i))
			check_in_range("dr_i", dr_i, MIN_DR, MAX_DR);
		else
			die("readInitialData = 3 requires \"dr_i\".\n");

		if (lookup_double(tab, "dz_i", &dz_i))
			check_in_range("dz_i", dz_i, MIN_DR, MAX_DR);
		else
			die("readInitialData = 3 requires \"dz_i\".\n");

		lookup_string(tab, "w_i", &w_i);
		break;

	// Read from file on the same grid (1) or a stated grid (2).
	case 2:
	case 1:
		lookup_string(tab, "log_alpha_i", &log_alpha_i);
		lookup_string(tab, "beta_i", &beta_i);
		lookup_string(tab, "log_h_i", &log_h_i);
		lookup_string(tab, "log_a_i", &log_a_i);
		lookup_string(tab, "psi_i", &psi_i);
		lookup_string(tab, "lambda_i", &lambda_i);
		lookup_string(tab, "w_i", &w_i);

		if (readInitialData == 2)
		{
			lookup_int(tab, "NrTotalInitial", &NrTotalInitial);
			lookup_int(tab, "NzTotalInitial", &NzTotalInitial);
		}
		else
		{
			NrTotalInitial = NrTotal;
			NzTotalInitial = NzTotal;
		}
		break;
	}

	// -- SCALE INITIAL DATA ------------------------------------------------
	lookup_double(tab, "scale_u0", &scale_u0);
	lookup_double(tab, "scale_u1", &scale_u1);
	lookup_double(tab, "scale_u2", &scale_u2);
	lookup_double(tab, "scale_u3", &scale_u3);
	lookup_double(tab, "scale_u4", &scale_u4);
	lookup_double(tab, "scale_u5", &scale_u5);
	lookup_double(tab, "scale_u6", &scale_u6);

	// -- ANALYTIC INITIAL GUESS -------------------------------------------
	if (!readInitialData)
	{
		if (lookup_double(tab, "psi0", &psi0))
			check_in_range("psi0", psi0, MIN_PSI0, MAX_PSI0);
		else
			warn("missing \"psi0\". Using default, psi0 = %3.5E.\n", psi0);

		if (lookup_double(tab, "sigmaR", &sigmaR))
			check_in_range("sigmaR", sigmaR, MIN_SIGMA, MAX_SIGMA);
		else
			warn("missing \"sigmaR\". Using default, sigmaR = %3.5E.\n", sigmaR);

		if (lookup_double(tab, "sigmaZ", &sigmaZ))
			check_in_range("sigmaZ", sigmaZ, MIN_SIGMA, MAX_SIGMA);
		else
			warn("missing \"sigmaZ\". Using default, sigmaZ = %3.5E.\n", sigmaZ);

		if (lookup_double(tab, "rExt", &rExt))
			check_in_range("rExt", rExt, MIN_R_EXT, MAX_R_EXT);
		else
			warn("missing \"rExt\". Using default, rExt = %3.5E.\n", rExt);
	}

	// -- INITIAL FREQUENCY -------------------------------------------------
	if (!w_i)
	{
		if (lookup_double(tab, "w0", &w0))
			check_in_range("w0 / m", w0 / m, MIN_W0, MAX_W0);
		else
			warn("missing \"w0\". Using default, w0 = %3.5E.\n", w0);
	}

	// -- SOLVER ------------------------------------------------------------
	if (lookup_int(tab, "solverType", &solverType))
	{
		if (solverType != 1 && solverType != 2 && solverType != 3)
			die("solverType = %lld is not supported. Allowed values are 1, 2 or 3.\n", solverType);
	}
	else
		warn("missing \"solverType\". Using default, solverType = %lld.\n", solverType);

	if (lookup_int(tab, "localSolver", &localSolver))
		check_is_int("localSolver", localSolver, 0, 1);
	else
		warn("missing \"localSolver\". Using default, localSolver = %lld.\n", localSolver);

	if (lookup_double(tab, "epsilon", &epsilon))
		check_in_range("epsilon", epsilon, MIN_EPS, MAX_EPS);
	else
		warn("missing \"epsilon\". Using default, epsilon = %3.5E.\n", epsilon);

	if (lookup_int(tab, "maxNewtonIter", &maxNewtonIter))
		check_int_in_range("maxNewtonIter", maxNewtonIter, MIN_MAXITER, MAX_MAXITER);
	else
		warn("missing \"maxNewtonIter\". Using default, maxNewtonIter = %lld.\n", maxNewtonIter);

	if (lookup_double(tab, "lambda0", &lambda0))
	{
		if (1.0 < lambda0 || lambda0 <= MIN_WEIGHT)
			die("lambda0 = %3.5E is not in range (%3.5E, 1.0].\n", lambda0, MIN_WEIGHT);
	}
	else
		warn("missing \"lambda0\". Using default, lambda0 = %3.5E.\n", lambda0);

	if (lookup_double(tab, "lambdaMin", &lambdaMin))
	{
		if (lambda0 <= lambdaMin || lambdaMin < MIN_WEIGHT)
			die("lambdaMin = %3.5E is not in range [%3.5E, lambda0).\n", lambdaMin, MIN_WEIGHT);
	}
	else
		warn("missing \"lambdaMin\". Using default, lambdaMin = %3.5E.\n", lambdaMin);

	if (lookup_int(tab, "useLowRank", &useLowRank))
		check_is_int("useLowRank", useLowRank, 0, 1);
	else
		warn("missing \"useLowRank\". Using default, useLowRank = %lld.\n", useLowRank);

	// -- INITIAL GUESS CHECK ----------------------------------------------
	if (lookup_int(tab, "max_initial_guess_checks", &max_initial_guess_checks))
	{
		if (max_initial_guess_checks < 0 || max_initial_guess_checks > 10)
			die("max_initial_guess_checks = %lld is out of bounds [0, 10].\n", max_initial_guess_checks);
	}
	else
		warn("missing \"max_initial_guess_checks\". Using default, max_initial_guess_checks = %lld.\n", max_initial_guess_checks);

	if (lookup_double(tab, "norm_f0_target", &norm_f0_target))
	{
		if (norm_f0_target < epsilon || norm_f0_target > 1.0)
			die("norm_f0_target = %3.5E is out of bounds [epsilon, 1.0].\n", norm_f0_target);
	}
	else
		warn("missing \"norm_f0_target\". Using default, norm_f0_target = %3.5E.\n", norm_f0_target);

	// -- SWEEP CONTROL -----------------------------------------------------
	if (lookup_double(tab, "rr_phi_max_minimum", &rr_phi_max_minimum))
	{
		if (rr_phi_max_minimum < 4 * dr || rr_phi_max_minimum > dr * NrInterior)
			die("rr_phi_max_minimum = %3.5E is out of bounds [4*dr, dr*NrInterior].\n", rr_phi_max_minimum);
	}
	else
		warn("missing \"rr_phi_max_minimum\". Using default, rr_phi_max_minimum = %3.5E.\n", rr_phi_max_minimum);

	if (lookup_double(tab, "rr_phi_max_maximum", &rr_phi_max_maximum))
	{
		if (rr_phi_max_maximum < rr_phi_max_minimum || rr_phi_max_maximum > dr * NrTotal)
			die("rr_phi_max_maximum = %3.5E is out of bounds [rr_phi_max_minimum, dr*NrTotal].\n", rr_phi_max_maximum);
	}
	else
		warn("missing \"rr_phi_max_maximum\". Using default, rr_phi_max_maximum = %3.5E.\n", rr_phi_max_maximum);

	lookup_int(tab, "sweep", &sweep);
	lookup_int(tab, "hwl_min", &hwl_min);
	lookup_int(tab, "hwl_max", &hwl_max);
	lookup_double(tab, "w_max", &w_max);
	lookup_double(tab, "w_min", &w_min);
	lookup_double(tab, "w_step", &w_step);

	// -- NEXT SCALE ADVANCEMENT -------------------------------------------
	lookup_double(tab, "scale_next", &scale_next);

	// -- OUTPUT ------------------------------------------------------------
	getcwd(work_dirname, MAX_STR_LEN);

	// Set initial directory name (w is unknown until the solve completes).
	snprintf(initial_dirname, MAX_STR_LEN, "l=%lld,w=X.XXXXXE-01,dr=%.5E,N=%04lld", l, dr, NrInterior);

	toml_free(tab);
}
