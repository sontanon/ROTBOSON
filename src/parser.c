#include "tools.h"
#include "context.h"
#include "exit_codes.h"
#include "log.h"
#include "toml.h"

#include <stdarg.h>
#include <string.h>

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
    exit(RB_EXIT_CONFIG); // Configuration/parse errors are exit code 3.
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
    "readInitialData", "log_alpha_i", "beta_i", "log_h_i", "log_a_i", "psi_i", "lambda_i", "w_i",
    "NrTotalInitial", "NzTotalInitial", "order_i", "ghost_i", "dr_i", "dz_i",
    // Scale initial data (seed shaping; the sweep driver decides when/how much).
    "scale_u0", "scale_u1", "scale_u2", "scale_u3", "scale_u4", "scale_u5", "scale_u6",
    // Analytic initial guess.
    "psi0", "sigmaR", "sigmaZ", "rExt",
    // Initial frequency.
    "w0",
    // Solver.
    "solverType", "localSolver", "epsilon", "maxNewtonIter", "lambda0", "lambdaMin", "useLowRank",
    // Initial guess check.
    "max_initial_guess_checks", "norm_f0_target",
    // Output.
    "outputFormat", "loglevel", NULL};

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

static int lookup_string(toml_table_t *tab, const char *key, char **out)
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

void parser(rb_context *ctx, const char *fname)
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
    if (lookup_double(tab, "dr", &ctx->dr))
        check_in_range("dr", ctx->dr, MIN_DR, MAX_DR);
    else
        warn("missing \"dr\". Using default, dr = %3.5E.\n", ctx->dr);

    if (lookup_double(tab, "dz", &ctx->dz))
        check_in_range("dz", ctx->dz, MIN_DR, MAX_DR);
    else
        warn("missing \"dz\". Using default, dz = %3.5E.\n", ctx->dz);

    if (lookup_int(tab, "NrInterior", &ctx->NrInterior))
        check_int_in_range("NrInterior", ctx->NrInterior, MIN_NRINTERIOR, MAX_NRINTERIOR);
    else
        warn("missing \"NrInterior\". Using default, NrInterior = %lld.\n", ctx->NrInterior);

    if (lookup_int(tab, "NzInterior", &ctx->NzInterior))
        check_int_in_range("NzInterior", ctx->NzInterior, MIN_NRINTERIOR, MAX_NRINTERIOR);
    else
        warn("missing \"NzInterior\". Using default, NzInterior = %lld.\n", ctx->NzInterior);

    if (lookup_int(tab, "order", &ctx->order))
        check_is_int("order", ctx->order, 2, 4);
    else
        warn("missing \"order\". Using default, order = %lld.\n", ctx->order);

    // Ghost zones follow from the FD order.
    if (ctx->order == 2)
        ctx->ghost = 1;
    else
        ctx->ghost = 2;

    // Derived grid sizes.
    ctx->NrTotal = ctx->NrInterior + 2 * ctx->ghost;
    ctx->NzTotal = ctx->NzInterior + 2 * ctx->ghost;
    ctx->dim = ctx->NrTotal * ctx->NzTotal;
    ctx->w_idx = GNUM * ctx->dim;

    // -- SCALAR FIELD ------------------------------------------------------
    if (lookup_int(tab, "l", &ctx->l))
        check_int_in_range("l", ctx->l, MIN_L, MAX_L);
    else
        warn("missing \"l\". Using default, l = %lld.\n", ctx->l);

    if (lookup_double(tab, "m", &ctx->m))
        check_in_range("m", ctx->m, MIN_M, MAX_M);
    else
        warn("missing \"m\". Using default, m = %3.5E.\n", ctx->m);

    if (lookup_int(tab, "fixedPhi", &ctx->fixedPhi))
        check_is_int("fixedPhi", ctx->fixedPhi, 0, 1);

    if (lookup_int(tab, "fixedOmega", &ctx->fixedOmega))
        check_is_int("fixedOmega", ctx->fixedOmega, 0, 1);

    // Exactly one of phi or omega must be held fixed.
    if (!ctx->fixedOmega && !ctx->fixedPhi)
        die("fixedPhi = %lld and fixedOmega = %lld. One quantity must be held fixed.\n",
            ctx->fixedPhi, ctx->fixedOmega);
    if (ctx->fixedOmega && ctx->fixedPhi)
        die("fixedPhi = %lld and fixedOmega = %lld. Only one variable can be fixed.\n",
            ctx->fixedPhi, ctx->fixedOmega);

    if (ctx->fixedPhi)
    {
        if (lookup_int(tab, "fixedPhiR", &ctx->fixedPhiR))
            check_int_in_range("fixedPhiR", ctx->fixedPhiR, 1, ctx->NrInterior);
        else
            warn("missing \"fixedPhiR\". Using default, fixedPhiR = %lld.\n", ctx->fixedPhiR);

        if (lookup_int(tab, "fixedPhiZ", &ctx->fixedPhiZ))
            check_int_in_range("fixedPhiZ", ctx->fixedPhiZ, 1, ctx->NzInterior);
        else
            warn("missing \"fixedPhiZ\". Using default, fixedPhiZ = %lld.\n", ctx->fixedPhiZ);
    }

    // -- INITIAL DATA ------------------------------------------------------
    if (lookup_int(tab, "readInitialData", &ctx->readInitialData))
    {
        if (ctx->readInitialData != 0 && ctx->readInitialData != 1 && ctx->readInitialData != 2 &&
            ctx->readInitialData != 3)
            die("readInitialData = %lld is not supported. Allowed values are 0, 1, 2 or 3.\n",
                ctx->readInitialData);
    }
    else
        warn("missing \"readInitialData\". Using default, readInitialData = %lld.\n",
             ctx->readInitialData);

    switch (ctx->readInitialData)
    {
    // Interpolation from a different grid (size and/or resolution).
    case 3:
        if (!lookup_string(tab, "log_alpha_i", &ctx->log_alpha_i))
            die("readInitialData = 3 requires \"log_alpha_i\".\n");
        if (!lookup_string(tab, "beta_i", &ctx->beta_i))
            die("readInitialData = 3 requires \"beta_i\".\n");
        if (!lookup_string(tab, "log_h_i", &ctx->log_h_i))
            die("readInitialData = 3 requires \"log_h_i\".\n");
        if (!lookup_string(tab, "log_a_i", &ctx->log_a_i))
            die("readInitialData = 3 requires \"log_a_i\".\n");
        if (!lookup_string(tab, "psi_i", &ctx->psi_i))
            die("readInitialData = 3 requires \"psi_i\".\n");
        if (!lookup_string(tab, "lambda_i", &ctx->lambda_i))
            warn("readInitialData = 3 expects \"lambda_i\" (optional).\n");

        if (lookup_int(tab, "NrTotalInitial", &ctx->NrTotalInitial))
            check_int_in_range("NrTotalInitial", ctx->NrTotalInitial, MIN_NRINTERIOR,
                               MAX_NRINTERIOR);
        else
            die("readInitialData = 3 requires \"NrTotalInitial\".\n");

        if (lookup_int(tab, "NzTotalInitial", &ctx->NzTotalInitial))
            check_int_in_range("NzTotalInitial", ctx->NzTotalInitial, MIN_NRINTERIOR,
                               MAX_NRINTERIOR);
        else
            die("readInitialData = 3 requires \"NzTotalInitial\".\n");

        if (lookup_int(tab, "order_i", &ctx->order_i))
            check_is_int("order_i", ctx->order_i, 2, 4);
        else
            die("readInitialData = 3 requires \"order_i\".\n");

        if (lookup_int(tab, "ghost_i", &ctx->ghost_i))
            check_is_int("ghost_i", ctx->ghost_i, 1, 2);
        else
            die("readInitialData = 3 requires \"ghost_i\".\n");

        if (lookup_double(tab, "dr_i", &ctx->dr_i))
            check_in_range("dr_i", ctx->dr_i, MIN_DR, MAX_DR);
        else
            die("readInitialData = 3 requires \"dr_i\".\n");

        if (lookup_double(tab, "dz_i", &ctx->dz_i))
            check_in_range("dz_i", ctx->dz_i, MIN_DR, MAX_DR);
        else
            die("readInitialData = 3 requires \"dz_i\".\n");

        lookup_string(tab, "w_i", &ctx->w_i);
        break;

    // Read from file on the same grid (1) or a stated grid (2).
    case 2:
    case 1:
        lookup_string(tab, "log_alpha_i", &ctx->log_alpha_i);
        lookup_string(tab, "beta_i", &ctx->beta_i);
        lookup_string(tab, "log_h_i", &ctx->log_h_i);
        lookup_string(tab, "log_a_i", &ctx->log_a_i);
        lookup_string(tab, "psi_i", &ctx->psi_i);
        lookup_string(tab, "lambda_i", &ctx->lambda_i);
        lookup_string(tab, "w_i", &ctx->w_i);

        if (ctx->readInitialData == 2)
        {
            lookup_int(tab, "NrTotalInitial", &ctx->NrTotalInitial);
            lookup_int(tab, "NzTotalInitial", &ctx->NzTotalInitial);
        }
        else
        {
            ctx->NrTotalInitial = ctx->NrTotal;
            ctx->NzTotalInitial = ctx->NzTotal;
        }
        break;
    }

    // -- SCALE INITIAL DATA ------------------------------------------------
    lookup_double(tab, "scale_u0", &ctx->scale_u0);
    lookup_double(tab, "scale_u1", &ctx->scale_u1);
    lookup_double(tab, "scale_u2", &ctx->scale_u2);
    lookup_double(tab, "scale_u3", &ctx->scale_u3);
    lookup_double(tab, "scale_u4", &ctx->scale_u4);
    lookup_double(tab, "scale_u5", &ctx->scale_u5);
    lookup_double(tab, "scale_u6", &ctx->scale_u6);

    // -- ANALYTIC INITIAL GUESS -------------------------------------------
    if (!ctx->readInitialData)
    {
        if (lookup_double(tab, "psi0", &ctx->psi0))
            check_in_range("psi0", ctx->psi0, MIN_PSI0, MAX_PSI0);
        else
            warn("missing \"psi0\". Using default, psi0 = %3.5E.\n", ctx->psi0);

        if (lookup_double(tab, "sigmaR", &ctx->sigmaR))
            check_in_range("sigmaR", ctx->sigmaR, MIN_SIGMA, MAX_SIGMA);
        else
            warn("missing \"sigmaR\". Using default, sigmaR = %3.5E.\n", ctx->sigmaR);

        if (lookup_double(tab, "sigmaZ", &ctx->sigmaZ))
            check_in_range("sigmaZ", ctx->sigmaZ, MIN_SIGMA, MAX_SIGMA);
        else
            warn("missing \"sigmaZ\". Using default, sigmaZ = %3.5E.\n", ctx->sigmaZ);

        if (lookup_double(tab, "rExt", &ctx->rExt))
            check_in_range("rExt", ctx->rExt, MIN_R_EXT, MAX_R_EXT);
        else
            warn("missing \"rExt\". Using default, rExt = %3.5E.\n", ctx->rExt);
    }

    // -- INITIAL FREQUENCY -------------------------------------------------
    if (!ctx->w_i)
    {
        if (lookup_double(tab, "w0", &ctx->w0))
            check_in_range("w0 / m", ctx->w0 / ctx->m, MIN_W0, MAX_W0);
        else
            warn("missing \"w0\". Using default, w0 = %3.5E.\n", ctx->w0);
    }

    // -- SOLVER ------------------------------------------------------------
    if (lookup_int(tab, "solverType", &ctx->solverType))
    {
        if (ctx->solverType != 1 && ctx->solverType != 2 && ctx->solverType != 3)
            die("solverType = %lld is not supported. Allowed values are 1, 2 or 3.\n",
                ctx->solverType);
    }
    else
        warn("missing \"solverType\". Using default, solverType = %lld.\n", ctx->solverType);

    if (lookup_int(tab, "localSolver", &ctx->localSolver))
        check_is_int("localSolver", ctx->localSolver, 0, 1);
    else
        warn("missing \"localSolver\". Using default, localSolver = %lld.\n", ctx->localSolver);

    if (lookup_double(tab, "epsilon", &ctx->epsilon))
        check_in_range("epsilon", ctx->epsilon, MIN_EPS, MAX_EPS);
    else
        warn("missing \"epsilon\". Using default, epsilon = %3.5E.\n", ctx->epsilon);

    if (lookup_int(tab, "maxNewtonIter", &ctx->maxNewtonIter))
        check_int_in_range("maxNewtonIter", ctx->maxNewtonIter, MIN_MAXITER, MAX_MAXITER);
    else
        warn("missing \"maxNewtonIter\". Using default, maxNewtonIter = %lld.\n",
             ctx->maxNewtonIter);

    if (lookup_double(tab, "lambda0", &ctx->lambda0))
    {
        if (1.0 < ctx->lambda0 || ctx->lambda0 <= MIN_WEIGHT)
            die("lambda0 = %3.5E is not in range (%3.5E, 1.0].\n", ctx->lambda0, MIN_WEIGHT);
    }
    else
        warn("missing \"lambda0\". Using default, lambda0 = %3.5E.\n", ctx->lambda0);

    if (lookup_double(tab, "lambdaMin", &ctx->lambdaMin))
    {
        if (ctx->lambda0 <= ctx->lambdaMin || ctx->lambdaMin < MIN_WEIGHT)
            die("lambdaMin = %3.5E is not in range [%3.5E, lambda0).\n", ctx->lambdaMin,
                MIN_WEIGHT);
    }
    else
        warn("missing \"lambdaMin\". Using default, lambdaMin = %3.5E.\n", ctx->lambdaMin);

    if (lookup_int(tab, "useLowRank", &ctx->useLowRank))
        check_is_int("useLowRank", ctx->useLowRank, 0, 1);
    else
        warn("missing \"useLowRank\". Using default, useLowRank = %lld.\n", ctx->useLowRank);

    // -- INITIAL GUESS CHECK ----------------------------------------------
    if (lookup_int(tab, "max_initial_guess_checks", &ctx->max_initial_guess_checks))
    {
        if (ctx->max_initial_guess_checks < 0 || ctx->max_initial_guess_checks > 10)
            die("max_initial_guess_checks = %lld is out of bounds [0, 10].\n",
                ctx->max_initial_guess_checks);
    }
    else
        warn("missing \"max_initial_guess_checks\". Using default, max_initial_guess_checks = "
             "%lld.\n",
             ctx->max_initial_guess_checks);

    if (lookup_double(tab, "norm_f0_target", &ctx->norm_f0_target))
    {
        if (ctx->norm_f0_target < ctx->epsilon || ctx->norm_f0_target > 1.0)
            die("norm_f0_target = %3.5E is out of bounds [epsilon, 1.0].\n", ctx->norm_f0_target);
    }
    else
        warn("missing \"norm_f0_target\". Using default, norm_f0_target = %3.5E.\n",
             ctx->norm_f0_target);

    // -- OUTPUT ------------------------------------------------------------
    // Backend selection: "ascii" (default) or "hdf5".
    char *output_format = NULL;
    if (lookup_string(tab, "outputFormat", &output_format))
    {
        if (strcmp(output_format, "ascii") == 0)
            ctx->output_format = 0;
        else if (strcmp(output_format, "hdf5") == 0)
            ctx->output_format = 1;
        else
            die("outputFormat = \"%s\" is not supported. Allowed values are \"ascii\" or "
                "\"hdf5\".\n",
                output_format);
        free(output_format);
    }

    // Log verbosity: "error", "warn", "info" (default), "debug".
    char *loglevel = NULL;
    if (lookup_string(tab, "loglevel", &loglevel))
    {
        int lv = rb_log_level_from_string(loglevel);
        if (lv < 0)
            die("loglevel = \"%s\" is not supported. Allowed values are \"error\", \"warn\", "
                "\"info\" or \"debug\".\n",
                loglevel);
        ctx->log_level = lv;
        free(loglevel);
    }
    rb_log_set_level(ctx->log_level);

    // Set initial directory name (w is unknown until the solve completes).
    snprintf(ctx->initial_dirname, MAX_STR_LEN, "l=%lld,w=X.XXXXXE-01,dr=%.5E,N=%04lld", ctx->l,
             ctx->dr, ctx->NrInterior);

    toml_free(tab);
}
