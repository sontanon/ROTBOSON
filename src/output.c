// solution_writer dispatcher + ASCII backend.
//
// The ASCII backend is a verbatim, path-aware re-homing of the legacy
// `write_single_file_*` writers that used to live in tools.c. Format strings,
// separators and row/column loops are unchanged so the output is byte-identical
// to the pre-Phase-5 `.asc` files.
#include "output_internal.h"
#include "exit_codes.h"

#include "log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// ---------------------------------------------------------------------------
// Path building
// ---------------------------------------------------------------------------

void solution_writer_field_path(const solution_writer *w, const char *name, char *buf, size_t n)
{
    snprintf(buf, n, "%s/%s.asc", w->dirname, name);
}

// ---------------------------------------------------------------------------
// ASCII backend
// ---------------------------------------------------------------------------

static void ascii_write_1d(solution_writer *w, const char *name, const double *u, MKL_INT dim)
{
    char path[RB_PATH_MAX];
    solution_writer_field_path(w, name, path, sizeof(path));

    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        rb_log(RB_LOG_ERROR, "OUTPUT: cannot open \"%s\" for writing.\n", path);
        exit(RB_EXIT_IO);
    }

    for (MKL_INT i = 0; i < dim; ++i)
        fprintf(fp, "%9.18E\n", u[i]);

    fclose(fp);
}

static void ascii_write_int_1d(solution_writer *w, const char *name, const MKL_INT *u, MKL_INT dim)
{
    char path[RB_PATH_MAX];
    solution_writer_field_path(w, name, path, sizeof(path));

    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        rb_log(RB_LOG_ERROR, "OUTPUT: cannot open \"%s\" for writing.\n", path);
        exit(RB_EXIT_IO);
    }

    for (MKL_INT i = 0; i < dim; ++i)
        fprintf(fp, "%lld\n", (long long)u[i]);

    fclose(fp);
}

static void ascii_write_2d(solution_writer *w, const char *name, const double *u, MKL_INT nr,
                           MKL_INT nz)
{
    char path[RB_PATH_MAX];
    solution_writer_field_path(w, name, path, sizeof(path));

    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        rb_log(RB_LOG_ERROR, "OUTPUT: cannot open \"%s\" for writing.\n", path);
        exit(RB_EXIT_IO);
    }

    for (MKL_INT i = 0; i < nr; ++i)
    {
        for (MKL_INT j = 0; j < nz; ++j)
        {
            fprintf(fp, (j < nz - 1) ? "%9.18E\t" : "%9.18E\n", u[i * nz + j]);
        }
    }

    fclose(fp);
}

static void ascii_write_2d_polar(solution_writer *w, const char *name, const double *u, MKL_INT nr,
                                 MKL_INT nth)
{
    char path[RB_PATH_MAX];
    solution_writer_field_path(w, name, path, sizeof(path));

    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        rb_log(RB_LOG_ERROR, "OUTPUT: cannot open \"%s\" for writing.\n", path);
        exit(RB_EXIT_IO);
    }

    for (MKL_INT i = 0; i < nr; ++i)
    {
        for (MKL_INT j = 0; j < nth; ++j)
        {
            fprintf(fp, (j < nth - 1) ? "%9.18E\t" : "%9.18E\n", u[i * nth + j]);
        }
    }

    fclose(fp);
}

int ascii_backend_init(solution_writer *w)
{
    w->write_1d = ascii_write_1d;
    w->write_int_1d = ascii_write_int_1d;
    w->write_2d = ascii_write_2d;
    w->write_2d_polar = ascii_write_2d_polar;
    w->close_backend = NULL; // nothing to flush/close
    w->backend = NULL;
    return 0;
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

solution_writer *solution_writer_open(const char *dirname, const rb_context *ctx,
                                      const char *parfile)
{
    // Create the output directory if it does not exist (single level, as the
    // legacy io() did). An existing directory is reused without prompting.
    struct stat st;
    if (stat(dirname, &st) == -1)
    {
        if (mkdir(dirname, 0755) == -1)
        {
            rb_log(RB_LOG_ERROR, "OUTPUT: cannot create output directory \"%s\".\n", dirname);
            return NULL;
        }
    }

    solution_writer *w = (solution_writer *)malloc(sizeof(*w));
    if (!w)
        return NULL;

    memset(w, 0, sizeof(*w));
    snprintf(w->dirname, sizeof(w->dirname), "%s", dirname);
    snprintf(w->parfile, sizeof(w->parfile), "%s", parfile ? parfile : "");
    w->ctx = ctx;
    w->closed = 0;
    w->format = (ctx && ctx->output_format == RB_OUTPUT_HDF5) ? OUTPUT_FORMAT_HDF5
                                                              : OUTPUT_FORMAT_ASCII;

    // Select the backend from the context's output_format field.
    int rc = 0;
    if (w->format == OUTPUT_FORMAT_HDF5)
        rc = hdf5_backend_init(w);
    else
        rc = ascii_backend_init(w);

    if (rc != 0)
    {
        free(w);
        return NULL;
    }

    return w;
}

void solution_writer_close(solution_writer *w)
{
    if (!w || w->closed)
        return;

    if (w->close_backend)
        w->close_backend(w);

    w->closed = 1;
    free(w);
}

// ---------------------------------------------------------------------------
// Field writers (dispatch to the active backend)
// ---------------------------------------------------------------------------

void solution_writer_write_1d(solution_writer *w, const char *name, const double *u, MKL_INT dim)
{
    w->write_1d(w, name, u, dim);
}

void solution_writer_write_int_1d(solution_writer *w, const char *name, const MKL_INT *u,
                                  MKL_INT dim)
{
    w->write_int_1d(w, name, u, dim);
}

void solution_writer_write_2d(solution_writer *w, const char *name, const double *u, MKL_INT nr,
                              MKL_INT nz)
{
    w->write_2d(w, name, u, nr, nz);
}

void solution_writer_write_2d_polar(solution_writer *w, const char *name, const double *u,
                                    MKL_INT nr, MKL_INT nth)
{
    w->write_2d_polar(w, name, u, nr, nth);
}
