// ROTBOSON solution output.
//
// A `solution_writer` abstracts where a solution's fields, grids and scalar
// observables are written. Two backends implement it:
//
//   - ASCII  (default): one file per field, "<dirname>/<name>.asc", in the
//     legacy `%9.18E` tab-separated format, byte-identical to the
//     pre-Phase-5 output (compatibility with the historical catalogue).
//   - HDF5:   a single self-describing file "<dirname>/solution.h5". Datasets
//     are named "<name>.asc" (1:1 with the ASCII files) so the two formats
//     round-trip through the same Python readers. Parameters, solver settings
//     and the build git hash are stored as file attributes.
//
// The writers are path-aware: callers pass a bare field name (no ".asc", no
// path) and the backend maps it into its own store. There is no global cwd
// juggling (the old io() chdir'd the process; that design is gone).
#ifndef ROTBOSON_OUTPUT_H
#define ROTBOSON_OUTPUT_H

#include "tools.h"    // MKL_INT
#include "context.h"  // rb_context

// Output backend selectors (stored in rb_context.output_format).
#define RB_OUTPUT_ASCII 0
#define RB_OUTPUT_HDF5 1

typedef enum
{
    OUTPUT_FORMAT_ASCII = RB_OUTPUT_ASCII,
    OUTPUT_FORMAT_HDF5 = RB_OUTPUT_HDF5
} output_format;

// Max length of a built field path ("<dirname>/<name>.asc").
#define RB_PATH_MAX 1024

typedef struct solution_writer solution_writer;

// Open a writer rooted at `dirname`. The directory is created if missing
// (single level, like the legacy io()); an existing directory is reused
// without an interactive prompt. `ctx` supplies the parameters/solver
// settings recorded as HDF5 attributes; `parfile` is the path of the
// originating parameter file (metadata only). Returns NULL on failure.
solution_writer *solution_writer_open(const char *dirname, const rb_context *ctx,
                                      const char *parfile);

// Flush metadata and close the underlying store. Idempotent; NULL-safe.
void solution_writer_close(solution_writer *w);

// Field writers. `name` is the bare field name. The ASCII backend writes
// "<dirname>/<name>.asc"; the HDF5 backend writes a dataset "<name>.asc".
void solution_writer_write_1d(solution_writer *w, const char *name, const double *u,
                              MKL_INT dim);
void solution_writer_write_int_1d(solution_writer *w, const char *name, const MKL_INT *u,
                                  MKL_INT dim);
void solution_writer_write_2d(solution_writer *w, const char *name, const double *u, MKL_INT nr,
                              MKL_INT nz);
void solution_writer_write_2d_polar(solution_writer *w, const char *name, const double *u,
                                    MKL_INT nr, MKL_INT nth);

#endif /* ROTBOSON_OUTPUT_H */
