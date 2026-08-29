// Internal definition of solution_writer, shared by the backend
// implementations. Only output.c and the backend .c files include this.
#ifndef ROTBOSON_OUTPUT_INTERNAL_H
#define ROTBOSON_OUTPUT_INTERNAL_H

#include "output.h"

// Name of the single HDF5 file written into the solution directory.
#define ROTBOSON_HDF5_FILENAME "solution.h5"

struct solution_writer
{
    output_format format;
    char dirname[RB_PATH_MAX];
    char parfile[RB_PATH_MAX];
    const rb_context *ctx; // borrowed; valid for the writer's lifetime
    void *backend;         // backend state (hid_t for HDF5, NULL for ASCII)
    int closed;

    // Backend callbacks, set by *_backend_init().
    void (*write_1d)(solution_writer *, const char *, const double *, MKL_INT);
    void (*write_int_1d)(solution_writer *, const char *, const MKL_INT *, MKL_INT);
    void (*write_2d)(solution_writer *, const char *, const double *, MKL_INT, MKL_INT);
    void (*write_2d_polar)(solution_writer *, const char *, const double *, MKL_INT, MKL_INT);
    void (*close_backend)(solution_writer *);
};

// Fill `buf` with the path "<dirname>/<name>.asc".
void solution_writer_field_path(const solution_writer *w, const char *name, char *buf, size_t n);

// Backend initializers. Each sets w's callbacks and backend state; return 0
// on success, non-zero on failure (an error message is logged).
int ascii_backend_init(solution_writer *w);
int hdf5_backend_init(solution_writer *w);

#endif /* ROTBOSON_OUTPUT_INTERNAL_H */
