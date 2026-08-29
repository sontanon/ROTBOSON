// HDF5 output backend.
//
// One self-describing file per solution: "<dirname>/solution.h5". Datasets are
// named "<name>.asc" (matching the ASCII backend 1:1) so the same Python
// readers round-trip both formats. Parameters, solver settings and the build
// git hash are stored as scalar attributes on the root group.
//
// Compiled only when CMake finds HDF5 (ROTBOSON_HDF5 defined); otherwise a
// stub reports the missing support at runtime.
#include "output_internal.h"

#include "log.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef ROTBOSON_GIT_HASH
#define ROTBOSON_GIT_HASH "unknown"
#endif

#ifndef ROTBOSON_HDF5

// Stub: HDF5 support was not compiled in.
int hdf5_backend_init(solution_writer *w)
{
    (void)w;
    rb_log(RB_LOG_ERROR,
           "OUTPUT: outputFormat=\"hdf5\" requested but this binary was built without HDF5.\n");
    return -1;
}

#else

#include <hdf5.h>

// File/dataset attribute helper for scalars and fixed strings.
static void put_double_attr(hid_t loc, const char *name, double v)
{
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr >= 0)
    {
        H5Awrite(attr, H5T_NATIVE_DOUBLE, &v);
        H5Aclose(attr);
    }
    H5Sclose(space);
}

static void put_llong_attr(hid_t loc, const char *name, long long v)
{
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, H5T_STD_I64LE, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr >= 0)
    {
        H5Awrite(attr, H5T_NATIVE_LLONG, &v);
        H5Aclose(attr);
    }
    H5Sclose(space);
}

static void put_string_attr(hid_t loc, const char *name, const char *v)
{
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, strlen(v) + 1);
    H5Tset_strpad(type, H5T_STR_NULLTERM);
    hid_t attr = H5Acreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr >= 0)
    {
        H5Awrite(attr, type, v);
        H5Aclose(attr);
    }
    H5Tclose(type);
    H5Sclose(space);
}

static hid_t writer_file(const solution_writer *w)
{
    return (hid_t)(intptr_t)w->backend;
}

// Create (or replace) a dataset of IEEE doubles.
static void write_double_dataset(solution_writer *w, const char *name, const double *u, int rank,
                                 const hsize_t *dims)
{
    hid_t file = writer_file(w);

    // Replace any pre-existing dataset with the same name (idempotent re-run).
    if (H5Lexists(file, name, H5P_DEFAULT) > 0)
        H5Ldelete(file, name, H5P_DEFAULT);

    hid_t space = H5Screate_simple(rank, dims, NULL);
    hid_t dset = H5Dcreate2(file, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0 || space < 0)
    {
        rb_log(RB_LOG_ERROR, "OUTPUT: cannot create HDF5 dataset \"%s\".\n", name);
        exit(EXIT_FAILURE);
    }
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, u);
    H5Dclose(dset);
    H5Sclose(space);
}

static void hdf5_write_1d(solution_writer *w, const char *name, const double *u, MKL_INT dim)
{
    char dname[RB_PATH_MAX];
    snprintf(dname, sizeof(dname), "%s.asc", name);
    hsize_t dims[1] = {(hsize_t)dim};
    write_double_dataset(w, dname, u, 1, dims);
}

static void hdf5_write_int_1d(solution_writer *w, const char *name, const MKL_INT *u, MKL_INT dim)
{
    char dname[RB_PATH_MAX];
    snprintf(dname, sizeof(dname), "%s.asc", name);

    hid_t file = writer_file(w);
    if (H5Lexists(file, dname, H5P_DEFAULT) > 0)
        H5Ldelete(file, dname, H5P_DEFAULT);

    hsize_t dims[1] = {(hsize_t)dim};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, dname, H5T_STD_I64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0 || space < 0)
    {
        rb_log(RB_LOG_ERROR, "OUTPUT: cannot create HDF5 dataset \"%s\".\n", dname);
        exit(EXIT_FAILURE);
    }
    H5Dwrite(dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, u);
    H5Dclose(dset);
    H5Sclose(space);
}

static void hdf5_write_2d(solution_writer *w, const char *name, const double *u, MKL_INT nr,
                          MKL_INT nz)
{
    char dname[RB_PATH_MAX];
    snprintf(dname, sizeof(dname), "%s.asc", name);
    hsize_t dims[2] = {(hsize_t)nr, (hsize_t)nz};
    write_double_dataset(w, dname, u, 2, dims);
}

static void hdf5_write_2d_polar(solution_writer *w, const char *name, const double *u, MKL_INT nr,
                                MKL_INT nth)
{
    char dname[RB_PATH_MAX];
    snprintf(dname, sizeof(dname), "%s.asc", name);
    hsize_t dims[2] = {(hsize_t)nr, (hsize_t)nth};
    write_double_dataset(w, dname, u, 2, dims);
}

// Write all scalar parameters + solver settings + provenance as root attributes.
static void write_attributes(solution_writer *w)
{
    const rb_context *c = w->ctx;
    if (!c)
        return;

    hid_t file = writer_file(w);

    put_string_attr(file, "git_hash", ROTBOSON_GIT_HASH);
    put_string_attr(file, "parfile", w->parfile);
    put_string_attr(file, "output_backend", "hdf5");
    put_llong_attr(file, "format_version", 1);

    // GRID.
    put_double_attr(file, "dr", c->dr);
    put_double_attr(file, "dz", c->dz);
    put_llong_attr(file, "NrInterior", c->NrInterior);
    put_llong_attr(file, "NzInterior", c->NzInterior);
    put_llong_attr(file, "NrTotal", c->NrTotal);
    put_llong_attr(file, "NzTotal", c->NzTotal);
    put_llong_attr(file, "dim", c->dim);
    put_llong_attr(file, "ghost", c->ghost);
    put_llong_attr(file, "order", c->order);

    // SCALAR FIELD.
    put_llong_attr(file, "l", c->l);
    put_double_attr(file, "m", c->m);
    put_double_attr(file, "psi0", c->psi0);
    put_double_attr(file, "sigmaR", c->sigmaR);
    put_double_attr(file, "sigmaZ", c->sigmaZ);
    put_double_attr(file, "rExt", c->rExt);
    put_double_attr(file, "w0", c->w0);
    put_llong_attr(file, "w_idx", c->w_idx);
    put_llong_attr(file, "fixedPhi", c->fixedPhi);
    put_llong_attr(file, "fixedPhiR", c->fixedPhiR);
    put_llong_attr(file, "fixedPhiZ", c->fixedPhiZ);
    put_llong_attr(file, "fixedOmega", c->fixedOmega);

    // INITIAL DATA (file paths, if any).
    if (c->log_alpha_i)
        put_string_attr(file, "log_alpha_i", c->log_alpha_i);
    if (c->beta_i)
        put_string_attr(file, "beta_i", c->beta_i);
    if (c->log_h_i)
        put_string_attr(file, "log_h_i", c->log_h_i);
    if (c->log_a_i)
        put_string_attr(file, "log_a_i", c->log_a_i);
    if (c->psi_i)
        put_string_attr(file, "psi_i", c->psi_i);
    if (c->lambda_i)
        put_string_attr(file, "lambda_i", c->lambda_i);
    if (c->w_i)
        put_string_attr(file, "w_i", c->w_i);
    put_llong_attr(file, "readInitialData", c->readInitialData);
    put_llong_attr(file, "NrTotalInitial", c->NrTotalInitial);
    put_llong_attr(file, "NzTotalInitial", c->NzTotalInitial);
    put_llong_attr(file, "ghost_i", c->ghost_i);
    put_llong_attr(file, "order_i", c->order_i);
    put_double_attr(file, "dr_i", c->dr_i);
    put_double_attr(file, "dz_i", c->dz_i);

    // SCALE INITIAL DATA.
    put_double_attr(file, "scale_u0", c->scale_u0);
    put_double_attr(file, "scale_u1", c->scale_u1);
    put_double_attr(file, "scale_u2", c->scale_u2);
    put_double_attr(file, "scale_u3", c->scale_u3);
    put_double_attr(file, "scale_u4", c->scale_u4);
    put_double_attr(file, "scale_u5", c->scale_u5);
    put_double_attr(file, "scale_u6", c->scale_u6);
    put_double_attr(file, "scale_next", c->scale_next);

    // SOLVER.
    put_llong_attr(file, "solverType", c->solverType);
    put_llong_attr(file, "localSolver", c->localSolver);
    put_double_attr(file, "epsilon", c->epsilon);
    put_llong_attr(file, "maxNewtonIter", c->maxNewtonIter);
    put_double_attr(file, "lambda0", c->lambda0);
    put_double_attr(file, "lambdaMin", c->lambdaMin);
    put_llong_attr(file, "useLowRank", c->useLowRank);

    // INITIAL GUESS CHECK.
    put_llong_attr(file, "max_initial_guess_checks", c->max_initial_guess_checks);
    put_double_attr(file, "norm_f0_target", c->norm_f0_target);

    // SWEEP CONTROL.
    put_llong_attr(file, "sweep", c->sweep);
    put_double_attr(file, "rr_phi_max_minimum", c->rr_phi_max_minimum);
    put_double_attr(file, "rr_phi_max_maximum", c->rr_phi_max_maximum);
    put_llong_attr(file, "hwl_min", c->hwl_min);
    put_llong_attr(file, "hwl_max", c->hwl_max);
    put_double_attr(file, "w_max", c->w_max);
    put_double_attr(file, "w_min", c->w_min);
    put_double_attr(file, "w_step", c->w_step);

    // SPHERICAL ANALYSIS GRID.
    put_llong_attr(file, "NrrTotal", c->NrrTotal);
    put_llong_attr(file, "NthTotal", c->NthTotal);
    put_llong_attr(file, "p_dim", c->p_dim);
    put_double_attr(file, "drr", c->drr);
    put_double_attr(file, "dth", c->dth);
    put_double_attr(file, "rr_inf", c->rr_inf);

    // ANALYSIS RESULTS (computed after the solve; valid at close time).
    put_double_attr(file, "M_KOMAR", c->M_KOMAR);
    put_double_attr(file, "J_KOMAR", c->J_KOMAR);
    put_double_attr(file, "GRV2", c->GRV2);
    put_double_attr(file, "GRV3", c->GRV3);
    put_double_attr(file, "phi_max", c->phi_max);
    put_double_attr(file, "rr_phi_max", c->rr_phi_max);
    put_llong_attr(file, "hwl_res", c->hwl_res);
}

static void hdf5_close(solution_writer *w)
{
    write_attributes(w);
    hid_t file = writer_file(w);
    if (file >= 0)
        H5Fclose(file);
    w->backend = (void *)(intptr_t)-1;
}

int hdf5_backend_init(solution_writer *w)
{
    char path[RB_PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", w->dirname, ROTBOSON_HDF5_FILENAME);

    hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
    {
        rb_log(RB_LOG_ERROR, "OUTPUT: cannot create HDF5 file \"%s\".\n", path);
        return -1;
    }

    w->write_1d = hdf5_write_1d;
    w->write_int_1d = hdf5_write_int_1d;
    w->write_2d = hdf5_write_2d;
    w->write_2d_polar = hdf5_write_2d_polar;
    w->close_backend = hdf5_close;
    w->backend = (void *)(intptr_t)file;
    return 0;
}

#endif /* ROTBOSON_HDF5 */
