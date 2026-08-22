// Minimal MKL compatibility shim for the OpenBLAS/SuiteSparse (OSS) build.
//
// The MKL build includes the real <mkl.h>; this header is placed FIRST on the
// include path only for the OSS backend, and provides:
//   - MKL_INT as int64_t (matches the ILP64 interface the code assumes)
//   - the level-1 CBLAS routines the code uses (inline, 0-based idamax/idamin,
//     matching MKL's CBLAS convention)
//   - thread-control entry points as no-ops
#ifndef ROTBOSON_MKL_COMPAT_H
#define ROTBOSON_MKL_COMPAT_H

#include <math.h>
#include <stdint.h>

// Match Intel MKL's ILP64 interface exactly: MKL_INT is `long long`. The
// codebase formats these with %lld, so this must NOT be `int64_t` (= `long`
// on Linux), which would change the underlying type.
typedef long long MKL_INT;

static inline double cblas_ddot(const MKL_INT n, const double *x, const MKL_INT incx,
                                const double *y, const MKL_INT incy)
{
    double sum = 0.0;
    MKL_INT i, ix = 0, iy = 0;
    for (i = 0; i < n; ++i, ix += incx, iy += incy)
        sum += x[ix] * y[iy];
    return sum;
}

static inline void cblas_dscal(const MKL_INT n, const double alpha, double *x,
                               const MKL_INT incx)
{
    MKL_INT i, ix = 0;
    for (i = 0; i < n; ++i, ix += incx)
        x[ix] *= alpha;
}

static inline double cblas_dnrm2(const MKL_INT n, const double *x, const MKL_INT incx)
{
    double scale = 0.0, ssq = 1.0;
    MKL_INT i, ix = 0;
    for (i = 0; i < n; ++i, ix += incx)
    {
        double ax = fabs(x[ix]);
        if (ax != 0.0)
        {
            if (scale < ax)
            {
                ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
                scale = ax;
            }
            else
            {
                ssq += (ax / scale) * (ax / scale);
            }
        }
    }
    return scale * sqrt(ssq);
}

// First occurrence of the maximum |x|; 0-based (MKL CBLAS convention).
static inline MKL_INT cblas_idamax(const MKL_INT n, const double *x, const MKL_INT incx)
{
    MKL_INT i, ix = 0, idx = 0;
    double amax = -1.0;
    for (i = 0; i < n; ++i, ix += incx)
    {
        double ax = fabs(x[ix]);
        if (ax > amax)
        {
            amax = ax;
            idx = i;
        }
    }
    return idx;
}

// First occurrence of the minimum |x|; 0-based (MKL CBLAS convention).
static inline MKL_INT cblas_idamin(const MKL_INT n, const double *x, const MKL_INT incx)
{
    MKL_INT i, ix = 0, idx = 0;
    double amin = -1.0;
    for (i = 0; i < n; ++i, ix += incx)
    {
        double ax = fabs(x[ix]);
        if (amin < 0.0 || ax < amin)
        {
            amin = ax;
            idx = i;
        }
    }
    return idx;
}

// Thread control: the OSS backend has no internal thread pool to configure.
static inline void mkl_set_dynamic(int enable) { (void)enable; }

static inline void mkl_set_num_threads(int num) { (void)num; }

#endif /* ROTBOSON_MKL_COMPAT_H */
