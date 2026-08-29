// Phase 5: ASCII solution_writer byte-parity.
//
// The ASCII backend must reproduce the legacy .asc format exactly
// (one file per field, "%9.18E" doubles tab-separated in 2D, "%lld" ints)
// and be path-aware (no chdir into the output directory). This test writes
// small known fields into a scratch directory and checks the bytes on disk.
#include "context.h"
#include "output.h"

#include "test.h"

#include <stdio.h>
#include <string.h>

// Read a whole file into a heap buffer (NUL-terminated).
static char *slurp(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
        return NULL;
    if (fseek(fp, 0, SEEK_END) != 0)
    {
        fclose(fp);
        return NULL;
    }
    long n = ftell(fp);
    if (n < 0)
    {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf)
    {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, fp);
    buf[got] = '\0';
    fclose(fp);
    return buf;
}

int main(void)
{
    const char *dir = "test_output_tmp";
    rb_context ctx;
    rb_context_init(&ctx); // output_format defaults to ASCII

    solution_writer *w = solution_writer_open(dir, &ctx, "dummy.toml");
    CHECK(w != NULL);
    if (!w)
        return rb_test_summary();

    // --- 1D double -------------------------------------------------------
    {
        double u[3] = {1.5, -2.25, 1.0 / 3.0};
        solution_writer_write_1d(w, "v1d", u, 3);
    }
    // --- 1D integer ------------------------------------------------------
    {
        MKL_INT u[2] = {42, -7};
        solution_writer_write_int_1d(w, "i1d", u, 2);
    }
    // --- 2D double (2 rows x 3 cols, row-major) ---------------------------
    {
        double u[6] = {1.5, -2.25, 1.0 / 3.0, 0.0, 1.0, -1.0};
        solution_writer_write_2d(w, "v2d", u, 2, 3);
    }
    // --- 2D polar (2 x 2) ------------------------------------------------
    {
        double u[4] = {1.0, 2.0, 3.0, 4.0};
        solution_writer_write_2d_polar(w, "vpol", u, 2, 2);
    }

    solution_writer_close(w);

    // Verify the files live in the scratch dir (path-aware, no chdir).
    char *c1d = slurp("test_output_tmp/v1d.asc");
    char *i1d = slurp("test_output_tmp/i1d.asc");
    char *c2d = slurp("test_output_tmp/v2d.asc");
    char *cp = slurp("test_output_tmp/vpol.asc");

    CHECK(c1d != NULL);
    CHECK(i1d != NULL);
    CHECK(c2d != NULL);
    CHECK(cp != NULL);

    if (c1d)
    {
        CHECK(strcmp(c1d,
                     "1.500000000000000000E+00\n"
                     "-2.250000000000000000E+00\n"
                     "3.333333333333333148E-01\n") == 0);
    }
    if (i1d)
    {
        CHECK(strcmp(i1d, "42\n-7\n") == 0);
    }
    if (c2d)
    {
        CHECK(strcmp(c2d,
                     "1.500000000000000000E+00\t-2.250000000000000000E+00\t3.333333333333333148E-01\n"
                     "0.000000000000000000E+00\t1.000000000000000000E+00\t-1.000000000000000000E+00\n") ==
              0);
    }
    if (cp)
    {
        CHECK(strcmp(cp,
                     "1.000000000000000000E+00\t2.000000000000000000E+00\n"
                     "3.000000000000000000E+00\t4.000000000000000000E+00\n") == 0);
    }

    free(c1d);
    free(i1d);
    free(c2d);
    free(cp);

    // Close is idempotent / NULL-safe.
    solution_writer_close(NULL);

    return rb_test_summary();
}
