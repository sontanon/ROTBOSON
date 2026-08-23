#include "fornberg.h"

#include <string.h>

void fornberg_weights(const double *x, int n, double z, int m, double *c)
{
    int i, j, d, mn;
    double c1, c2, c3, c4, c5;

    // c is laid out node-major: c[node * (m + 1) + deriv].
    // The recurrence reads entries before they are written, so it must start
    // all-zero (only c[0][0] is seeded below).
    memset(c, 0, (size_t)n * (m + 1) * sizeof(double));

    c[0 * (m + 1) + 0] = 1.0;
    c1 = 1.0;
    c4 = x[0] - z;
    for (i = 1; i < n; ++i)
    {
        mn = (i < m) ? i : m;
        c2 = 1.0;
        c5 = c4; // previous node offset (x[i-1] - z)
        c4 = x[i] - z; // current node offset (x[i] - z)
        for (j = 0; j < i; ++j)
        {
            c3 = x[i] - x[j];
            c2 *= c3;
            if (j == i - 1)
            {
                for (d = mn; d >= 1; --d)
                    c[i * (m + 1) + d] =
                        c1 * (d * c[(i - 1) * (m + 1) + (d - 1)] - c5 * c[(i - 1) * (m + 1) + d]) /
                        c2;
                c[i * (m + 1) + 0] = -c1 * c5 * c[(i - 1) * (m + 1) + 0] / c2;
            }
            for (d = mn; d >= 1; --d)
                c[j * (m + 1) + d] = (c4 * c[j * (m + 1) + d] - d * c[j * (m + 1) + (d - 1)]) / c3;
            c[j * (m + 1) + 0] = c4 * c[j * (m + 1) + 0] / c3;
        }
        c1 = c2;
    }
}

void uniform_nodes(double *x, int half, double h)
{
    int k;
    for (k = 0; k <= 2 * half; ++k)
        x[k] = (k - half) * h;
}
