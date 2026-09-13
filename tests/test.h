// Minimal, dependency-free test harness for the Phase 3 C tests.
//
// Deliberately tiny (no framework): a test binary calls CHECK* and returns 0 on
// success / nonzero on failure, which CTest understands. The richer solution
// comparisons live in the Python tools instead.
#ifndef ROTBOSON_TEST_H
#define ROTBOSON_TEST_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int rb_checks = 0;
static int rb_failures = 0;

#define CHECK(cond)                                                                                \
    do                                                                                             \
    {                                                                                              \
        rb_checks++;                                                                               \
        if (!(cond))                                                                               \
        {                                                                                          \
            rb_failures++;                                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                        \
        }                                                                                          \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                      \
    do                                                                                             \
    {                                                                                              \
        rb_checks++;                                                                               \
        double _a = (double)(a);                                                                   \
        double _b = (double)(b);                                                                   \
        double _d = fabs(_a - _b);                                                                 \
        if (_d > (double)(tol))                                                                    \
        {                                                                                          \
            rb_failures++;                                                                         \
            fprintf(stderr, "FAIL %s:%d: %s = %.16e vs %s = %.16e (diff %.3e)\n", __FILE__,        \
                    __LINE__, #a, _a, #b, _b, _d);                                                 \
        }                                                                                          \
    } while (0)

// Call at the end of main(): print the summary and return the exit code.
static int rb_test_summary(void)
{
    if (rb_failures == 0)
    {
        printf("OK: %d checks passed\n", rb_checks);
        return 0;
    }
    fprintf(stderr, "FAILED: %d of %d checks\n", rb_failures, rb_checks);
    return 1;
}

#endif /* ROTBOSON_TEST_H */
