// Exit-code contract test (design doc §2).
//
// The Python sweep driver keys its decision table off the binary's exit
// codes, so this test pins the public contract:
//
//   0 = converged, 1 = Newton non-convergence, 2 = solver error,
//   3 = configuration/parse error, 4 = I/O error.
//
// Config-error paths (3) and the happy path (0, tiny real solve) are checked
// here. Solver-error (2) and Newton-failure (1) paths need pathological
// inputs and are left to the driver's integration tests.

#include "test.h"

#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef ROTBOSON_BIN
#define ROTBOSON_BIN "./ROTBOSON"
#endif

#ifndef ROTBOSON_SMOKE_CONFIG
#define ROTBOSON_SMOKE_CONFIG "l1_smoke_ci.toml"
#endif

static int run_binary(const char *arg)
{
    char cmd[1024];
    if (arg)
        snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\" >/dev/null 2>&1", ROTBOSON_BIN, arg);
    else
        snprintf(cmd, sizeof(cmd), "\"%s\" >/dev/null 2>&1", ROTBOSON_BIN);

    int status = system(cmd);
    if (status < 0 || !WIFEXITED(status))
        return -1;
    return WEXITSTATUS(status);
}

// Write a minimal TOML with one extra line appended.
static int write_toml(const char *path, const char *extra_line)
{
    FILE *src = fopen(ROTBOSON_SMOKE_CONFIG, "r");
    if (!src)
    {
        fprintf(stderr, "cannot open smoke config \"%s\"\n", ROTBOSON_SMOKE_CONFIG);
        return -1;
    }
    FILE *dst = fopen(path, "w");
    if (!dst)
    {
        fclose(src);
        return -1;
    }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), src)) > 0)
        fwrite(buf, 1, n, dst);
    fclose(src);
    if (extra_line)
        fprintf(dst, "%s\n", extra_line);
    fclose(dst);
    return 0;
}

int main(void)
{
    int failures = 0;

    // Work in a scratch directory so solution outputs don't pollute the build
    // tree root.
    char tmpl[] = "/tmp/rotboson_exit_codes_XXXXXX";
    char *scratch = mkdtemp(tmpl);
    if (!scratch)
    {
        perror("mkdtemp");
        return 1;
    }
    if (chdir(scratch) != 0)
    {
        perror("chdir");
        return 1;
    }

    // 1. No arguments -> configuration error (3).
    {
        int code = run_binary(NULL);
        printf("no-arg exit code: %d (expect 3)\n", code);
        if (code != 3)
            failures++;
    }

    // 2. Nonexistent parameter file -> configuration error (3).
    {
        int code = run_binary("/nonexistent/params.toml");
        printf("missing-file exit code: %d (expect 3)\n", code);
        if (code != 3)
            failures++;
    }

    // 3. Unknown parameter key -> configuration error (3).
    {
        const char *f = "unknown_key.toml";
        if (write_toml(f, "not_a_real_key = 1") != 0)
            return 1;
        int code = run_binary(f);
        printf("unknown-key exit code: %d (expect 3)\n", code);
        if (code != 3)
            failures++;
    }

    // 4. Out-of-range value -> configuration error (3).
    {
        const char *f = "bad_range.toml";
        if (write_toml(f, "dr = 5.0") != 0) // MAX_DR = 1.0
            return 1;
        int code = run_binary(f);
        printf("bad-range exit code: %d (expect 3)\n", code);
        if (code != 3)
            failures++;
    }

    // 5. Valid solve -> success (0). Reuses the coarse CI smoke config
    //    (~seconds on both backends).
    {
        int code = run_binary(ROTBOSON_SMOKE_CONFIG);
        printf("valid-solve exit code: %d (expect 0)\n", code);
        if (code != 0)
            failures++;
    }

    // Scratch dir is small (one coarse solution); leave it for inspection on
    // failure, clean it on success.
    if (failures == 0)
    {
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "rm -rf \"%s\"", scratch);
        system(cmd);
    }

    if (failures != 0)
    {
        fprintf(stderr, "FAILED: %d exit-code checks\n", failures);
        return 1;
    }
    printf("OK: exit-code contract holds\n");
    return 0;
}
