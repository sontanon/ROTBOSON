#include "profile.h"

#include "tools.h" // omp_get_wtime

#include <string.h>

typedef struct
{
    double seconds;      // accumulated wall time in this phase
    long long calls;     // number of bracketed calls
    double last_rss_kb;  // RSS at the last exit from this phase
    double max_rss_kb;   // max RSS observed at any exit from this phase
} prof_slot;

static prof_slot slots[PROF_COUNT];
static double entry_time[PROF_COUNT];
static double entry_rss[PROF_COUNT];

static int enabled = 0;
static double wall_t0 = 0.0;

static const char *phase_names[PROF_COUNT] = {
    "jacobian assembly",
    "pardiso analysis",
    "pardiso factor",
    "pardiso solve",
    "rhs residual",
    "norms",
};

void prof_init(void)
{
    const char *env = getenv("ROTBOSON_PROFILE");
    if (!env || env[0] == '\0' || env[0] == '0')
        return;
    enabled = 1;
    wall_t0 = omp_get_wtime();
    atexit(prof_report);
}

int prof_enabled(void) { return enabled; }

void prof_rss(double *current_kb, double *peak_kb)
{
    *current_kb = 0.0;
    *peak_kb = 0.0;

    FILE *f = fopen("/proc/self/status", "r");
    if (!f)
        return;

    char line[256];
    while (fgets(line, sizeof(line), f))
    {
        if (strncmp(line, "VmRSS:", 6) == 0)
            sscanf(line + 6, "%lf", current_kb);
        else if (strncmp(line, "VmHWM:", 6) == 0)
            sscanf(line + 6, "%lf", peak_kb);
    }
    fclose(f);
}

void prof_begin(prof_phase p)
{
    if (!enabled)
        return;
    prof_rss(&entry_rss[p], &entry_rss[p]); // current RSS into entry_rss[p]
    entry_time[p] = omp_get_wtime();
}

void prof_end(prof_phase p)
{
    if (!enabled)
        return;
    double t = omp_get_wtime();
    slots[p].seconds += t - entry_time[p];
    slots[p].calls++;

    double cur, peak;
    prof_rss(&cur, &peak);
    slots[p].last_rss_kb = cur;
    if (cur > slots[p].max_rss_kb)
        slots[p].max_rss_kb = cur;
}

void prof_report(void)
{
    if (!enabled)
        return;

    double wall = omp_get_wtime() - wall_t0;
    double cur, peak;
    prof_rss(&cur, &peak);

    printf("\n");
    printf("=================================================================="
           "====\n");
    printf("PROFILING SUMMARY (ROTBOSON_PROFILE=1)\n");
    printf("=================================================================="
           "====\n");
    printf("%-20s %7s %12s %12s %12s\n", "phase", "calls", "seconds", "ms/call",
           "maxRSS(kB)");
    printf("------------------------------------------------------------------"
           "----\n");
    double accounted = 0.0;
    for (int p = 0; p < PROF_COUNT; ++p)
    {
        if (slots[p].calls == 0)
            continue;
        accounted += slots[p].seconds;
        printf("%-20s %7lld %12.3f %12.1f %12.0f\n", phase_names[p], slots[p].calls,
               slots[p].seconds, 1000.0 * slots[p].seconds / slots[p].calls,
               slots[p].max_rss_kb);
    }
    printf("------------------------------------------------------------------"
           "----\n");
    printf("%-20s %7s %12.3f %12s %12.0f\n", "accounted total", "", accounted, "",
           0.0);
    printf("%-20s %7s %12.3f\n", "wall since init", "", wall);
    printf("%-20s                      %12.0f\n", "peak RSS (VmHWM)", peak);
    printf("%-20s                      %12.0f\n", "current RSS", cur);
    printf("=================================================================="
           "====\n");
}
