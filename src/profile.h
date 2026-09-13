// Lightweight profiling: phase timers + RSS sampling.
//
// Enabled by ROTBOSON_PROFILE=1; costs one omp_get_wtime() pair per
// instrumented call and reads /proc/self/status only at phase boundaries.
// prof_report() is registered via atexit(), so every exit path (including
// error exits) prints the summary.
#ifndef ROTBOSON_PROFILE_H
#define ROTBOSON_PROFILE_H

// Phase buckets, in report order. Instrumentation lives at the shared
// chokepoints so every solverType path (nleq_err/nleq_res/newton) is covered:
//   JACOBIAN -> csr_gen_jacobian            (assembly)
//   ANALYSIS -> PARDISO phase 11            (reordering + symbolic factorization)
//   FACTOR   -> PARDISO phase 22            (numerical factorization)
//   SOLVE    -> PARDISO phase 33            (back substitution + iterative refinement)
//   RHS      -> rhs()                       (residual evaluation)
//   NORMS    -> norm2_all_variables         (all NORM() call sites)
typedef enum
{
    PROF_JACOBIAN = 0,
    PROF_ANALYSIS,
    PROF_FACTOR,
    PROF_SOLVE,
    PROF_RHS,
    PROF_NORMS,
    PROF_COUNT
} prof_phase;

// Read ROTBOSON_PROFILE and (when enabled) register the atexit report.
void prof_init(void);

// 1 when profiling is enabled.
int prof_enabled(void);

// Timing bracket for one call of phase p.
void prof_begin(prof_phase p);
void prof_end(prof_phase p);

// Current and high-water RSS of this process, in kB (VmRSS / VmHWM).
void prof_rss(double *current_kb, double *peak_kb);

// Print the summary table (no-op unless enabled).
void prof_report(void);

#endif // ROTBOSON_PROFILE_H
