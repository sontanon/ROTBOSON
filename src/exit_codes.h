// Process exit codes for the ROTBOSON binary (single-solution contract).
//
// One invocation = one Newton solve. The Python sweep driver (see
// docs/sweep-driver-design.md) keys its decision table off these codes, so
// they are part of the binary's public contract: do not renumber.

#ifndef ROTBOSON_EXIT_CODES_H
#define ROTBOSON_EXIT_CODES_H

enum
{
    RB_EXIT_OK = 0,      // Converged (error_code = 0), solution written.
    RB_EXIT_NEWTON = 1,  // Newton did not converge within maxNewtonIter.
    RB_EXIT_SOLVER = 2,  // Linear solver error (PARDISO/UMFPACK failure).
    RB_EXIT_CONFIG = 3,  // Configuration/parse error (bad TOML, bad seed paths).
    RB_EXIT_IO = 4       // I/O error (output directory/file failure).
};

#endif /* ROTBOSON_EXIT_CODES_H */
