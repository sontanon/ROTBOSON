#include "context.h"
#include "log.h"

#include <string.h>

// Initialize a context to the same pre-parse defaults that param.h used to
// hard-code as global initializers. The parser then overwrites whichever keys
// are present in the parameter file.
void rb_context_init(rb_context *ctx)
{
    memset(ctx, 0, sizeof(*ctx));

    // GRID.
    ctx->dr = 0.0625;
    ctx->dz = 0.0625;
    ctx->NrInterior = 128;
    ctx->NzInterior = 128;
    ctx->NrTotal = 130;
    ctx->NzTotal = 130;
    ctx->dim = 16900;
    ctx->ghost = 1;
    ctx->order = 2;

    // SCALAR FIELD.
    ctx->l = 1;
    ctx->m = 1.0;
    ctx->psi0 = 0.2;
    ctx->sigmaR = 4.0;
    ctx->sigmaZ = 4.0;
    ctx->rExt = 16.0;
    ctx->w0 = 0.7;
    ctx->w_idx = 135200;
    ctx->fixedPhi = 1;
    ctx->fixedPhiR = 4;
    ctx->fixedPhiZ = 4;
    ctx->fixedOmega = 0;

    // INITIAL DATA.
    ctx->readInitialData = 0;
    ctx->ghost_i = 1;
    ctx->order_i = 2;
    ctx->dr_i = 1.0;
    ctx->dz_i = 1.0;

    // SCALE INITIAL DATA.
    ctx->scale_u0 = 1.0;
    ctx->scale_u1 = 1.0;
    ctx->scale_u2 = 1.0;
    ctx->scale_u3 = 1.0;
    ctx->scale_u4 = 1.0;
    ctx->scale_u5 = 1.0;
    ctx->scale_u6 = 1.0;

    // SOLVER.
    ctx->solverType = 1;
    ctx->localSolver = 1;
    ctx->epsilon = 1E-5;
    ctx->maxNewtonIter = 10;
    ctx->lambda0 = 1.0E-3;
    ctx->lambdaMin = 1.0E-8;
    ctx->useLowRank = 0;

    // INITIAL GUESS CHECK.
    ctx->max_initial_guess_checks = 8;
    ctx->norm_f0_target = 1.0E-05;

    // ANALYSIS.
    ctx->phi_max = 1.0;

    // OUTPUT.
    ctx->output_format = 0;      // ASCII (legacy .asc files)
    ctx->log_level = RB_LOG_INFO;

    // All pointer fields and dirname buffers are zeroed by memset above.
}
