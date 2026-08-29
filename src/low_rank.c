#include "tools.h"
#include "context.h"
#include "pardiso_param.h"

// Debug diff printer.
#undef DEBUG

void solver_diff_gen(rb_context *ctx)
{
    // Local alias for the IDX macro, which indexes by row-major stride NzTotal.
    const MKL_INT NzTotal = ctx->NzTotal;

    const MKL_INT P4_CC[6] = {30, 29, 28, 38, 31, 40};
    const MKL_INT P4_CS[6] = {30, 30, 29, 38, 31, 41};
    const MKL_INT P4_SC[6] = {30, 30, 29, 38, 31, 43};
    const MKL_INT P4_SS[6] = {30, 30, 29, 38, 31, 43};
    const MKL_INT P2_CC[6] = {18, 17, 16, 22, 19, 22};

    // Auxiliary integers.
    MKL_INT i, j, k;
    // MKL_INT offset1, offset2, offset3, offset4, offset5, offset6;
    MKL_INT offset[6] = {0, 0, 0, 0, 0, 0};

    // Number of different elements after update.
    MKL_INT ndiff = 0;

    // Base ndiff with Dirichlet.
    if (ctx->order == 4)
    {
        // Main interior points.
        // 30 + 29 + 28 + 38 + 31 + 40 = 196.
        ndiff = (P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4] + P4_CC[5]) *
                ctx->NrInterior * ctx->NzInterior;
        // Add boundary strip.
        // cs: 30 + 30 + 29 + 38 + 31 + 41 = 199.
        // sc: 30 + 30 + 29 + 38 + 31 + 43 = 201.
        // ss: 30 + 30 + 29 + 38 + 31 + 43 = 201.
        ndiff +=
            (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + P4_CS[5]) * ctx->NrInterior;
        ndiff +=
            (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4] + P4_SC[5]) * ctx->NzInterior;
        ndiff += (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3] + P4_SS[4] + P4_SS[5]);
        // Add phiBoundOrder.
        ndiff += 2 * (ctx->NrInterior + ctx->NzInterior + 3);
    }
    else
    {
        // Main interior points.
        // 18 + 17 + 16 + 22 + 19 + 22 = 114.
        ndiff = (P2_CC[0] + P2_CC[1] + P2_CC[2] + P2_CC[3] + P2_CC[4] + P2_CC[5]) *
                ctx->NrInterior * ctx->NzInterior;
        // Add phiBoundOrder.
        ndiff += 2 * (ctx->NrInterior + ctx->NzInterior + 1);
    }

    // Allocate memory for diff array.
    diff = (MKL_INT *)SAFE_MALLOC(sizeof(MKL_INT) * (2 * ndiff + 1));

    // First element of diff is ndiff.
    diff[0] = ndiff;

    // Fill in rest of array.
    if (ctx->order == 4)
    {
// Interior points.
#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(i, j, k, offset)
        for (i = ctx->ghost; i < ctx->ghost + ctx->NrInterior; ++i)
        {
            // 1. log_alpha: 30 different points.
            offset[0] = 1 + 2 * ((P4_CC[0] * ctx->NzInterior + P4_CS[0]) * (i - ctx->ghost));
            for (j = ctx->ghost; j < ctx->ghost + ctx->NzInterior; ++j)
            {
                // Row indices are all row IDX(i, j).
                for (k = 0; k < P4_CC[0]; ++k)
                {
                    diff[offset[0] + 2 * k] = IDX(i, j);
                }
                // Column indices.
                diff[offset[0] + 0 * 2 + 1] = IDX(i - 2, j);
                diff[offset[0] + 1 * 2 + 1] = IDX(i - 1, j);
                diff[offset[0] + 2 * 2 + 1] = IDX(i, j - 2);
                diff[offset[0] + 3 * 2 + 1] = IDX(i, j - 1);
                diff[offset[0] + 4 * 2 + 1] = IDX(i, j);
                diff[offset[0] + 5 * 2 + 1] = IDX(i, j + 1);
                diff[offset[0] + 6 * 2 + 1] = IDX(i, j + 2);
                diff[offset[0] + 7 * 2 + 1] = IDX(i + 1, j);
                diff[offset[0] + 8 * 2 + 1] = IDX(i + 2, j);
                diff[offset[0] + 9 * 2 + 1] = ctx->dim + IDX(i - 2, j);
                diff[offset[0] + 10 * 2 + 1] = ctx->dim + IDX(i - 1, j);
                diff[offset[0] + 11 * 2 + 1] = ctx->dim + IDX(i, j - 2);
                diff[offset[0] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 1);
                diff[offset[0] + 13 * 2 + 1] = ctx->dim + IDX(i, j);
                diff[offset[0] + 14 * 2 + 1] = ctx->dim + IDX(i, j + 1);
                diff[offset[0] + 15 * 2 + 1] = ctx->dim + IDX(i, j + 2);
                diff[offset[0] + 16 * 2 + 1] = ctx->dim + IDX(i + 1, j);
                diff[offset[0] + 17 * 2 + 1] = ctx->dim + IDX(i + 2, j);
                diff[offset[0] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
                diff[offset[0] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[0] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
                diff[offset[0] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[0] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
                diff[offset[0] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[0] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
                diff[offset[0] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[0] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
                diff[offset[0] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
                diff[offset[0] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
                diff[offset[0] + 29 * 2 + 1] = GNUM * ctx->dim;
                // Update offset by 30.
                offset[0] += 2 * P4_CC[0];
            }
            // Semi-one-sided: 30 points.
            j = ctx->ghost + ctx->NzInterior;
            // Row indices are all row IDX(i, j).
            for (k = 0; k < P4_CS[0]; ++k)
            {
                diff[offset[0] + 2 * k] = IDX(i, j);
            }
            // Columns.
            diff[offset[0] + 0 * 2 + 1] = IDX(i - 2, j);
            diff[offset[0] + 1 * 2 + 1] = IDX(i - 1, j);
            diff[offset[0] + 2 * 2 + 1] = IDX(i, j - 3);
            diff[offset[0] + 3 * 2 + 1] = IDX(i, j - 2);
            diff[offset[0] + 4 * 2 + 1] = IDX(i, j - 1);
            diff[offset[0] + 5 * 2 + 1] = IDX(i, j);
            diff[offset[0] + 6 * 2 + 1] = IDX(i, j + 1);
            diff[offset[0] + 7 * 2 + 1] = IDX(i + 1, j);
            diff[offset[0] + 8 * 2 + 1] = IDX(i + 2, j);
            diff[offset[0] + 9 * 2 + 1] = ctx->dim + IDX(i - 2, j);
            diff[offset[0] + 10 * 2 + 1] = ctx->dim + IDX(i - 1, j);
            diff[offset[0] + 11 * 2 + 1] = ctx->dim + IDX(i, j - 3);
            diff[offset[0] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 2);
            diff[offset[0] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 1);
            diff[offset[0] + 14 * 2 + 1] = ctx->dim + IDX(i, j);
            diff[offset[0] + 15 * 2 + 1] = ctx->dim + IDX(i, j + 1);
            diff[offset[0] + 16 * 2 + 1] = ctx->dim + IDX(i + 1, j);
            diff[offset[0] + 17 * 2 + 1] = ctx->dim + IDX(i + 2, j);
            diff[offset[0] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[0] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[0] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
            diff[offset[0] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[0] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[0] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[0] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[0] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[0] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
            diff[offset[0] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
            diff[offset[0] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[0] + 29 * 2 + 1] = GNUM * ctx->dim;

            // 2. beta: 29 points.
            offset[1] =
                1 + 2 * (P4_CC[0] * ctx->NrInterior * ctx->NzInterior + P4_CS[0] * ctx->NrInterior +
                         P4_SC[0] * ctx->NzInterior + P4_SS[0] +
                         (P4_CC[1] * ctx->NzInterior + P4_CS[1]) * (i - ctx->ghost));
            for (j = ctx->ghost; j < ctx->ghost + ctx->NzInterior; ++j)
            {
                // Row indices are all row dim + IDX(i, j).
                for (k = 0; k < P4_CC[1]; ++k)
                {
                    diff[offset[1] + 2 * k] = ctx->dim + IDX(i, j);
                }
                // Column indices.
                diff[offset[1] + 0 * 2 + 1] = IDX(i - 2, j);
                diff[offset[1] + 1 * 2 + 1] = IDX(i - 1, j);
                diff[offset[1] + 2 * 2 + 1] = IDX(i, j - 2);
                diff[offset[1] + 3 * 2 + 1] = IDX(i, j - 1);
                diff[offset[1] + 4 * 2 + 1] = IDX(i, j + 1);
                diff[offset[1] + 5 * 2 + 1] = IDX(i, j + 2);
                diff[offset[1] + 6 * 2 + 1] = IDX(i + 1, j);
                diff[offset[1] + 7 * 2 + 1] = IDX(i + 2, j);
                diff[offset[1] + 8 * 2 + 1] = ctx->dim + IDX(i - 2, j);
                diff[offset[1] + 9 * 2 + 1] = ctx->dim + IDX(i - 1, j);
                diff[offset[1] + 10 * 2 + 1] = ctx->dim + IDX(i, j - 2);
                diff[offset[1] + 11 * 2 + 1] = ctx->dim + IDX(i, j - 1);
                diff[offset[1] + 12 * 2 + 1] = ctx->dim + IDX(i, j);
                diff[offset[1] + 13 * 2 + 1] = ctx->dim + IDX(i, j + 1);
                diff[offset[1] + 14 * 2 + 1] = ctx->dim + IDX(i, j + 2);
                diff[offset[1] + 15 * 2 + 1] = ctx->dim + IDX(i + 1, j);
                diff[offset[1] + 16 * 2 + 1] = ctx->dim + IDX(i + 2, j);
                diff[offset[1] + 17 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
                diff[offset[1] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[1] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
                diff[offset[1] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[1] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
                diff[offset[1] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[1] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
                diff[offset[1] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[1] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
                diff[offset[1] + 26 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
                diff[offset[1] + 27 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
                diff[offset[1] + 28 * 2 + 1] = GNUM * ctx->dim;
                // Update offset by 29.
                offset[1] += 2 * P4_CC[1];
            }
            // Semi-one-sided: 30 points.
            j = ctx->ghost + ctx->NzInterior;
            // Row indices are all row dim + IDX(i, j).
            for (k = 0; k < P4_CS[1]; ++k)
            {
                diff[offset[1] + 2 * k] = ctx->dim + IDX(i, j);
            }
            // Columns.
            diff[offset[1] + 0 * 2 + 1] = IDX(i - 2, j);
            diff[offset[1] + 1 * 2 + 1] = IDX(i - 1, j);
            diff[offset[1] + 2 * 2 + 1] = IDX(i, j - 3);
            diff[offset[1] + 3 * 2 + 1] = IDX(i, j - 2);
            diff[offset[1] + 4 * 2 + 1] = IDX(i, j - 1);
            diff[offset[1] + 5 * 2 + 1] = IDX(i, j);
            diff[offset[1] + 6 * 2 + 1] = IDX(i, j + 1);
            diff[offset[1] + 7 * 2 + 1] = IDX(i + 1, j);
            diff[offset[1] + 8 * 2 + 1] = IDX(i + 2, j);
            diff[offset[1] + 9 * 2 + 1] = ctx->dim + IDX(i - 2, j);
            diff[offset[1] + 10 * 2 + 1] = ctx->dim + IDX(i - 1, j);
            diff[offset[1] + 11 * 2 + 1] = ctx->dim + IDX(i, j - 3);
            diff[offset[1] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 2);
            diff[offset[1] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 1);
            diff[offset[1] + 14 * 2 + 1] = ctx->dim + IDX(i, j);
            diff[offset[1] + 15 * 2 + 1] = ctx->dim + IDX(i, j + 1);
            diff[offset[1] + 16 * 2 + 1] = ctx->dim + IDX(i + 1, j);
            diff[offset[1] + 17 * 2 + 1] = ctx->dim + IDX(i + 2, j);
            diff[offset[1] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[1] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[1] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
            diff[offset[1] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[1] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[1] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[1] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[1] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[1] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
            diff[offset[1] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
            diff[offset[1] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[1] + 29 * 2 + 1] = GNUM * ctx->dim;

            // 3. log_h: 28 points.
            offset[2] = 1 + 2 * ((P4_CC[0] + P4_CC[1]) * ctx->NrInterior * ctx->NzInterior +
                                 (P4_CS[0] + P4_CS[1]) * ctx->NrInterior +
                                 (P4_SC[0] + P4_SC[1]) * ctx->NzInterior + (P4_SS[0] + P4_SS[1]) +
                                 (P4_CC[2] * ctx->NzInterior + P4_CS[2]) * (i - ctx->ghost));
            for (j = ctx->ghost; j < ctx->ghost + ctx->NzInterior; ++j)
            {
                // Row indices are all row 2 * dim + IDX(i, j).
                for (k = 0; k < P4_CC[2]; ++k)
                {
                    diff[offset[2] + 2 * k] = 2 * ctx->dim + IDX(i, j);
                }
                // Column indices.
                diff[offset[2] + 0 * 2 + 1] = IDX(i - 2, j);
                diff[offset[2] + 1 * 2 + 1] = IDX(i - 1, j);
                diff[offset[2] + 2 * 2 + 1] = IDX(i, j - 2);
                diff[offset[2] + 3 * 2 + 1] = IDX(i, j - 1);
                diff[offset[2] + 4 * 2 + 1] = IDX(i, j);
                diff[offset[2] + 5 * 2 + 1] = IDX(i, j + 1);
                diff[offset[2] + 6 * 2 + 1] = IDX(i, j + 2);
                diff[offset[2] + 7 * 2 + 1] = IDX(i + 1, j);
                diff[offset[2] + 8 * 2 + 1] = IDX(i + 2, j);
                diff[offset[2] + 9 * 2 + 1] = ctx->dim + IDX(i - 2, j);
                diff[offset[2] + 10 * 2 + 1] = ctx->dim + IDX(i - 1, j);
                diff[offset[2] + 11 * 2 + 1] = ctx->dim + IDX(i, j - 2);
                diff[offset[2] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 1);
                diff[offset[2] + 13 * 2 + 1] = ctx->dim + IDX(i, j + 1);
                diff[offset[2] + 14 * 2 + 1] = ctx->dim + IDX(i, j + 2);
                diff[offset[2] + 15 * 2 + 1] = ctx->dim + IDX(i + 1, j);
                diff[offset[2] + 16 * 2 + 1] = ctx->dim + IDX(i + 2, j);
                diff[offset[2] + 17 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
                diff[offset[2] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[2] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
                diff[offset[2] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[2] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[2] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
                diff[offset[2] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[2] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
                diff[offset[2] + 26 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
                diff[offset[2] + 27 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
                // Update offset by 28.
                offset[2] += 2 * P4_CC[2];
            }
            // Semi-one-sided: 29 points.
            j = ctx->ghost + ctx->NzInterior;
            // Row indices are all row 2 * dim + IDX(i, j).
            for (k = 0; k < P4_CS[2]; ++k)
            {
                diff[offset[2] + 2 * k] = 2 * ctx->dim + IDX(i, j);
            }
            // Columns.
            diff[offset[2] + 0 * 2 + 1] = IDX(i - 2, j);
            diff[offset[2] + 1 * 2 + 1] = IDX(i - 1, j);
            diff[offset[2] + 2 * 2 + 1] = IDX(i, j - 3);
            diff[offset[2] + 3 * 2 + 1] = IDX(i, j - 2);
            diff[offset[2] + 4 * 2 + 1] = IDX(i, j - 1);
            diff[offset[2] + 5 * 2 + 1] = IDX(i, j);
            diff[offset[2] + 6 * 2 + 1] = IDX(i, j + 1);
            diff[offset[2] + 7 * 2 + 1] = IDX(i + 1, j);
            diff[offset[2] + 8 * 2 + 1] = IDX(i + 2, j);
            diff[offset[2] + 9 * 2 + 1] = ctx->dim + IDX(i - 2, j);
            diff[offset[2] + 10 * 2 + 1] = ctx->dim + IDX(i - 1, j);
            diff[offset[2] + 11 * 2 + 1] = ctx->dim + IDX(i, j - 3);
            diff[offset[2] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 2);
            diff[offset[2] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 1);
            diff[offset[2] + 14 * 2 + 1] = ctx->dim + IDX(i, j);
            diff[offset[2] + 15 * 2 + 1] = ctx->dim + IDX(i, j + 1);
            diff[offset[2] + 16 * 2 + 1] = ctx->dim + IDX(i + 1, j);
            diff[offset[2] + 17 * 2 + 1] = ctx->dim + IDX(i + 2, j);
            diff[offset[2] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[2] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[2] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
            diff[offset[2] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[2] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[2] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[2] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[2] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[2] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
            diff[offset[2] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
            diff[offset[2] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i, j);

            // 4. log_a: 38 points.
            offset[3] =
                1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2]) * ctx->NrInterior * ctx->NzInterior +
                         (P4_CS[0] + P4_CS[1] + P4_CS[2]) * ctx->NrInterior +
                         (P4_SC[0] + P4_SC[1] + P4_SC[2]) * ctx->NzInterior +
                         (P4_SS[0] + P4_SS[1] + P4_SS[2]) +
                         (P4_CC[3] * ctx->NzInterior + P4_CS[3]) * (i - ctx->ghost));
            for (j = ctx->ghost; j < ctx->ghost + ctx->NzInterior; ++j)
            {
                // Row indices are all row 3 * dim + IDX(i, j).
                for (k = 0; k < P4_CC[3]; ++k)
                {
                    diff[offset[3] + 2 * k] = 3 * ctx->dim + IDX(i, j);
                }
                // Column indices.
                diff[offset[3] + 0 * 2 + 1] = IDX(i - 2, j);
                diff[offset[3] + 1 * 2 + 1] = IDX(i - 1, j);
                diff[offset[3] + 2 * 2 + 1] = IDX(i, j - 2);
                diff[offset[3] + 3 * 2 + 1] = IDX(i, j - 1);
                diff[offset[3] + 4 * 2 + 1] = IDX(i, j);
                diff[offset[3] + 5 * 2 + 1] = IDX(i, j + 1);
                diff[offset[3] + 6 * 2 + 1] = IDX(i, j + 2);
                diff[offset[3] + 7 * 2 + 1] = IDX(i + 1, j);
                diff[offset[3] + 8 * 2 + 1] = IDX(i + 2, j);
                diff[offset[3] + 9 * 2 + 1] = ctx->dim + IDX(i - 2, j);
                diff[offset[3] + 10 * 2 + 1] = ctx->dim + IDX(i - 1, j);
                diff[offset[3] + 11 * 2 + 1] = ctx->dim + IDX(i, j - 2);
                diff[offset[3] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 1);
                diff[offset[3] + 13 * 2 + 1] = ctx->dim + IDX(i, j);
                diff[offset[3] + 14 * 2 + 1] = ctx->dim + IDX(i, j + 1);
                diff[offset[3] + 15 * 2 + 1] = ctx->dim + IDX(i, j + 2);
                diff[offset[3] + 16 * 2 + 1] = ctx->dim + IDX(i + 1, j);
                diff[offset[3] + 17 * 2 + 1] = ctx->dim + IDX(i + 2, j);
                diff[offset[3] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
                diff[offset[3] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[3] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
                diff[offset[3] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[3] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
                diff[offset[3] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[3] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
                diff[offset[3] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[3] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
                diff[offset[3] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
                diff[offset[3] + 29 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
                diff[offset[3] + 30 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 2);
                diff[offset[3] + 31 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 1);
                diff[offset[3] + 32 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
                diff[offset[3] + 33 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 1);
                diff[offset[3] + 34 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 2);
                diff[offset[3] + 35 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
                diff[offset[3] + 36 * 2 + 1] = 4 * ctx->dim + IDX(i + 2, j);
                diff[offset[3] + 37 * 2 + 1] = GNUM * ctx->dim;
                // Update offset by 38.
                offset[3] += 2 * P4_CC[3];
            }
            // Semi-one-sided: 38 points.
            j = ctx->ghost + ctx->NzInterior;
            // Row indices are all row 3 * dim + IDX(i, j).
            for (k = 0; k < P4_CS[3]; ++k)
            {
                diff[offset[3] + 2 * k] = 3 * ctx->dim + IDX(i, j);
            }
            // Columns.
            diff[offset[3] + 0 * 2 + 1] = IDX(i - 2, j);
            diff[offset[3] + 1 * 2 + 1] = IDX(i - 1, j);
            diff[offset[3] + 2 * 2 + 1] = IDX(i, j - 3);
            diff[offset[3] + 3 * 2 + 1] = IDX(i, j - 2);
            diff[offset[3] + 4 * 2 + 1] = IDX(i, j - 1);
            diff[offset[3] + 5 * 2 + 1] = IDX(i, j);
            diff[offset[3] + 6 * 2 + 1] = IDX(i, j + 1);
            diff[offset[3] + 7 * 2 + 1] = IDX(i + 1, j);
            diff[offset[3] + 8 * 2 + 1] = IDX(i + 2, j);
            diff[offset[3] + 9 * 2 + 1] = ctx->dim + IDX(i - 2, j);
            diff[offset[3] + 10 * 2 + 1] = ctx->dim + IDX(i - 1, j);
            diff[offset[3] + 11 * 2 + 1] = ctx->dim + IDX(i, j - 3);
            diff[offset[3] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 2);
            diff[offset[3] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 1);
            diff[offset[3] + 14 * 2 + 1] = ctx->dim + IDX(i, j);
            diff[offset[3] + 15 * 2 + 1] = ctx->dim + IDX(i, j + 1);
            diff[offset[3] + 16 * 2 + 1] = ctx->dim + IDX(i + 1, j);
            diff[offset[3] + 17 * 2 + 1] = ctx->dim + IDX(i + 2, j);
            diff[offset[3] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[3] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[3] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
            diff[offset[3] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[3] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[3] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[3] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[3] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[3] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
            diff[offset[3] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
            diff[offset[3] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
            diff[offset[3] + 29 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
            diff[offset[3] + 30 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 3);
            diff[offset[3] + 31 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 2);
            diff[offset[3] + 32 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 1);
            diff[offset[3] + 33 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[3] + 34 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 1);
            diff[offset[3] + 35 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
            diff[offset[3] + 36 * 2 + 1] = 4 * ctx->dim + IDX(i + 2, j);
            diff[offset[3] + 37 * 2 + 1] = GNUM * ctx->dim;

            // 5. psi: 31 points.
            offset[4] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3]) * ctx->NrInterior *
                                     ctx->NzInterior +
                                 (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3]) * ctx->NrInterior +
                                 (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3]) * ctx->NzInterior +
                                 (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3]) +
                                 (P4_CC[4] * ctx->NzInterior + P4_CS[4] + 2) * (i - ctx->ghost));
            for (j = ctx->ghost; j < ctx->ghost + ctx->NzInterior; ++j)
            {
                // Row indices are all row 4 * dim + IDX(i, j).
                for (k = 0; k < P4_CC[4]; ++k)
                {
                    diff[offset[4] + 2 * k] = 4 * ctx->dim + IDX(i, j);
                }
                // Column indices.
                diff[offset[4] + 0 * 2 + 1] = IDX(i - 2, j);
                diff[offset[4] + 1 * 2 + 1] = IDX(i - 1, j);
                diff[offset[4] + 2 * 2 + 1] = IDX(i, j - 2);
                diff[offset[4] + 3 * 2 + 1] = IDX(i, j - 1);
                diff[offset[4] + 4 * 2 + 1] = IDX(i, j);
                diff[offset[4] + 5 * 2 + 1] = IDX(i, j + 1);
                diff[offset[4] + 6 * 2 + 1] = IDX(i, j + 2);
                diff[offset[4] + 7 * 2 + 1] = IDX(i + 1, j);
                diff[offset[4] + 8 * 2 + 1] = IDX(i + 2, j);
                diff[offset[4] + 9 * 2 + 1] = ctx->dim + IDX(i, j);
                diff[offset[4] + 10 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
                diff[offset[4] + 11 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[4] + 12 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
                diff[offset[4] + 13 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[4] + 14 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
                diff[offset[4] + 15 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[4] + 16 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
                diff[offset[4] + 17 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[4] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
                diff[offset[4] + 19 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
                diff[offset[4] + 20 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
                diff[offset[4] + 21 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
                diff[offset[4] + 22 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 2);
                diff[offset[4] + 23 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 1);
                diff[offset[4] + 24 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 25 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 1);
                diff[offset[4] + 26 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 2);
                diff[offset[4] + 27 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
                diff[offset[4] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i + 2, j);
                diff[offset[4] + 29 * 2 + 1] = 5 * ctx->dim + IDX(i, j);
                diff[offset[4] + 30 * 2 + 1] = GNUM * ctx->dim;
                // Update offset by 31.
                offset[4] += 2 * P4_CC[4];
            }
            // Semi-one-sided: 31 points.
            j = ctx->ghost + ctx->NzInterior;
            // Row indices are all row 4 * dim + IDX(i, j).
            for (k = 0; k < P4_CS[4]; ++k)
            {
                diff[offset[4] + 2 * k] = 4 * ctx->dim + IDX(i, j);
            }
            // Columns.
            diff[offset[4] + 0 * 2 + 1] = IDX(i - 2, j);
            diff[offset[4] + 1 * 2 + 1] = IDX(i - 1, j);
            diff[offset[4] + 2 * 2 + 1] = IDX(i, j - 3);
            diff[offset[4] + 3 * 2 + 1] = IDX(i, j - 2);
            diff[offset[4] + 4 * 2 + 1] = IDX(i, j - 1);
            diff[offset[4] + 5 * 2 + 1] = IDX(i, j);
            diff[offset[4] + 6 * 2 + 1] = IDX(i, j + 1);
            diff[offset[4] + 7 * 2 + 1] = IDX(i + 1, j);
            diff[offset[4] + 8 * 2 + 1] = IDX(i + 2, j);
            diff[offset[4] + 9 * 2 + 1] = ctx->dim + IDX(i, j);
            diff[offset[4] + 10 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[4] + 11 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[4] + 12 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
            diff[offset[4] + 13 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[4] + 14 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[4] + 15 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[4] + 16 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[4] + 17 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[4] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
            diff[offset[4] + 19 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
            diff[offset[4] + 20 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
            diff[offset[4] + 21 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
            diff[offset[4] + 22 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 3);
            diff[offset[4] + 23 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 2);
            diff[offset[4] + 24 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 1);
            diff[offset[4] + 25 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 26 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 1);
            diff[offset[4] + 27 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
            diff[offset[4] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i + 2, j);
            diff[offset[4] + 29 * 2 + 1] = 5 * ctx->dim + IDX(i, j);
            diff[offset[4] + 30 * 2 + 1] = GNUM * ctx->dim;
            // Update offset by 31.
            offset[4] += 2 * P4_CS[4];
            // Boundary: 2 points.
            j = ctx->NzTotal - 1;
            diff[offset[4] + 0] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 2] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 3] = GNUM * ctx->dim;

            // 6. lambda: 40 points.
            offset[5] =
                1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4]) * ctx->NrInterior *
                             ctx->NzInterior +
                         (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4]) * ctx->NrInterior +
                         (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4]) * ctx->NzInterior +
                         (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3] + P4_SS[4]) +
                         2 * (ctx->NrInterior + ctx->NzInterior + 3) +
                         (P4_CC[5] * ctx->NzInterior + P4_CS[5]) * (i - ctx->ghost));
            for (j = ctx->ghost; j < ctx->ghost + ctx->NzInterior; ++j)
            {
                // Row indices are all row 5 * dim + IDX(i, j).
                for (k = 0; k < P4_CC[5]; ++k)
                {
                    diff[offset[5] + 2 * k] = 5 * ctx->dim + IDX(i, j);
                }
                // Column indices.
                diff[offset[5] + 0 * 2 + 1] = IDX(i - 2, j);
                diff[offset[5] + 1 * 2 + 1] = IDX(i - 1, j);
                diff[offset[5] + 2 * 2 + 1] = IDX(i, j - 2);
                diff[offset[5] + 3 * 2 + 1] = IDX(i, j - 1);
                diff[offset[5] + 4 * 2 + 1] = IDX(i, j);
                diff[offset[5] + 5 * 2 + 1] = IDX(i, j + 1);
                diff[offset[5] + 6 * 2 + 1] = IDX(i, j + 2);
                diff[offset[5] + 7 * 2 + 1] = IDX(i + 1, j);
                diff[offset[5] + 8 * 2 + 1] = IDX(i + 2, j);
                diff[offset[5] + 9 * 2 + 1] = ctx->dim + IDX(i - 2, j);
                diff[offset[5] + 10 * 2 + 1] = ctx->dim + IDX(i - 1, j);
                diff[offset[5] + 11 * 2 + 1] = ctx->dim + IDX(i, j - 2);
                diff[offset[5] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 1);
                diff[offset[5] + 13 * 2 + 1] = ctx->dim + IDX(i, j + 1);
                diff[offset[5] + 14 * 2 + 1] = ctx->dim + IDX(i, j + 2);
                diff[offset[5] + 15 * 2 + 1] = ctx->dim + IDX(i + 1, j);
                diff[offset[5] + 16 * 2 + 1] = ctx->dim + IDX(i + 2, j);
                diff[offset[5] + 17 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
                diff[offset[5] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[5] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
                diff[offset[5] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[5] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
                diff[offset[5] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[5] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
                diff[offset[5] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[5] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
                diff[offset[5] + 26 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
                diff[offset[5] + 27 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
                diff[offset[5] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
                diff[offset[5] + 29 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
                diff[offset[5] + 30 * 2 + 1] = 4 * ctx->dim + IDX(i + 2, j);
                diff[offset[5] + 31 * 2 + 1] = 5 * ctx->dim + IDX(i - 2, j);
                diff[offset[5] + 32 * 2 + 1] = 5 * ctx->dim + IDX(i - 1, j);
                diff[offset[5] + 33 * 2 + 1] = 5 * ctx->dim + IDX(i, j - 2);
                diff[offset[5] + 34 * 2 + 1] = 5 * ctx->dim + IDX(i, j - 1);
                diff[offset[5] + 35 * 2 + 1] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 36 * 2 + 1] = 5 * ctx->dim + IDX(i, j + 1);
                diff[offset[5] + 37 * 2 + 1] = 5 * ctx->dim + IDX(i, j + 2);
                diff[offset[5] + 38 * 2 + 1] = 5 * ctx->dim + IDX(i + 1, j);
                diff[offset[5] + 39 * 2 + 1] = 5 * ctx->dim + IDX(i + 2, j);
                // Update offset by 40.
                offset[5] += 2 * P4_CC[5];
            }
            // Semi-one-sided: 41 points.
            j = ctx->ghost + ctx->NzInterior;
            // Row indices are all row 5 * dim + IDX(i, j).
            for (k = 0; k < P4_CS[5]; ++k)
            {
                diff[offset[5] + 2 * k] = 5 * ctx->dim + IDX(i, j);
            }
            // Columns.
            diff[offset[5] + 0 * 2 + 1] = IDX(i - 2, j);
            diff[offset[5] + 1 * 2 + 1] = IDX(i - 1, j);
            diff[offset[5] + 2 * 2 + 1] = IDX(i, j - 3);
            diff[offset[5] + 3 * 2 + 1] = IDX(i, j - 2);
            diff[offset[5] + 4 * 2 + 1] = IDX(i, j - 1);
            diff[offset[5] + 5 * 2 + 1] = IDX(i, j);
            diff[offset[5] + 6 * 2 + 1] = IDX(i, j + 1);
            diff[offset[5] + 7 * 2 + 1] = IDX(i + 1, j);
            diff[offset[5] + 8 * 2 + 1] = IDX(i + 2, j);
            diff[offset[5] + 9 * 2 + 1] = 1 * ctx->dim + IDX(i - 2, j);
            diff[offset[5] + 10 * 2 + 1] = 1 * ctx->dim + IDX(i - 1, j);
            diff[offset[5] + 11 * 2 + 1] = 1 * ctx->dim + IDX(i, j - 3);
            diff[offset[5] + 12 * 2 + 1] = 1 * ctx->dim + IDX(i, j - 2);
            diff[offset[5] + 13 * 2 + 1] = 1 * ctx->dim + IDX(i, j - 1);
            diff[offset[5] + 14 * 2 + 1] = 1 * ctx->dim + IDX(i, j);
            diff[offset[5] + 15 * 2 + 1] = 1 * ctx->dim + IDX(i, j + 1);
            diff[offset[5] + 16 * 2 + 1] = 1 * ctx->dim + IDX(i + 1, j);
            diff[offset[5] + 17 * 2 + 1] = 1 * ctx->dim + IDX(i + 2, j);
            diff[offset[5] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[5] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[5] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
            diff[offset[5] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[5] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[5] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[5] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[5] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[5] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 2, j);
            diff[offset[5] + 27 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
            diff[offset[5] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
            diff[offset[5] + 29 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[5] + 30 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
            diff[offset[5] + 31 * 2 + 1] = 4 * ctx->dim + IDX(i + 2, j);
            diff[offset[5] + 32 * 2 + 1] = 5 * ctx->dim + IDX(i - 2, j);
            diff[offset[5] + 33 * 2 + 1] = 5 * ctx->dim + IDX(i - 1, j);
            diff[offset[5] + 34 * 2 + 1] = 5 * ctx->dim + IDX(i, j - 3);
            diff[offset[5] + 35 * 2 + 1] = 5 * ctx->dim + IDX(i, j - 2);
            diff[offset[5] + 36 * 2 + 1] = 5 * ctx->dim + IDX(i, j - 1);
            diff[offset[5] + 37 * 2 + 1] = 5 * ctx->dim + IDX(i, j);
            diff[offset[5] + 38 * 2 + 1] = 5 * ctx->dim + IDX(i, j + 1);
            diff[offset[5] + 39 * 2 + 1] = 5 * ctx->dim + IDX(i + 1, j);
            diff[offset[5] + 40 * 2 + 1] = 5 * ctx->dim + IDX(i + 2, j);
            // Update offset by 41.
            offset[5] += 2 * P4_CS[5];
        }

        // Now next-to-last rho strip.
        i = ctx->ghost + ctx->NrInterior;
#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(j, k, offset)
        for (j = ctx->ghost; j < ctx->ghost + ctx->NzInterior; ++j)
        {
            // 1. log_alpha: 30 points.
            offset[0] = 1 + 2 * (P4_CC[0] * ctx->NrInterior * ctx->NzInterior +
                                 P4_CS[0] * ctx->NrInterior + P4_SC[0] * (j - ctx->ghost));
            // All rows are IDX(i, j).
            for (k = 0; k < P4_SC[0]; ++k)
            {
                diff[offset[0] + 2 * k] = IDX(i, j);
            }
            // Columns.
            diff[offset[0] + 0 * 2 + 1] = IDX(i - 3, j);
            diff[offset[0] + 1 * 2 + 1] = IDX(i - 2, j);
            diff[offset[0] + 2 * 2 + 1] = IDX(i - 1, j);
            diff[offset[0] + 3 * 2 + 1] = IDX(i, j - 2);
            diff[offset[0] + 4 * 2 + 1] = IDX(i, j - 1);
            diff[offset[0] + 5 * 2 + 1] = IDX(i, j);
            diff[offset[0] + 6 * 2 + 1] = IDX(i, j + 1);
            diff[offset[0] + 7 * 2 + 1] = IDX(i, j + 2);
            diff[offset[0] + 8 * 2 + 1] = IDX(i + 1, j);
            diff[offset[0] + 9 * 2 + 1] = ctx->dim + IDX(i - 3, j);
            diff[offset[0] + 10 * 2 + 1] = ctx->dim + IDX(i - 2, j);
            diff[offset[0] + 11 * 2 + 1] = ctx->dim + IDX(i - 1, j);
            diff[offset[0] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 2);
            diff[offset[0] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 1);
            diff[offset[0] + 14 * 2 + 1] = ctx->dim + IDX(i, j);
            diff[offset[0] + 15 * 2 + 1] = ctx->dim + IDX(i, j + 1);
            diff[offset[0] + 16 * 2 + 1] = ctx->dim + IDX(i, j + 2);
            diff[offset[0] + 17 * 2 + 1] = ctx->dim + IDX(i + 1, j);
            diff[offset[0] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
            diff[offset[0] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[0] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[0] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[0] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[0] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[0] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[0] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
            diff[offset[0] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[0] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
            diff[offset[0] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[0] + 29 * 2 + 1] = GNUM * ctx->dim;

            // 2. beta: 30 points.
            offset[1] =
                1 + 2 * ((P4_CC[0] + P4_CC[1]) * ctx->NrInterior * ctx->NzInterior +
                         (P4_CS[0] + P4_CS[1]) * ctx->NrInterior + (P4_SC[0]) * ctx->NzInterior +
                         (P4_SS[0]) + P4_SC[1] * (j - ctx->ghost));
            // All rows are dim + IDX(i, j).
            for (k = 0; k < P4_SC[1]; ++k)
            {
                diff[offset[1] + 2 * k] = ctx->dim + IDX(i, j);
            }
            // Columns.
            diff[offset[1] + 0 * 2 + 1] = IDX(i - 3, j);
            diff[offset[1] + 1 * 2 + 1] = IDX(i - 2, j);
            diff[offset[1] + 2 * 2 + 1] = IDX(i - 1, j);
            diff[offset[1] + 3 * 2 + 1] = IDX(i, j - 2);
            diff[offset[1] + 4 * 2 + 1] = IDX(i, j - 1);
            diff[offset[1] + 5 * 2 + 1] = IDX(i, j);
            diff[offset[1] + 6 * 2 + 1] = IDX(i, j + 1);
            diff[offset[1] + 7 * 2 + 1] = IDX(i, j + 2);
            diff[offset[1] + 8 * 2 + 1] = IDX(i + 1, j);
            diff[offset[1] + 9 * 2 + 1] = ctx->dim + IDX(i - 3, j);
            diff[offset[1] + 10 * 2 + 1] = ctx->dim + IDX(i - 2, j);
            diff[offset[1] + 11 * 2 + 1] = ctx->dim + IDX(i - 1, j);
            diff[offset[1] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 2);
            diff[offset[1] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 1);
            diff[offset[1] + 14 * 2 + 1] = ctx->dim + IDX(i, j);
            diff[offset[1] + 15 * 2 + 1] = ctx->dim + IDX(i, j + 1);
            diff[offset[1] + 16 * 2 + 1] = ctx->dim + IDX(i, j + 2);
            diff[offset[1] + 17 * 2 + 1] = ctx->dim + IDX(i + 1, j);
            diff[offset[1] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
            diff[offset[1] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[1] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[1] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[1] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[1] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[1] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[1] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
            diff[offset[1] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[1] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
            diff[offset[1] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[1] + 29 * 2 + 1] = GNUM * ctx->dim;

            // 3. log_h: 29 points.
            offset[2] =
                1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2]) * ctx->NrInterior * ctx->NzInterior +
                         (P4_CS[0] + P4_CS[1] + P4_CS[2]) * ctx->NrInterior +
                         (P4_SC[0] + P4_SC[1]) * ctx->NzInterior + (P4_SS[0] + P4_SS[1]) +
                         P4_SC[2] * (j - ctx->ghost));
            // All rows are 2 * dim + IDX(i, j).
            for (k = 0; k < P4_SC[2]; ++k)
            {
                diff[offset[2] + 2 * k] = 2 * ctx->dim + IDX(i, j);
            }
            // Columns.
            diff[offset[2] + 0 * 2 + 1] = IDX(i - 3, j);
            diff[offset[2] + 1 * 2 + 1] = IDX(i - 2, j);
            diff[offset[2] + 2 * 2 + 1] = IDX(i - 1, j);
            diff[offset[2] + 3 * 2 + 1] = IDX(i, j - 2);
            diff[offset[2] + 4 * 2 + 1] = IDX(i, j - 1);
            diff[offset[2] + 5 * 2 + 1] = IDX(i, j);
            diff[offset[2] + 6 * 2 + 1] = IDX(i, j + 1);
            diff[offset[2] + 7 * 2 + 1] = IDX(i, j + 2);
            diff[offset[2] + 8 * 2 + 1] = IDX(i + 1, j);
            diff[offset[2] + 9 * 2 + 1] = ctx->dim + IDX(i - 3, j);
            diff[offset[2] + 10 * 2 + 1] = ctx->dim + IDX(i - 2, j);
            diff[offset[2] + 11 * 2 + 1] = ctx->dim + IDX(i - 1, j);
            diff[offset[2] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 2);
            diff[offset[2] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 1);
            diff[offset[2] + 14 * 2 + 1] = ctx->dim + IDX(i, j);
            diff[offset[2] + 15 * 2 + 1] = ctx->dim + IDX(i, j + 1);
            diff[offset[2] + 16 * 2 + 1] = ctx->dim + IDX(i, j + 2);
            diff[offset[2] + 17 * 2 + 1] = ctx->dim + IDX(i + 1, j);
            diff[offset[2] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
            diff[offset[2] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[2] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[2] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[2] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[2] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[2] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[2] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
            diff[offset[2] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[2] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
            diff[offset[2] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i, j);

            // 4. log_h: 38 points.
            offset[3] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3]) * ctx->NrInterior *
                                     ctx->NzInterior +
                                 (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3]) * ctx->NrInterior +
                                 (P4_SC[0] + P4_SC[1] + P4_SC[2]) * ctx->NzInterior +
                                 (P4_SS[0] + P4_SS[1] + P4_SS[2]) + P4_SC[3] * (j - ctx->ghost));
            // All rows are 3 * dim + IDX(i, j).
            for (k = 0; k < P4_SC[3]; ++k)
            {
                diff[offset[3] + 2 * k] = 3 * ctx->dim + IDX(i, j);
            }
            // Columns.
            diff[offset[3] + 0 * 2 + 1] = IDX(i - 3, j);
            diff[offset[3] + 1 * 2 + 1] = IDX(i - 2, j);
            diff[offset[3] + 2 * 2 + 1] = IDX(i - 1, j);
            diff[offset[3] + 3 * 2 + 1] = IDX(i, j - 2);
            diff[offset[3] + 4 * 2 + 1] = IDX(i, j - 1);
            diff[offset[3] + 5 * 2 + 1] = IDX(i, j);
            diff[offset[3] + 6 * 2 + 1] = IDX(i, j + 1);
            diff[offset[3] + 7 * 2 + 1] = IDX(i, j + 2);
            diff[offset[3] + 8 * 2 + 1] = IDX(i + 1, j);
            diff[offset[3] + 9 * 2 + 1] = ctx->dim + IDX(i - 3, j);
            diff[offset[3] + 10 * 2 + 1] = ctx->dim + IDX(i - 2, j);
            diff[offset[3] + 11 * 2 + 1] = ctx->dim + IDX(i - 1, j);
            diff[offset[3] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 2);
            diff[offset[3] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 1);
            diff[offset[3] + 14 * 2 + 1] = ctx->dim + IDX(i, j);
            diff[offset[3] + 15 * 2 + 1] = ctx->dim + IDX(i, j + 1);
            diff[offset[3] + 16 * 2 + 1] = ctx->dim + IDX(i, j + 2);
            diff[offset[3] + 17 * 2 + 1] = ctx->dim + IDX(i + 1, j);
            diff[offset[3] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
            diff[offset[3] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[3] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[3] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[3] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[3] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[3] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[3] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
            diff[offset[3] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[3] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
            diff[offset[3] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i - 3, j);
            diff[offset[3] + 29 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
            diff[offset[3] + 30 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
            diff[offset[3] + 31 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 2);
            diff[offset[3] + 32 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 1);
            diff[offset[3] + 33 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[3] + 34 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 1);
            diff[offset[3] + 35 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 2);
            diff[offset[3] + 36 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
            diff[offset[3] + 37 * 2 + 1] = GNUM * ctx->dim;

            // 5. psi: 31 points.
            offset[4] =
                1 +
                2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4]) * ctx->NrInterior *
                         ctx->NzInterior +
                     (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + 2) * ctx->NrInterior +
                     (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3]) * ctx->NzInterior +
                     (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3]) + P4_SC[4] * (j - ctx->ghost));
            // All rows are 4 * dim + IDX(i, j).
            for (k = 0; k < P4_SC[4]; ++k)
            {
                diff[offset[4] + 2 * k] = 4 * ctx->dim + IDX(i, j);
            }
            // Columns.
            diff[offset[4] + 0 * 2 + 1] = IDX(i - 3, j);
            diff[offset[4] + 1 * 2 + 1] = IDX(i - 2, j);
            diff[offset[4] + 2 * 2 + 1] = IDX(i - 1, j);
            diff[offset[4] + 3 * 2 + 1] = IDX(i, j - 2);
            diff[offset[4] + 4 * 2 + 1] = IDX(i, j - 1);
            diff[offset[4] + 5 * 2 + 1] = IDX(i, j);
            diff[offset[4] + 6 * 2 + 1] = IDX(i, j + 1);
            diff[offset[4] + 7 * 2 + 1] = IDX(i, j + 2);
            diff[offset[4] + 8 * 2 + 1] = IDX(i + 1, j);
            diff[offset[4] + 9 * 2 + 1] = ctx->dim + IDX(i, j);
            diff[offset[4] + 10 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
            diff[offset[4] + 11 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[4] + 12 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[4] + 13 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[4] + 14 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[4] + 15 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[4] + 16 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[4] + 17 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
            diff[offset[4] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[4] + 19 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
            diff[offset[4] + 20 * 2 + 1] = 4 * ctx->dim + IDX(i - 3, j);
            diff[offset[4] + 21 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
            diff[offset[4] + 22 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
            diff[offset[4] + 23 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 2);
            diff[offset[4] + 24 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 1);
            diff[offset[4] + 25 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 26 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 1);
            diff[offset[4] + 27 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 2);
            diff[offset[4] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
            diff[offset[4] + 29 * 2 + 1] = 5 * ctx->dim + IDX(i, j);
            diff[offset[4] + 30 * 2 + 1] = GNUM * ctx->dim;

            // 6. lambda: 43 points.
            offset[5] =
                1 +
                2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4] + P4_CC[5]) *
                         ctx->NrInterior * ctx->NzInterior +
                     (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + 2 + P4_CS[5]) *
                         ctx->NrInterior +
                     (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4] + 2) * ctx->NzInterior +
                     (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3] + P4_SS[4] + 6) +
                     P4_SC[5] * (j - ctx->ghost));
            // All rows are 5 * dim + IDX(i, j).
            for (k = 0; k < P4_SC[5]; ++k)
            {
                diff[offset[5] + 2 * k] = 5 * ctx->dim + IDX(i, j);
            }
            // Columns.
            diff[offset[5] + 0 * 2 + 1] = IDX(i - 4, j);
            diff[offset[5] + 1 * 2 + 1] = IDX(i - 3, j);
            diff[offset[5] + 2 * 2 + 1] = IDX(i - 2, j);
            diff[offset[5] + 3 * 2 + 1] = IDX(i - 1, j);
            diff[offset[5] + 4 * 2 + 1] = IDX(i, j - 2);
            diff[offset[5] + 5 * 2 + 1] = IDX(i, j - 1);
            diff[offset[5] + 6 * 2 + 1] = IDX(i, j);
            diff[offset[5] + 7 * 2 + 1] = IDX(i, j + 1);
            diff[offset[5] + 8 * 2 + 1] = IDX(i, j + 2);
            diff[offset[5] + 9 * 2 + 1] = IDX(i + 1, j);
            diff[offset[5] + 10 * 2 + 1] = 1 * ctx->dim + IDX(i - 3, j);
            diff[offset[5] + 11 * 2 + 1] = 1 * ctx->dim + IDX(i - 2, j);
            diff[offset[5] + 12 * 2 + 1] = 1 * ctx->dim + IDX(i - 1, j);
            diff[offset[5] + 13 * 2 + 1] = 1 * ctx->dim + IDX(i, j - 2);
            diff[offset[5] + 14 * 2 + 1] = 1 * ctx->dim + IDX(i, j - 1);
            diff[offset[5] + 15 * 2 + 1] = 1 * ctx->dim + IDX(i, j);
            diff[offset[5] + 16 * 2 + 1] = 1 * ctx->dim + IDX(i, j + 1);
            diff[offset[5] + 17 * 2 + 1] = 1 * ctx->dim + IDX(i, j + 2);
            diff[offset[5] + 18 * 2 + 1] = 1 * ctx->dim + IDX(i + 1, j);
            diff[offset[5] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 4, j);
            diff[offset[5] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
            diff[offset[5] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
            diff[offset[5] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
            diff[offset[5] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
            diff[offset[5] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
            diff[offset[5] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
            diff[offset[5] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
            diff[offset[5] + 27 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 2);
            diff[offset[5] + 28 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
            diff[offset[5] + 29 * 2 + 1] = 4 * ctx->dim + IDX(i - 3, j);
            diff[offset[5] + 30 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
            diff[offset[5] + 31 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
            diff[offset[5] + 32 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[5] + 33 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
            diff[offset[5] + 34 * 2 + 1] = 5 * ctx->dim + IDX(i - 3, j);
            diff[offset[5] + 35 * 2 + 1] = 5 * ctx->dim + IDX(i - 2, j);
            diff[offset[5] + 36 * 2 + 1] = 5 * ctx->dim + IDX(i - 1, j);
            diff[offset[5] + 37 * 2 + 1] = 5 * ctx->dim + IDX(i, j - 2);
            diff[offset[5] + 38 * 2 + 1] = 5 * ctx->dim + IDX(i, j - 1);
            diff[offset[5] + 39 * 2 + 1] = 5 * ctx->dim + IDX(i, j);
            diff[offset[5] + 40 * 2 + 1] = 5 * ctx->dim + IDX(i, j + 1);
            diff[offset[5] + 41 * 2 + 1] = 5 * ctx->dim + IDX(i, j + 2);
            diff[offset[5] + 42 * 2 + 1] = 5 * ctx->dim + IDX(i + 1, j);
        }

        // Corner.
        j = ctx->ghost + ctx->NzInterior;
        // 1. log_alpha: 30 points.
        offset[0] = 1 + 2 * (P4_CC[0] * ctx->NrInterior * ctx->NzInterior +
                             P4_CS[0] * ctx->NrInterior + P4_SC[0] * ctx->NzInterior);
        // All rows are IDX(i, j).
        for (k = 0; k < P4_SS[0]; ++k)
        {
            diff[offset[0] + 2 * k] = IDX(i, j);
        }
        // Columns.
        diff[offset[0] + 0 * 2 + 1] = IDX(i - 3, j);
        diff[offset[0] + 1 * 2 + 1] = IDX(i - 2, j);
        diff[offset[0] + 2 * 2 + 1] = IDX(i - 1, j);
        diff[offset[0] + 3 * 2 + 1] = IDX(i, j - 3);
        diff[offset[0] + 4 * 2 + 1] = IDX(i, j - 2);
        diff[offset[0] + 5 * 2 + 1] = IDX(i, j - 1);
        diff[offset[0] + 6 * 2 + 1] = IDX(i, j);
        diff[offset[0] + 7 * 2 + 1] = IDX(i, j + 1);
        diff[offset[0] + 8 * 2 + 1] = IDX(i + 1, j);
        diff[offset[0] + 9 * 2 + 1] = ctx->dim + IDX(i - 3, j);
        diff[offset[0] + 10 * 2 + 1] = ctx->dim + IDX(i - 2, j);
        diff[offset[0] + 11 * 2 + 1] = ctx->dim + IDX(i - 1, j);
        diff[offset[0] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 3);
        diff[offset[0] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 2);
        diff[offset[0] + 14 * 2 + 1] = ctx->dim + IDX(i, j - 1);
        diff[offset[0] + 15 * 2 + 1] = ctx->dim + IDX(i, j);
        diff[offset[0] + 16 * 2 + 1] = ctx->dim + IDX(i, j + 1);
        diff[offset[0] + 17 * 2 + 1] = ctx->dim + IDX(i + 1, j);
        diff[offset[0] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
        diff[offset[0] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
        diff[offset[0] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
        diff[offset[0] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
        diff[offset[0] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
        diff[offset[0] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
        diff[offset[0] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
        diff[offset[0] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
        diff[offset[0] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
        diff[offset[0] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
        diff[offset[0] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
        diff[offset[0] + 29 * 2 + 1] = GNUM * ctx->dim;

        // 2. beta: 30 points.
        offset[1] = 1 + 2 * ((P4_CC[0] + P4_CC[1]) * ctx->NrInterior * ctx->NzInterior +
                             (P4_CS[0] + P4_CS[1]) * ctx->NrInterior +
                             (P4_SC[0] + P4_SC[1]) * ctx->NzInterior + (P4_SS[0]));
        // All rows are dim + IDX(i, j).
        for (k = 0; k < P4_SS[1]; ++k)
        {
            diff[offset[1] + 2 * k] = ctx->dim + IDX(i, j);
        }
        // Columns.
        diff[offset[1] + 0 * 2 + 1] = IDX(i - 3, j);
        diff[offset[1] + 1 * 2 + 1] = IDX(i - 2, j);
        diff[offset[1] + 2 * 2 + 1] = IDX(i - 1, j);
        diff[offset[1] + 3 * 2 + 1] = IDX(i, j - 3);
        diff[offset[1] + 4 * 2 + 1] = IDX(i, j - 2);
        diff[offset[1] + 5 * 2 + 1] = IDX(i, j - 1);
        diff[offset[1] + 6 * 2 + 1] = IDX(i, j);
        diff[offset[1] + 7 * 2 + 1] = IDX(i, j + 1);
        diff[offset[1] + 8 * 2 + 1] = IDX(i + 1, j);
        diff[offset[1] + 9 * 2 + 1] = ctx->dim + IDX(i - 3, j);
        diff[offset[1] + 10 * 2 + 1] = ctx->dim + IDX(i - 2, j);
        diff[offset[1] + 11 * 2 + 1] = ctx->dim + IDX(i - 1, j);
        diff[offset[1] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 3);
        diff[offset[1] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 2);
        diff[offset[1] + 14 * 2 + 1] = ctx->dim + IDX(i, j - 1);
        diff[offset[1] + 15 * 2 + 1] = ctx->dim + IDX(i, j);
        diff[offset[1] + 16 * 2 + 1] = ctx->dim + IDX(i, j + 1);
        diff[offset[1] + 17 * 2 + 1] = ctx->dim + IDX(i + 1, j);
        diff[offset[1] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
        diff[offset[1] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
        diff[offset[1] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
        diff[offset[1] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
        diff[offset[1] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
        diff[offset[1] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
        diff[offset[1] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
        diff[offset[1] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
        diff[offset[1] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
        diff[offset[1] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
        diff[offset[1] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
        diff[offset[1] + 29 * 2 + 1] = GNUM * ctx->dim;

        // 3. log_h: 29 points.
        offset[2] =
            1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2]) * ctx->NrInterior * ctx->NzInterior +
                     (P4_CS[0] + P4_CS[1] + P4_CS[2]) * ctx->NrInterior +
                     (P4_SC[0] + P4_SC[1] + P4_SC[2]) * ctx->NzInterior + (P4_SS[0] + P4_SS[1]));
        // All rows are 2 * dim + IDX(i, j).
        for (k = 0; k < P4_SS[2]; ++k)
        {
            diff[offset[2] + 2 * k] = 2 * ctx->dim + IDX(i, j);
        }
        // Columns.
        diff[offset[2] + 0 * 2 + 1] = IDX(i - 3, j);
        diff[offset[2] + 1 * 2 + 1] = IDX(i - 2, j);
        diff[offset[2] + 2 * 2 + 1] = IDX(i - 1, j);
        diff[offset[2] + 3 * 2 + 1] = IDX(i, j - 3);
        diff[offset[2] + 4 * 2 + 1] = IDX(i, j - 2);
        diff[offset[2] + 5 * 2 + 1] = IDX(i, j - 1);
        diff[offset[2] + 6 * 2 + 1] = IDX(i, j);
        diff[offset[2] + 7 * 2 + 1] = IDX(i, j + 1);
        diff[offset[2] + 8 * 2 + 1] = IDX(i + 1, j);
        diff[offset[2] + 9 * 2 + 1] = ctx->dim + IDX(i - 3, j);
        diff[offset[2] + 10 * 2 + 1] = ctx->dim + IDX(i - 2, j);
        diff[offset[2] + 11 * 2 + 1] = ctx->dim + IDX(i - 1, j);
        diff[offset[2] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 3);
        diff[offset[2] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 2);
        diff[offset[2] + 14 * 2 + 1] = ctx->dim + IDX(i, j - 1);
        diff[offset[2] + 15 * 2 + 1] = ctx->dim + IDX(i, j);
        diff[offset[2] + 16 * 2 + 1] = ctx->dim + IDX(i, j + 1);
        diff[offset[2] + 17 * 2 + 1] = ctx->dim + IDX(i + 1, j);
        diff[offset[2] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
        diff[offset[2] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
        diff[offset[2] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
        diff[offset[2] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
        diff[offset[2] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
        diff[offset[2] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
        diff[offset[2] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
        diff[offset[2] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
        diff[offset[2] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
        diff[offset[2] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
        diff[offset[2] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i, j);

        // 4. log_h: 38 points.
        offset[3] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3]) * ctx->NrInterior *
                                 ctx->NzInterior +
                             (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3]) * ctx->NrInterior +
                             (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3]) * ctx->NzInterior +
                             (P4_SS[0] + P4_SS[1] + P4_SS[2]));
        // All rows are 3 * dim + IDX(i, j).
        for (k = 0; k < P4_SS[3]; ++k)
        {
            diff[offset[3] + 2 * k] = 3 * ctx->dim + IDX(i, j);
        }
        // Columns.
        diff[offset[3] + 0 * 2 + 1] = IDX(i - 3, j);
        diff[offset[3] + 1 * 2 + 1] = IDX(i - 2, j);
        diff[offset[3] + 2 * 2 + 1] = IDX(i - 1, j);
        diff[offset[3] + 3 * 2 + 1] = IDX(i, j - 3);
        diff[offset[3] + 4 * 2 + 1] = IDX(i, j - 2);
        diff[offset[3] + 5 * 2 + 1] = IDX(i, j - 1);
        diff[offset[3] + 6 * 2 + 1] = IDX(i, j);
        diff[offset[3] + 7 * 2 + 1] = IDX(i, j + 1);
        diff[offset[3] + 8 * 2 + 1] = IDX(i + 1, j);
        diff[offset[3] + 9 * 2 + 1] = ctx->dim + IDX(i - 3, j);
        diff[offset[3] + 10 * 2 + 1] = ctx->dim + IDX(i - 2, j);
        diff[offset[3] + 11 * 2 + 1] = ctx->dim + IDX(i - 1, j);
        diff[offset[3] + 12 * 2 + 1] = ctx->dim + IDX(i, j - 3);
        diff[offset[3] + 13 * 2 + 1] = ctx->dim + IDX(i, j - 2);
        diff[offset[3] + 14 * 2 + 1] = ctx->dim + IDX(i, j - 1);
        diff[offset[3] + 15 * 2 + 1] = ctx->dim + IDX(i, j);
        diff[offset[3] + 16 * 2 + 1] = ctx->dim + IDX(i, j + 1);
        diff[offset[3] + 17 * 2 + 1] = ctx->dim + IDX(i + 1, j);
        diff[offset[3] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
        diff[offset[3] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
        diff[offset[3] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
        diff[offset[3] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
        diff[offset[3] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
        diff[offset[3] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
        diff[offset[3] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
        diff[offset[3] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
        diff[offset[3] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
        diff[offset[3] + 27 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
        diff[offset[3] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i - 3, j);
        diff[offset[3] + 29 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
        diff[offset[3] + 30 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
        diff[offset[3] + 31 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 3);
        diff[offset[3] + 32 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 2);
        diff[offset[3] + 33 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 1);
        diff[offset[3] + 34 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
        diff[offset[3] + 35 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 1);
        diff[offset[3] + 36 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
        diff[offset[3] + 37 * 2 + 1] = GNUM * ctx->dim;

        // 5. psi: 31 points.
        offset[4] =
            1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4]) * ctx->NrInterior *
                         ctx->NzInterior +
                     (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + 2) * ctx->NrInterior +
                     (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4]) * ctx->NzInterior +
                     (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3]));
        // All rows are 4 * dim + IDX(i, j).
        for (k = 0; k < P4_SS[4]; ++k)
        {
            diff[offset[4] + 2 * k] = 4 * ctx->dim + IDX(i, j);
        }
        // Columns.
        diff[offset[4] + 0 * 2 + 1] = IDX(i - 3, j);
        diff[offset[4] + 1 * 2 + 1] = IDX(i - 2, j);
        diff[offset[4] + 2 * 2 + 1] = IDX(i - 1, j);
        diff[offset[4] + 3 * 2 + 1] = IDX(i, j - 3);
        diff[offset[4] + 4 * 2 + 1] = IDX(i, j - 2);
        diff[offset[4] + 5 * 2 + 1] = IDX(i, j - 1);
        diff[offset[4] + 6 * 2 + 1] = IDX(i, j);
        diff[offset[4] + 7 * 2 + 1] = IDX(i, j + 1);
        diff[offset[4] + 8 * 2 + 1] = IDX(i + 1, j);
        diff[offset[4] + 9 * 2 + 1] = ctx->dim + IDX(i, j);
        diff[offset[4] + 10 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
        diff[offset[4] + 11 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
        diff[offset[4] + 12 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
        diff[offset[4] + 13 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
        diff[offset[4] + 14 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
        diff[offset[4] + 15 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
        diff[offset[4] + 16 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
        diff[offset[4] + 17 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
        diff[offset[4] + 18 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
        diff[offset[4] + 19 * 2 + 1] = 3 * ctx->dim + IDX(i, j);
        diff[offset[4] + 20 * 2 + 1] = 4 * ctx->dim + IDX(i - 3, j);
        diff[offset[4] + 21 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
        diff[offset[4] + 22 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
        diff[offset[4] + 23 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 3);
        diff[offset[4] + 24 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 2);
        diff[offset[4] + 25 * 2 + 1] = 4 * ctx->dim + IDX(i, j - 1);
        diff[offset[4] + 26 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
        diff[offset[4] + 27 * 2 + 1] = 4 * ctx->dim + IDX(i, j + 1);
        diff[offset[4] + 28 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
        diff[offset[4] + 29 * 2 + 1] = 5 * ctx->dim + IDX(i, j);
        diff[offset[4] + 30 * 2 + 1] = GNUM * ctx->dim;

        // Phi boundary: i = NrTotal - 2, j = NzTotal - 1.
        offset[4] += 2 * P4_SS[4];
        j = ctx->NzTotal - 1;
        diff[offset[4] + 0] = 4 * ctx->dim + IDX(i, j);
        diff[offset[4] + 2] = 4 * ctx->dim + IDX(i, j);
        diff[offset[4] + 1] = 4 * ctx->dim + IDX(i, j);
        diff[offset[4] + 3] = GNUM * ctx->dim;
        offset[4] += 4;

        // 6. lambda: 43 points.
        offset[5] = 1 + 2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4] + P4_CC[5]) *
                                 ctx->NrInterior * ctx->NzInterior +
                             (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + 2 + P4_CS[5]) *
                                 ctx->NrInterior +
                             (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4] + 2 + P4_SC[5]) *
                                 ctx->NzInterior +
                             (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3] + P4_SS[4] + 6));
        // All rows are 5 * dim + IDX(i, j).
        for (k = 0; k < P4_SS[5]; ++k)
        {
            diff[offset[5] + 2 * k] = 5 * ctx->dim + IDX(i, j);
        }
        // Columns.
        diff[offset[5] + 0 * 2 + 1] = IDX(i - 4, j);
        diff[offset[5] + 1 * 2 + 1] = IDX(i - 3, j);
        diff[offset[5] + 2 * 2 + 1] = IDX(i - 2, j);
        diff[offset[5] + 3 * 2 + 1] = IDX(i - 1, j);
        diff[offset[5] + 4 * 2 + 1] = IDX(i, j - 3);
        diff[offset[5] + 5 * 2 + 1] = IDX(i, j - 2);
        diff[offset[5] + 6 * 2 + 1] = IDX(i, j - 1);
        diff[offset[5] + 7 * 2 + 1] = IDX(i, j);
        diff[offset[5] + 8 * 2 + 1] = IDX(i, j + 1);
        diff[offset[5] + 9 * 2 + 1] = IDX(i + 1, j);
        diff[offset[5] + 10 * 2 + 1] = 1 * ctx->dim + IDX(i - 3, j);
        diff[offset[5] + 11 * 2 + 1] = 1 * ctx->dim + IDX(i - 2, j);
        diff[offset[5] + 12 * 2 + 1] = 1 * ctx->dim + IDX(i - 1, j);
        diff[offset[5] + 13 * 2 + 1] = 1 * ctx->dim + IDX(i, j - 3);
        diff[offset[5] + 14 * 2 + 1] = 1 * ctx->dim + IDX(i, j - 2);
        diff[offset[5] + 15 * 2 + 1] = 1 * ctx->dim + IDX(i, j - 1);
        diff[offset[5] + 16 * 2 + 1] = 1 * ctx->dim + IDX(i, j);
        diff[offset[5] + 17 * 2 + 1] = 1 * ctx->dim + IDX(i, j + 1);
        diff[offset[5] + 18 * 2 + 1] = 1 * ctx->dim + IDX(i + 1, j);
        diff[offset[5] + 19 * 2 + 1] = 2 * ctx->dim + IDX(i - 4, j);
        diff[offset[5] + 20 * 2 + 1] = 2 * ctx->dim + IDX(i - 3, j);
        diff[offset[5] + 21 * 2 + 1] = 2 * ctx->dim + IDX(i - 2, j);
        diff[offset[5] + 22 * 2 + 1] = 2 * ctx->dim + IDX(i - 1, j);
        diff[offset[5] + 23 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 3);
        diff[offset[5] + 24 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 2);
        diff[offset[5] + 25 * 2 + 1] = 2 * ctx->dim + IDX(i, j - 1);
        diff[offset[5] + 26 * 2 + 1] = 2 * ctx->dim + IDX(i, j);
        diff[offset[5] + 27 * 2 + 1] = 2 * ctx->dim + IDX(i, j + 1);
        diff[offset[5] + 28 * 2 + 1] = 2 * ctx->dim + IDX(i + 1, j);
        diff[offset[5] + 29 * 2 + 1] = 4 * ctx->dim + IDX(i - 3, j);
        diff[offset[5] + 30 * 2 + 1] = 4 * ctx->dim + IDX(i - 2, j);
        diff[offset[5] + 31 * 2 + 1] = 4 * ctx->dim + IDX(i - 1, j);
        diff[offset[5] + 32 * 2 + 1] = 4 * ctx->dim + IDX(i, j);
        diff[offset[5] + 33 * 2 + 1] = 4 * ctx->dim + IDX(i + 1, j);
        diff[offset[5] + 34 * 2 + 1] = 5 * ctx->dim + IDX(i - 3, j);
        diff[offset[5] + 35 * 2 + 1] = 5 * ctx->dim + IDX(i - 2, j);
        diff[offset[5] + 36 * 2 + 1] = 5 * ctx->dim + IDX(i - 1, j);
        diff[offset[5] + 37 * 2 + 1] = 5 * ctx->dim + IDX(i, j - 3);
        diff[offset[5] + 38 * 2 + 1] = 5 * ctx->dim + IDX(i, j - 2);
        diff[offset[5] + 39 * 2 + 1] = 5 * ctx->dim + IDX(i, j - 1);
        diff[offset[5] + 40 * 2 + 1] = 5 * ctx->dim + IDX(i, j);
        diff[offset[5] + 41 * 2 + 1] = 5 * ctx->dim + IDX(i, j + 1);
        diff[offset[5] + 42 * 2 + 1] = 5 * ctx->dim + IDX(i + 1, j);

        // Last boundary points.
        i = ctx->NrTotal - 1;
#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(j, offset)
        for (j = ctx->ghost; j < ctx->NzTotal; ++j)
        {
            // 5. psi: 2 points.
            offset[4] =
                1 +
                2 * ((P4_CC[0] + P4_CC[1] + P4_CC[2] + P4_CC[3] + P4_CC[4]) * ctx->NrInterior *
                         ctx->NzInterior +
                     (P4_CS[0] + P4_CS[1] + P4_CS[2] + P4_CS[3] + P4_CS[4] + 2) * ctx->NrInterior +
                     (P4_SC[0] + P4_SC[1] + P4_SC[2] + P4_SC[3] + P4_SC[4]) * ctx->NzInterior +
                     (P4_SS[0] + P4_SS[1] + P4_SS[2] + P4_SS[3] + P4_SS[4] + 2) +
                     2 * (j - ctx->ghost));

            diff[offset[4] + 0] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 2] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 3] = GNUM * ctx->dim;
        }
    }
    else
    {
// Interior points.
#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(i, j, offset)
        for (i = ctx->ghost; i < ctx->NrInterior + ctx->ghost; i++)
        {
            // log_alpha: 18 different points.
            offset[0] = 1 + 2 * P2_CC[0] * ctx->NzInterior * (i - ctx->ghost);
            for (j = ctx->ghost; j < ctx->NzInterior + ctx->ghost; j++)
            {
                // Row indices are all row IDX(i, j).
                diff[offset[0] + 0] = IDX(i, j);
                diff[offset[0] + 2] = IDX(i, j);
                diff[offset[0] + 4] = IDX(i, j);
                diff[offset[0] + 6] = IDX(i, j);
                diff[offset[0] + 8] = IDX(i, j);
                diff[offset[0] + 10] = IDX(i, j);
                diff[offset[0] + 12] = IDX(i, j);
                diff[offset[0] + 14] = IDX(i, j);
                diff[offset[0] + 16] = IDX(i, j);
                diff[offset[0] + 18] = IDX(i, j);
                diff[offset[0] + 20] = IDX(i, j);
                diff[offset[0] + 22] = IDX(i, j);
                diff[offset[0] + 24] = IDX(i, j);
                diff[offset[0] + 26] = IDX(i, j);
                diff[offset[0] + 28] = IDX(i, j);
                diff[offset[0] + 30] = IDX(i, j);
                diff[offset[0] + 32] = IDX(i, j);
                diff[offset[0] + 34] = IDX(i, j);
                // Column indices.
                diff[offset[0] + 1] = IDX(i - 1, j);
                diff[offset[0] + 3] = IDX(i, j - 1);
                diff[offset[0] + 5] = IDX(i, j);
                diff[offset[0] + 7] = IDX(i, j + 1);
                diff[offset[0] + 9] = IDX(i + 1, j);
                diff[offset[0] + 11] = ctx->dim + IDX(i - 1, j);
                diff[offset[0] + 13] = ctx->dim + IDX(i, j - 1);
                diff[offset[0] + 15] = ctx->dim + IDX(i, j);
                diff[offset[0] + 17] = ctx->dim + IDX(i, j + 1);
                diff[offset[0] + 19] = ctx->dim + IDX(i + 1, j);
                diff[offset[0] + 21] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[0] + 23] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[0] + 25] = 2 * ctx->dim + IDX(i, j);
                diff[offset[0] + 27] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[0] + 29] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[0] + 31] = 3 * ctx->dim + IDX(i, j);
                diff[offset[0] + 33] = 4 * ctx->dim + IDX(i, j);
                diff[offset[0] + 35] = GNUM * ctx->dim;
                offset[0] += 2 * P2_CC[0];
            }

            // beta: 17 different points.
            offset[1] = 1 + 2 * (P2_CC[0]) * ctx->NrInterior * ctx->NzInterior +
                        2 * P2_CC[1] * ctx->NzInterior * (i - ctx->ghost);
            for (j = ctx->ghost; j < ctx->NzInterior + ctx->ghost; j++)
            {
                // Row indices are all row dim + IDX(i, j).
                diff[offset[1] + 0] = ctx->dim + IDX(i, j);
                diff[offset[1] + 2] = ctx->dim + IDX(i, j);
                diff[offset[1] + 4] = ctx->dim + IDX(i, j);
                diff[offset[1] + 6] = ctx->dim + IDX(i, j);
                diff[offset[1] + 8] = ctx->dim + IDX(i, j);
                diff[offset[1] + 10] = ctx->dim + IDX(i, j);
                diff[offset[1] + 12] = ctx->dim + IDX(i, j);
                diff[offset[1] + 14] = ctx->dim + IDX(i, j);
                diff[offset[1] + 16] = ctx->dim + IDX(i, j);
                diff[offset[1] + 18] = ctx->dim + IDX(i, j);
                diff[offset[1] + 20] = ctx->dim + IDX(i, j);
                diff[offset[1] + 22] = ctx->dim + IDX(i, j);
                diff[offset[1] + 24] = ctx->dim + IDX(i, j);
                diff[offset[1] + 26] = ctx->dim + IDX(i, j);
                diff[offset[1] + 28] = ctx->dim + IDX(i, j);
                diff[offset[1] + 30] = ctx->dim + IDX(i, j);
                diff[offset[1] + 32] = ctx->dim + IDX(i, j);
                // Column indices.
                diff[offset[1] + 1] = IDX(i - 1, j);
                diff[offset[1] + 3] = IDX(i, j - 1);
                diff[offset[1] + 5] = IDX(i, j + 1);
                diff[offset[1] + 7] = IDX(i + 1, j);
                diff[offset[1] + 9] = ctx->dim + IDX(i - 1, j);
                diff[offset[1] + 11] = ctx->dim + IDX(i, j - 1);
                diff[offset[1] + 13] = ctx->dim + IDX(i, j);
                diff[offset[1] + 15] = ctx->dim + IDX(i, j + 1);
                diff[offset[1] + 17] = ctx->dim + IDX(i + 1, j);
                diff[offset[1] + 19] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[1] + 21] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[1] + 23] = 2 * ctx->dim + IDX(i, j);
                diff[offset[1] + 25] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[1] + 27] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[1] + 29] = 3 * ctx->dim + IDX(i, j);
                diff[offset[1] + 31] = 4 * ctx->dim + IDX(i, j);
                diff[offset[1] + 33] = GNUM * ctx->dim;
                offset[1] += 2 * P2_CC[1];
            }

            // log_h: 16 different points.
            offset[2] = 1 + 2 * (P2_CC[0] + P2_CC[1]) * ctx->NrInterior * ctx->NzInterior +
                        2 * P2_CC[2] * ctx->NzInterior * (i - ctx->ghost);
            for (j = ctx->ghost; j < ctx->NzInterior + ctx->ghost; j++)
            {
                // Row indices are all row 2 * dim + IDX(i, j).
                diff[offset[2] + 0] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 2] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 4] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 6] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 8] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 10] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 12] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 14] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 16] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 18] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 20] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 22] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 24] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 26] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 28] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 30] = 2 * ctx->dim + IDX(i, j);
                // Column indices.
                diff[offset[2] + 1] = IDX(i - 1, j);
                diff[offset[2] + 3] = IDX(i, j - 1);
                diff[offset[2] + 5] = IDX(i, j);
                diff[offset[2] + 7] = IDX(i, j + 1);
                diff[offset[2] + 9] = IDX(i + 1, j);
                diff[offset[2] + 11] = ctx->dim + IDX(i - 1, j);
                diff[offset[2] + 13] = ctx->dim + IDX(i, j - 1);
                diff[offset[2] + 15] = ctx->dim + IDX(i, j + 1);
                diff[offset[2] + 17] = ctx->dim + IDX(i + 1, j);
                diff[offset[2] + 19] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[2] + 21] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[2] + 23] = 2 * ctx->dim + IDX(i, j);
                diff[offset[2] + 25] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[2] + 27] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[2] + 29] = 3 * ctx->dim + IDX(i, j);
                diff[offset[2] + 31] = 4 * ctx->dim + IDX(i, j);
                offset[2] += 2 * P2_CC[2];
            }

            // log_a: 22 different points.
            offset[3] = 1 +
                        2 * (P2_CC[0] + P2_CC[1] + P2_CC[2]) * ctx->NrInterior * ctx->NzInterior +
                        2 * P2_CC[3] * ctx->NzInterior * (i - ctx->ghost);
            for (j = ctx->ghost; j < ctx->NzInterior + ctx->ghost; j++)
            {
                // Row indices are all row 3 * dim + IDX(i, j).
                diff[offset[3] + 0] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 2] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 4] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 6] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 8] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 10] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 12] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 14] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 16] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 18] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 20] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 22] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 24] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 26] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 28] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 30] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 32] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 34] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 36] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 38] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 40] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 42] = 3 * ctx->dim + IDX(i, j);
                // Column indices.
                diff[offset[3] + 1] = IDX(i - 1, j);
                diff[offset[3] + 3] = IDX(i, j - 1);
                diff[offset[3] + 5] = IDX(i, j);
                diff[offset[3] + 7] = IDX(i, j + 1);
                diff[offset[3] + 9] = IDX(i + 1, j);
                diff[offset[3] + 11] = ctx->dim + IDX(i - 1, j);
                diff[offset[3] + 13] = ctx->dim + IDX(i, j - 1);
                diff[offset[3] + 15] = ctx->dim + IDX(i, j);
                diff[offset[3] + 17] = ctx->dim + IDX(i, j + 1);
                diff[offset[3] + 19] = ctx->dim + IDX(i + 1, j);
                diff[offset[3] + 21] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[3] + 23] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[3] + 25] = 2 * ctx->dim + IDX(i, j);
                diff[offset[3] + 27] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[3] + 29] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[3] + 31] = 3 * ctx->dim + IDX(i, j);
                diff[offset[3] + 33] = 4 * ctx->dim + IDX(i - 1, j);
                diff[offset[3] + 35] = 4 * ctx->dim + IDX(i, j - 1);
                diff[offset[3] + 37] = 4 * ctx->dim + IDX(i, j);
                diff[offset[3] + 39] = 4 * ctx->dim + IDX(i, j + 1);
                diff[offset[3] + 41] = 4 * ctx->dim + IDX(i + 1, j);
                diff[offset[3] + 43] = GNUM * ctx->dim;
                offset[3] += 2 * P2_CC[3];
            }

            // psi: 19 different points, plus p_bound points.
            offset[4] = 1 +
                        2 * (P2_CC[0] + P2_CC[1] + P2_CC[2] + P2_CC[3]) * ctx->NrInterior *
                            ctx->NzInterior +
                        2 * (P2_CC[4] * ctx->NzInterior + 2) * (i - ctx->ghost);
            for (j = ctx->ghost; j < ctx->NzInterior + ctx->ghost; j++)
            {
                // Row indices are all row 4 * dim + IDX(i, j).
                diff[offset[4] + 0] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 2] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 4] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 6] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 8] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 10] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 12] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 14] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 16] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 18] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 20] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 22] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 24] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 26] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 28] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 30] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 32] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 34] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 36] = 4 * ctx->dim + IDX(i, j);
                // Column indices.
                diff[offset[4] + 1] = IDX(i - 1, j);
                diff[offset[4] + 3] = IDX(i, j - 1);
                diff[offset[4] + 5] = IDX(i, j);
                diff[offset[4] + 7] = IDX(i, j + 1);
                diff[offset[4] + 9] = IDX(i + 1, j);
                diff[offset[4] + 11] = ctx->dim + IDX(i, j);
                diff[offset[4] + 13] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[4] + 15] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[4] + 17] = 2 * ctx->dim + IDX(i, j);
                diff[offset[4] + 19] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[4] + 21] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[4] + 23] = 3 * ctx->dim + IDX(i, j);
                diff[offset[4] + 25] = 4 * ctx->dim + IDX(i - 1, j);
                diff[offset[4] + 27] = 4 * ctx->dim + IDX(i, j - 1);
                diff[offset[4] + 29] = 4 * ctx->dim + IDX(i, j);
                diff[offset[4] + 31] = 4 * ctx->dim + IDX(i, j + 1);
                diff[offset[4] + 33] = 4 * ctx->dim + IDX(i + 1, j);
                diff[offset[4] + 35] = 5 * ctx->dim + IDX(i, j);
                diff[offset[4] + 37] = GNUM * ctx->dim;
                offset[4] += 2 * P2_CC[4];
            }
            j = ctx->NzInterior + ctx->ghost;
            // Row.
            diff[offset[4] + 0] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 2] = 4 * ctx->dim + IDX(i, j);
            // Columns.
            diff[offset[4] + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 3] = GNUM * ctx->dim;
        }

        // Boundary points.
        offset[4] = 1 +
                    2 * (P2_CC[0] + P2_CC[1] + P2_CC[2] + P2_CC[3] + P2_CC[4]) * ctx->NrInterior *
                        ctx->NzInterior +
                    4 * ctx->NrInterior;
        i = ctx->NrInterior + ctx->ghost;
        for (j = ctx->ghost; j < ctx->NzTotal; j++)
        {
            // Row.
            diff[offset[4] + 0] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 2] = 4 * ctx->dim + IDX(i, j);
            // Columns.
            diff[offset[4] + 1] = 4 * ctx->dim + IDX(i, j);
            diff[offset[4] + 3] = GNUM * ctx->dim;
            offset[4] += 4;
        }

// Lambda interior points.
#pragma omp parallel for schedule(dynamic, 1) shared(diff) private(i, j, offset)
        for (i = ctx->ghost; i < ctx->NrInterior + ctx->ghost; i++)
        {
            // lambda: 22 different points.
            offset[5] = 1 +
                        2 * ((P2_CC[0] + P2_CC[1] + P2_CC[2] + P2_CC[3] + P2_CC[4]) *
                                 ctx->NrInterior * ctx->NzInterior +
                             2 * (ctx->NrInterior + ctx->NzInterior + 1)) +
                        2 * P2_CC[5] * ctx->NzInterior * (i - ctx->ghost);
            for (j = ctx->ghost; j < ctx->NzInterior + ctx->ghost; j++)
            {
                // Row indices are all row 5 * dim + IDX(i, j).
                diff[offset[5] + 0] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 2] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 4] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 6] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 8] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 10] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 12] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 14] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 16] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 18] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 20] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 22] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 24] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 26] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 28] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 30] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 32] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 34] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 36] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 38] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 40] = 5 * ctx->dim + IDX(i, j);
                // Column indices.
                diff[offset[5] + 1] = IDX(i - 1, j);
                diff[offset[5] + 3] = IDX(i, j - 1);
                diff[offset[5] + 5] = IDX(i, j);
                diff[offset[5] + 7] = IDX(i, j + 1);
                diff[offset[5] + 9] = IDX(i + 1, j);
                diff[offset[5] + 11] = ctx->dim + IDX(i - 1, j);
                diff[offset[5] + 13] = ctx->dim + IDX(i, j - 1);
                diff[offset[5] + 15] = ctx->dim + IDX(i - 1, j);
                diff[offset[5] + 17] = ctx->dim + IDX(i, j - 1);
                diff[offset[5] + 19] = 2 * ctx->dim + IDX(i - 1, j);
                diff[offset[5] + 21] = 2 * ctx->dim + IDX(i, j - 1);
                diff[offset[5] + 23] = 2 * ctx->dim + IDX(i, j);
                diff[offset[5] + 25] = 2 * ctx->dim + IDX(i, j + 1);
                diff[offset[5] + 27] = 2 * ctx->dim + IDX(i + 1, j);
                diff[offset[5] + 29] = 4 * ctx->dim + IDX(i - 1, j);
                diff[offset[5] + 31] = 4 * ctx->dim + IDX(i, j);
                diff[offset[5] + 33] = 4 * ctx->dim + IDX(i + 1, j);
                diff[offset[5] + 35] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 37] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 39] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 41] = 5 * ctx->dim + IDX(i, j);
                diff[offset[5] + 43] = 5 * ctx->dim + IDX(i, j);
                offset[5] += 2 * P2_CC[5];
            }
        }
    }

    // All done.
    return;
}
