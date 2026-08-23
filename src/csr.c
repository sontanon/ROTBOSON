#include "tools.h"
#include "context.h"

#include "csr_grid_fill.h"
#include "csr_vars.h"
#include "csr_omega_constraint.h"

#define EVEN 1
#define ODD -1

#define DIRICHLET_TYPE -1
#define EXP_DECAY_TYPE 0
#define ROBIN_TYPE_1 1
#define ROBIN_TYPE_3 3
#define REG_VARIABLE 5

// Print nnz cumulative sum.
#undef DEBUG

void nnz_jacobian_get_nnzs(const rb_context *ctx, MKL_INT *p_nnz1, MKL_INT *p_nnz2, MKL_INT *p_nnz3,
                           MKL_INT *p_nnz4, MKL_INT *p_nnz5, MKL_INT *p_nnz6)
{
    MKL_INT nnz1 = 0, nnz2 = 0, nnz3 = 0, nnz4 = 0, nnz5 = 0, nnz6 = 0;

    // Select order.
    if (ctx->order == 4)
    {
        // Interior points plus parity boundaries.
        nnz1 = 30 * ctx->NrInterior * ctx->NzInterior + 31 * (ctx->NrInterior + ctx->NzInterior) +
               32 + 4 * (ctx->NrInterior + ctx->NzInterior) + 24;
        nnz2 = 29 * ctx->NrInterior * ctx->NzInterior + 31 * (ctx->NrInterior + ctx->NzInterior) +
               32 + 4 * (ctx->NrInterior + ctx->NzInterior) + 24;
        nnz3 = 28 * ctx->NrInterior * ctx->NzInterior + 30 * (ctx->NrInterior + ctx->NzInterior) +
               31 + 4 * (ctx->NrInterior + ctx->NzInterior) + 24;
        nnz4 = 46 * ctx->NrInterior * ctx->NzInterior + 47 * (ctx->NrInterior + ctx->NzInterior) +
               48 + 4 * (ctx->NrInterior + ctx->NzInterior) + 24;
        nnz5 = 31 * ctx->NrInterior * ctx->NzInterior + 32 * (ctx->NrInterior + ctx->NzInterior) +
               33 + 4 * (ctx->NrInterior + ctx->NzInterior) + 24;
        nnz6 = 40 * ctx->NrInterior * ctx->NzInterior + 42 * ctx->NrInterior +
               44 * ctx->NzInterior + 45 + 4 * (ctx->NrInterior + ctx->NzInterior) + 24;

        // Select boundary conditions.
        nnz1 += 9 * (3 + (ctx->NrInterior + ctx->NzInterior));
        nnz2 += 9 * (3 + (ctx->NrInterior + ctx->NzInterior));
        nnz3 += 9 * (3 + (ctx->NrInterior + ctx->NzInterior));
        nnz4 += 9 * (3 + (ctx->NrInterior + ctx->NzInterior));
        nnz5 += 10 * (3 + (ctx->NrInterior + ctx->NzInterior));
        nnz6 += 9 * (3 + (ctx->NrInterior + ctx->NzInterior));
    }
    else
    {
        // Interior points plus parity boundaries.
        nnz1 = 18 * ctx->NrInterior * ctx->NzInterior + 2 * (ctx->NrInterior + ctx->NzInterior) + 6;
        nnz2 = 17 * ctx->NrInterior * ctx->NzInterior + 2 * (ctx->NrInterior + ctx->NzInterior) + 6;
        nnz3 = 16 * ctx->NrInterior * ctx->NzInterior + 2 * (ctx->NrInterior + ctx->NzInterior) + 6;
        nnz4 = 26 * ctx->NrInterior * ctx->NzInterior + 2 * (ctx->NrInterior + ctx->NzInterior) + 6;
        nnz5 = 19 * ctx->NrInterior * ctx->NzInterior + 2 * (ctx->NrInterior + ctx->NzInterior) + 6;
        nnz6 = 22 * ctx->NrInterior * ctx->NzInterior + 2 * (ctx->NrInterior + ctx->NzInterior) + 6;

        // Select boundary conditions.
        nnz1 += 5 * (1 + (ctx->NrInterior + ctx->NzInterior));
        nnz2 += 5 * (1 + (ctx->NrInterior + ctx->NzInterior));
        nnz3 += 5 * (1 + (ctx->NrInterior + ctx->NzInterior));
        nnz4 += 5 * (1 + (ctx->NrInterior + ctx->NzInterior));
        nnz5 += 6 * (1 + (ctx->NrInterior + ctx->NzInterior));
        nnz6 += 5 * (1 + (ctx->NrInterior + ctx->NzInterior));
    }

    *p_nnz1 = nnz1;
    *p_nnz2 = nnz2;
    *p_nnz3 = nnz3;
    *p_nnz4 = nnz4;
    *p_nnz5 = nnz5;
    *p_nnz6 = nnz6;

    return;
}

MKL_INT nnz_jacobian(const rb_context *ctx)
{
    // Number of nonzero elements per grid function.
    MKL_INT nnz1 = 0, nnz2 = 0, nnz3 = 0, nnz4 = 0, nnz5 = 0, nnz6 = 0;

    nnz_jacobian_get_nnzs(ctx, &nnz1, &nnz2, &nnz3, &nnz4, &nnz5, &nnz6);
#ifdef DEBUG
    printf("nnz1 = %lld,\n", nnz1);
    printf("nnz2 = %lld,\n", nnz2);
    printf("nnz3 = %lld,\n", nnz3);
    printf("nnz4 = %lld,\n", nnz4);
    printf("nnz5 = %lld,\n", nnz5);
    printf("\n");
    printf("nnz1 + nnz2 = %lld,\n", nnz1 + nnz2);
    printf("nnz1 + nnz2 + nnz3 = %lld,\n", nnz1 + nnz2 + nnz3);
    printf("nnz1 + nnz2 + nnz3 + nnz4 = %lld.\n", nnz1 + nnz2 + nnz3 + nnz4);
#endif

    // Total number of nonzeros.
    return nnz1 + nnz2 + nnz3 + nnz4 + nnz5 + nnz6 + 1;
}

void csr_gen_jacobian(rb_context *ctx, csr_matrix A, double *u, const MKL_INT print)
{
    // Number of elements we have filled in.
    MKL_INT offset = 0;

    // Number of nonzero elements per grid function.
    MKL_INT nnz1 = 0, nnz2 = 0, nnz3 = 0, nnz4 = 0, nnz5 = 0, nnz6 = 0;

    // Calculate nonzeros.
    nnz_jacobian_get_nnzs(ctx, &nnz1, &nnz2, &nnz3, &nnz4, &nnz5, &nnz6);

    // Integer arrays.
    MKL_INT r_sym[GNUM] = {EVEN, EVEN, EVEN, EVEN, EVEN, EVEN};
    MKL_INT z_sym[GNUM] = {EVEN, EVEN, EVEN, EVEN, EVEN, EVEN};
    MKL_INT bound_order[GNUM] = {ROBIN_TYPE_1, ROBIN_TYPE_1,   ROBIN_TYPE_1,
                                 ROBIN_TYPE_1, EXP_DECAY_TYPE, ROBIN_TYPE_1};
    MKL_INT nnzs[GNUM] = {nnz1, nnz2, nnz3, nnz4, nnz5, nnz6};

    MKL_INT p_cc[GNUM] = {0, 0, 0, 0, 0, 0};
    MKL_INT p_cs[GNUM] = {0, 0, 0, 0, 0, 0};
    MKL_INT p_sc[GNUM] = {0, 0, 0, 0, 0, 0};
    MKL_INT p_ss[GNUM] = {0, 0, 0, 0, 0, 0};
    MKL_INT p_bound[GNUM] = {0, 0, 0, 0, 0, 0};

    // Set integer arrays according to order.
    if (ctx->order == 4)
    {
        p_cc[0] = 30;
        p_cc[1] = 29;
        p_cc[2] = 28;
        p_cc[3] = 46;
        p_cc[4] = 31;
        p_cc[5] = 40;

        p_cs[0] = p_sc[0] = 31;
        p_cs[1] = p_sc[1] = 31;
        p_cs[2] = p_sc[2] = 30;
        p_cs[3] = p_sc[3] = 47;
        p_cs[4] = p_sc[4] = 32;
        p_cs[5] = 42;
        p_sc[5] = 44;

        p_ss[0] = 32;
        p_ss[1] = 32;
        p_ss[2] = 31;
        p_ss[3] = 48;
        p_ss[4] = 33;
        p_ss[5] = 45;

        p_bound[0] = p_bound[1] = p_bound[2] = p_bound[3] = p_bound[5] = 9;
        p_bound[4] = 10;
    }
    else
    {
        p_cc[0] = 18;
        p_cc[1] = 17;
        p_cc[2] = 16;
        p_cc[3] = 26;
        p_cc[4] = 19;
        p_cc[5] = 22;

        p_bound[0] = p_bound[1] = p_bound[2] = p_bound[3] = p_bound[5] = 5;
        p_bound[4] = 6;
    }

    // Check for order and fill matrix.
    if (ctx->order == 4)
    {
        csr_grid_fill_4th(
            A, ctx->NrInterior, ctx->NzInterior, ctx->dr, ctx->dz, u, ctx->l, ctx->m, r_sym, z_sym,
            bound_order, nnzs, p_cc, p_cs, p_sc, p_ss, p_bound,
            jacobian_4th_order_variable_omega_cc, jacobian_4th_order_variable_omega_cs,
            jacobian_4th_order_variable_omega_sc, jacobian_4th_order_variable_omega_ss);
    }
    // Second order default.
    else
    {
        csr_grid_fill_2nd(A, ctx->NrInterior, ctx->NzInterior, ctx->dr, ctx->dz, u, ctx->l, ctx->m,
                          r_sym, z_sym, bound_order, nnzs, p_cc, p_bound,
                          jacobian_2nd_order_variable_omega_cc);
    }

    // FINALLY FILL OMEGA EQUATION OR u5(1,1) CONSTRAINT.
    // printf("ROTBOSON-JACOBIAN: Starting omega...\n");
    offset = nnz1 + nnz2 + nnz3 + nnz4 + nnz5 + nnz6;
    omega_constraint(A.a, A.ia, A.ja, offset, ctx->NrTotal, ctx->NzTotal, ctx->dim, 5, ctx->w_idx,
                     ctx->fixedPhi, ctx->fixedPhiR, ctx->fixedPhiZ);
    // printf("ROTBOSON-JACOBIAN: Done omega.\n");

    // FILL LAST ELEMENT WITH NUMBER OF NONZEROS.
    A.ia[ctx->w_idx + 1] = BASE + A.nnz;

    // PRINT MATRIX.
    if (print)
        csr_print(&A, "a.asc", "ia.asc", "ja.asc");

    // All done.
    return;
}
