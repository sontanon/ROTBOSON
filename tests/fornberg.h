// Reference finite-difference weights via Fornberg's algorithm.
//
// B. Fornberg, "Generation of finite difference formulas on arbitrarily spaced
// grids", Mathematics of Computation 51 (1988) 699-706.
//
// This is the *reference* implementation used by the unit tests to cross-check
// the derivative operators in src/derivatives.c. It is deliberately independent
// of the production code (no shared headers) so a bug in one cannot mask a bug
// in the other.
#ifndef ROTBOSON_FORNBERG_H
#define ROTBOSON_FORNBERG_H

// Compute finite-difference weights for derivatives 0..m evaluated at point z,
// on nodes x[0..n-1]. The result is written to c, laid out node-major:
//   c[node * (m + 1) + deriv]  = weight of node x[node] for the deriv-th
// derivative.
void fornberg_weights(const double *x, int n, double z, int m, double *c);

// Fill x with a uniform grid of 2*half+1 nodes, step h, centered at 0:
//   x[k] = (k - half) * h   for k = 0..2*half.
void uniform_nodes(double *x, int half, double h);

#endif /* ROTBOSON_FORNBERG_H */
