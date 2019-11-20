#include <math.h>

// Define hyperbolic secant.
#define sech(X) (1.0 / cosh((X)))
// Absolute value macro.
#define ABS(X) ((X) < 0) ? -(X) : (X)

double omega_calc(const double u, const double m)
{
	//return u;
	return 0.5 * m * (1.0 + tanh(u));
}

double inverse_omega_calc(const double w, const double m)
{
	//return w;
	return atanh(2.0 * w / m - 1.0);
}

double dw_du(const double u, const double m)
{
	// return 1.0;
	return 0.5 * m * sech(u) * sech(u);
}
