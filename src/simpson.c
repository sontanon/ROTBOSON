#include <stdio.h>
#include <omp.h>
#include <mkl.h>

#undef PARALLEL

double simps(double *y, const double dx, const MKL_INT dim)
{
	// Step size over 3.0
	double dx_o_3 = dx / 3.0;
	// Result initializes to zero.
	double i = 0.0;
	// Loop counter.
	MKL_INT k = 0;
	// Number of intervals.
	MKL_INT n = dim - 1;

	// Assert that we are integrating at least 5 intervals.
	if (dim < 6)
	{
		printf("CRITICAL WARNING ON SIMPSON INTEGRATION!\n");
		printf("Need at least dim = 6 but dim = %lld.\n", dim);
		return 0.0;
	}

	// Classical rule uses an even ammount of intervals.
	if (n % 2 == 0)
	{
		// Main loop.
#ifdef PARALLEL
		#pragma omp parallel for schedule(dynamic, 1) private(k, dx_o_3) reduction(+:i)
#endif
		for (k = 1; k < n / 2; ++k)
			i += dx_o_3 * 2.0 * (2.0 * y[2 * k - 1] + y[2 * k]);

		// Add values at beginning.
		i += dx_o_3 * y[0];
		// Add last two values.
		i += dx_o_3 * (4.0 * y[dim - 2] + y[dim - 1]);
	}
	// Otherwise apply Simpson's 3/8 rule on last four points (last three intervals).
	else
	{
		// Recursion on Simpson with proper even number of intervals.
		i = simps(y, dx, dim - 3);
		// Add 3/8 rule on last four points.
		i += 0.125 * dx * (y[dim - 4] + 3.0 * (y[dim - 3] + y[dim - 2]) + y[dim - 1]);
	}

	return i;
}