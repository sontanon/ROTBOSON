#include "tools.h"
#include "param.h"

#include "omega_calc.h"

#undef BDRY_DEBUG

void initial_guess(double *u)
{
	// The main idea is to set:
	// 1. Lapse alpha to one.
	// 2. Shift beta to zero.
	// 3. h function to one.
	// 4. a function to one.
	// 5. Scalar field to Gaussian profile (see below).
	// 6. Omega to one.
	//
	// In terms of the five functions u1, u2, u3, u4, u5.
	//  u1 = 0.0
	//  u2 = 0.0
	//  u3 = 0.0
	//  u4 = 0.0
	//  u5 = Gaussian profile..

	// Integer counter.
	MKL_INT i = 0;
	MKL_INT j = 0;

	// Auxiliary variables.
	double r, z, rr;
	double m2 = m * m;
	double w2 = w0 * w0;
	double chi = sqrt(m2 - w2);

	// log(alpha)	= 0.0
	// beta 	= 0.0
	// log(h)	= 0.0
	// log(a)	= 0.0

	if (!log_alpha_i)
	{
		#pragma omp parallel shared(u)
		{
			#pragma omp for schedule(guided)
			for (i = 0 * dim; i <  1 * dim; ++i)
			{
			    u[i] = 0.0;
			}
		}

#ifdef BDRY_DEBUG
		#pragma omp parallel shared(u) private(i, j, r, z, rr) // rr.
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost; i < NrTotal; ++i)
			{
			    r = dr * (i + 0.5 - ghost);

			    for (j = ghost; j < NzTotal; ++j)
			    {
				z = dz * (j + 0.5 - ghost);
				rr = sqrt(r * r + z * z);

				u[0 * dim + IDX(i, j)] = log(1.0 + 1.0 / rr);
			    }
			}
		}
#endif
	}
	else
	{
		read_single_file_2d(u + 0 * dim, log_alpha_i, NrTotal, NzTotal, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
		printf("***           Read log_alpha initial data.        \n");
	}

	if (!beta_i)
	{
		#pragma omp parallel shared(u)
		{
			#pragma omp for schedule(guided)
			for (i = 1 * dim; i <  2 * dim; ++i)
			{
			    u[i] = 0.0;
			}
		}

#ifdef BDRY_DEBUG
		#pragma omp parallel shared(u) private(i, j, r, z, rr) // rr.
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost; i < NrTotal; ++i)
			{
			    r = dr * (i + 0.5 - ghost);

			    for (j = ghost; j < NzTotal; ++j)
			    {
				z = dz * (j + 0.5 - ghost);
				rr = sqrt(r * r + z * z);

				u[1 * dim + IDX(i, j)] = 1.0 / pow(rr, 3);
			    }
			}
		}
#endif
	}
	else
	{
		read_single_file_2d(u + 1 * dim, beta_i, NrTotal, NzTotal, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
		printf("***           Read beta initial data.        \n");
	}

	if (!log_h_i)
	{
		#pragma omp parallel shared(u)
		{
			#pragma omp for schedule(guided)
			for (i = 2 * dim; i <  3 * dim; ++i)
			{
			    u[i] = 0.0;
			}
		}
#ifdef BDRY_DEBUG
		#pragma omp parallel shared(u) private(i, j, r, z, rr) // rr.
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost; i < NrTotal; ++i)
			{
			    r = dr * (i + 0.5 - ghost);

			    for (j = ghost; j < NzTotal; ++j)
			    {
				z = dz * (j + 0.5 - ghost);
				rr = sqrt(r * r + z * z);

				u[2 * dim + IDX(i, j)] = log(1.0 + 1.0 / rr);
			    }
			}
		}
#endif
	}
	else
	{
		read_single_file_2d(u + 2 * dim, log_h_i, NrTotal, NzTotal, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
		printf("***           Read log_h initial data.        \n");
	}

	if (!log_a_i)
	{
		#pragma omp parallel shared(u)
		{
			#pragma omp for schedule(guided)
			for (i = 3 * dim; i <  4 * dim; ++i)
			{
			    u[i] = 0.0;
			}
		}
#ifdef BDRY_DEBUG
		#pragma omp parallel shared(u) private(i, j, r, z, rr) // rr.
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost; i < NrTotal; ++i)
			{
			    r = dr * (i + 0.5 - ghost);

			    for (j = ghost; j < NzTotal; ++j)
			    {
				z = dz * (j + 0.5 - ghost);
				rr = sqrt(r * r + z * z);

				u[3 * dim + IDX(i, j)] = log(1.0 + 1.0 / rr);
			    }
			}
		}
#endif
	}
	else
	{
		read_single_file_2d(u + 3 * dim, log_a_i, NrTotal, NzTotal, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
		printf("***           Read log_a initial data.        \n");
	}

	if (!psi_i)
	{
		// Now do initial guess for phi.
		#pragma omp parallel shared(u) private(i, j, r, z, rr) // rr.
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost; i < NrTotal; ++i)
			{
			    r = dr * (i + 0.5 - ghost);
			    //rl = pow(r, l);

			    for (j = ghost; j < NzTotal; ++j)
			    {
				z = dz * (j + 0.5 - ghost);
				rr = sqrt(r * r + z * z);

				u[4 * dim + IDX(i, j)] = psi0 * exp(-0.5 * r * r / (sigmaR * sigmaR)) * exp(-0.5 * z * z / (sigmaZ * sigmaZ))
					+ (psi0 * exp(-chi * rr) / pow(rr, l + 1)) * (0.5 + 0.5 * erf(2.0 * (rr - rExt) / M_2_SQRTPI));	

				//u[4 * dim + IDX(i, j)] = log(psi0) - 0.5 * (r * r / (sigmaR * sigmaR) + z * z / (sigmaZ * sigmaZ));
				/* Previous deprecated inital data: phi = r**l * psi.
				u[4 * dim + IDX(i, j)] = psi0 * exp(-0.5 * r * r / (sigmaR * sigmaR)) * exp(-0.5 * z * z / (sigmaZ * sigmaZ))
					+ (psi0 * exp(-chi * rr) / pow(rr, l + 1)) * (0.5 + 0.5 * erf(2.0 * (rr - rExt) / M_2_SQRTPI));	
				*/
			    }
			}
		}
#ifdef BDRY_DEBUG
		#pragma omp parallel shared(u) private(i, j, r, z, rr) // rr.
		{
			#pragma omp for schedule(dynamic, 1)
			for (i = ghost; i < NrTotal; ++i)
			{
			    r = dr * (i + 0.5 - ghost);

			    for (j = ghost; j < NzTotal; ++j)
			    {
				z = dz * (j + 0.5 - ghost);
				rr = sqrt(r * r + z * z);

				u[4 * dim + IDX(i, j)] = -(l + 1.0) * log(rr);
			    }
			}
		}
#endif
	}
	else
	{
		// Rescale scalar field by constant psi0.
		cblas_dscal(dim, psi0, u + 4 * dim, 1);

		read_single_file_2d(u + 4 * dim, psi_i, NrTotal, NzTotal, NrTotalInitial, NzTotalInitial, __FILE__, __LINE__);
		printf("***           Read psi initial data.        \n");
	}

	// Assert symmetries since they might not be automatic.
	// All functions are even with respect to the axis and equator.
	// Corner.
	for (i = 0; i < ghost; ++i)
	{
		for (j = 0; j < ghost; ++j)
		{
			u[0 * dim + IDX(i, j)] = u[0 * dim + IDX(2 * ghost - (i + 1), 2 * ghost - (j + 1))];
			u[1 * dim + IDX(i, j)] = u[1 * dim + IDX(2 * ghost - (i + 1), 2 * ghost - (j + 1))];
			u[2 * dim + IDX(i, j)] = u[2 * dim + IDX(2 * ghost - (i + 1), 2 * ghost - (j + 1))];
			u[3 * dim + IDX(i, j)] = u[3 * dim + IDX(2 * ghost - (i + 1), 2 * ghost - (j + 1))];
			u[4 * dim + IDX(i, j)] = u[4 * dim + IDX(2 * ghost - (i + 1), 2 * ghost - (j + 1))];
		}
	}
	// Axis.
	#pragma omp parallel shared(u) private(i, j)
	{
	    	#pragma omp for schedule(dynamic, 1)
	    	for (j = ghost; j < NzTotal; ++j)
	    	{
			for (i = 0; i < ghost; ++i)
			{
				u[0 * dim + IDX(i, j)] = u[0 * dim + IDX(2 * ghost - (i + 1), j)];
				u[1 * dim + IDX(i, j)] = u[1 * dim + IDX(2 * ghost - (i + 1), j)];
				u[2 * dim + IDX(i, j)] = u[2 * dim + IDX(2 * ghost - (i + 1), j)];
				u[3 * dim + IDX(i, j)] = u[3 * dim + IDX(2 * ghost - (i + 1), j)];
				u[4 * dim + IDX(i, j)] = u[4 * dim + IDX(2 * ghost - (i + 1), j)];
			}
	    	}
	}
	// Equator.
	#pragma omp parallel shared(u) private(i, j)
	{
		#pragma omp for schedule(dynamic, 1)
		for (i = ghost; i < NrTotal; ++i)
		{
			for (j = 0; j < ghost; ++j)
			{
				u[0 * dim + IDX(i, j)] = u[0 * dim + IDX(i, 2 * ghost - (j + 1))];
				u[1 * dim + IDX(i, j)] = u[1 * dim + IDX(i, 2 * ghost - (j + 1))];
				u[2 * dim + IDX(i, j)] = u[2 * dim + IDX(i, 2 * ghost - (j + 1))];
				u[3 * dim + IDX(i, j)] = u[3 * dim + IDX(i, 2 * ghost - (j + 1))];
				u[4 * dim + IDX(i, j)] = u[4 * dim + IDX(i, 2 * ghost - (j + 1))];
			}
		}
	}

	// Finally set omega variable.
	if (w_i)
	{
		read_single_file_1d(&w0, w_i, 1, __FILE__, __LINE__);
		printf("***          Read omega initial data.       \n");
	}
	u[w_idx] = inverse_omega_calc(w0, m);

	// All done.
	return;
}
