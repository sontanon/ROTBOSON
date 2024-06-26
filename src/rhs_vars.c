#include "tools.h"
#include "omega_calc.h"

// Regularized terms.
#define Q1 1.0
#define Q2 1.0

#undef ANALYTIC

void rhs_vars(
	double *f, 
	double *u, 
	double *Dr_u,
	double *Dz_u,
	double *Drr_u, 
	double *Dzz_u,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim,
	const MKL_INT ghost,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const MKL_INT l,
	const double m,
	const double w,
	const double rescale,
	double *u_aux,
	double *Dr_u_aux
)
{
	// Omega.
	//double w2 = w * w;
	double m2 = m * m;
	//double chi = sqrt(m2 - w2);

	// Auxiliary doubles.
	double r;
	//double z;
	double r2, rlm1, rl;
	//double z2, rr;
	//double scale;
	double l_alpha, Dr_l_alpha, Dz_l_alpha, Drr_l_alpha, Dzz_l_alpha;
	double l_h, Dr_l_h, Dz_l_h, Drr_l_h, Dzz_l_h;
	//double l_a, Dr_l_a, Dz_l_a, Drr_l_a, Dzz_l_a;
	double l_a, Drr_l_a, Dzz_l_a;
	double psi, Dr_psi, Dz_psi, Drr_psi, Dzz_psi;
	double beta,  Dr_beta,  Dz_beta,  Drr_beta,  Dzz_beta;

	// Step ratio.
	double dzodr = dz / dr;

	// Short-hand variables.
	double alpha, alpha2, h2, a2, w_plus_l_beta, w_plus_l_beta2;
	double phi, phi_over_r, phi2, phi2_over_r2;
	double lambda, Dr_lambda, Dz_lambda, Drr_lambda, Dzz_lambda;
	double Dr_u6, Dr_u7;

	// Derivative terms.
	double D_l_alpha_D_l_alpha, D_l_alpha_D_beta, D_l_alpha_D_l_h, D_beta_D_l_h;
	double D_l_h_D_l_h, D_psi_D_psi, D_l_alpha_D_psi, D_l_h_D_psi;
	double r2_h2_over_alpha2_D_beta_D_beta;

	// Fetch values.
	l_alpha     = u[IDX(i, j)];
	Dr_l_alpha  = Dr_u[IDX(i, j)];
	Dz_l_alpha  = Dz_u[IDX(i, j)];
	Drr_l_alpha = Drr_u[IDX(i, j)];
	Dzz_l_alpha = Dzz_u[IDX(i, j)];

	beta      = u[dim + IDX(i, j)];
	Dr_beta   = Dr_u[dim + IDX(i, j)];
	Dz_beta   = Dz_u[dim + IDX(i, j)];
	Drr_beta  = Drr_u[dim + IDX(i, j)];
	Dzz_beta  = Dzz_u[dim + IDX(i, j)];

	l_h     = u[2 * dim + IDX(i, j)];
	Dr_l_h  = Dr_u[2 * dim + IDX(i, j)];
	Dz_l_h  = Dz_u[2 * dim + IDX(i, j)];
	Drr_l_h = Drr_u[2 * dim + IDX(i, j)];
	Dzz_l_h = Dzz_u[2 * dim + IDX(i, j)];

	l_a     = u[3 * dim + IDX(i, j)];
	//Dr_l_a  = Dr_u[3 * dim + IDX(i, j)];
	//Dz_l_a  = Dz_u[3 * dim + IDX(i, j)];
	Drr_l_a = Drr_u[3 * dim + IDX(i, j)];
	Dzz_l_a = Dzz_u[3 * dim + IDX(i, j)];

	psi     = u[4 * dim + IDX(i, j)];
	Dr_psi  = Dr_u[4 * dim + IDX(i, j)];
	Dz_psi  = Dz_u[4 * dim + IDX(i, j)];
	Drr_psi = Drr_u[4 * dim + IDX(i, j)];
	Dzz_psi = Dzz_u[4 * dim + IDX(i, j)];

	lambda = u[5 * dim + IDX(i, j)];
	Dr_lambda  = Dr_u[5 * dim + IDX(i, j)];
	Dz_lambda  = Dz_u[5 * dim + IDX(i, j)];
	Drr_lambda = Drr_u[5 * dim + IDX(i, j)];
	Dzz_lambda = Dzz_u[5 * dim + IDX(i, j)];

	// Coordinates.
	r = dr * (i + 0.5 - ghost);
	r2 = r * r;
	//z = dz * (j + 0.5 - ghost);
	//z2 = z * z;
	//rr = sqrt(r2 + z2);
	rlm1 = (l ==1) ? 1.0 : pow(r, l - 1);
	rl = rlm1 * r;

	// Auxiliary variables.
	alpha = exp(l_alpha);
	alpha2 = alpha * alpha;
	h2 = exp(2.0 * l_h);
	a2 = exp(2.0 * l_a);
	w_plus_l_beta = w + (double)l * beta;
	w_plus_l_beta2 = w_plus_l_beta * w_plus_l_beta;
	//phi = rl * exp(-chi * rr) * exp(psi);
	phi = rl * psi;
	//phi_over_r = rlm1 * exp(-chi * rr) * exp(psi);
	phi_over_r = rlm1 * psi;
	phi2 = phi * phi;
	phi2_over_r2 = phi_over_r * phi_over_r;

	// Derivative terms.
	D_l_alpha_D_l_alpha = Dr_l_alpha * Dr_l_alpha + Dz_l_alpha * Dz_l_alpha;
	D_l_alpha_D_beta = Dr_l_alpha * Dr_beta + Dz_l_alpha * Dz_beta;
	D_l_alpha_D_l_h = Dr_l_alpha * Dr_l_h + Dz_l_alpha * Dz_l_h;
	D_beta_D_l_h = Dr_beta * Dr_l_h + Dz_beta * Dz_l_h;
	D_l_h_D_l_h = Dr_l_h * Dr_l_h + Dz_l_h * Dz_l_h;
	r2_h2_over_alpha2_D_beta_D_beta = (r2 * h2 / alpha2) * (Dr_beta * Dr_beta + Dz_beta * Dz_beta);
	D_psi_D_psi = Dr_psi * Dr_psi + Dz_psi * Dz_psi;
	D_l_alpha_D_psi = Dr_l_alpha * Dr_psi + Dz_l_alpha * Dz_psi;
	D_l_h_D_psi = Dr_psi * Dr_l_h + Dz_psi * Dz_l_h;
	
	// Radial derivatives.
	//double Dx_l_alpha = (r / rr) * Dr_l_alpha + (z / rr) * Dz_l_alpha;
	//double Dx_l_h 	= (r / rr) * Dr_l_h + (z / rr) * Dz_l_h;
	//double Dx_psi	= (r / rr) * Dr_psi + (z / rr) * Dz_psi;

	// Regularization auxiliaries.
	//u6 = u_aux[0 * dim + IDX(i, j)];
	//u7 = u_aux[1 * dim + IDX(i, j)];
	Dr_u6 = Dr_u_aux[0 * dim + IDX(i, j)];
	Dr_u7 = Dr_u_aux[1 * dim + IDX(i, j)];

	// u0 = log(alpha).
	f[IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_l_alpha + Dzz_l_alpha + (Dr_l_alpha / r)
		+ D_l_alpha_D_l_alpha + D_l_alpha_D_l_h 
		- 0.5 * r2_h2_over_alpha2_D_beta_D_beta 
		+ 4.0 * M_PI * a2 * (m2 - 2.0 * w_plus_l_beta2 / alpha2) * phi2));

	// u1 = beta.
	f[dim + IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_beta + Dzz_beta + 3.0 * (Dr_beta / r)
		- D_l_alpha_D_beta + 3.0 * D_beta_D_l_h
		- 16.0 * M_PI * a2 * l * w_plus_l_beta * phi2_over_r2 / h2));

	// u2 = log(h).
	f[2 * dim + IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_l_h + Dzz_l_h + 2.0 * (Dr_l_h / r)
		+ D_l_h_D_l_h + D_l_alpha_D_l_h + 0.5 * r2_h2_over_alpha2_D_beta_D_beta
		+ (Dr_l_alpha / r) 
		+ 4.0 * M_PI * a2 * (r2 * m2 + 2.0 * l * l / h2) * phi2_over_r2));

	// u3 = log(a).
	f[3 * dim + IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_l_a + Dzz_l_a
		- D_l_alpha_D_l_h - 0.25 * r2_h2_over_alpha2_D_beta_D_beta 
		- (Dr_l_alpha / r)
		+ 4.0 * M_PI * ((l * l * (1.0 - a2 / h2) + a2 * r2 * w_plus_l_beta2 / alpha2) * phi2_over_r2
			+ (2.0 * l * r * psi * Dr_psi + r2 * D_psi_D_psi) * (rlm1 * rlm1))));

	// u4 = log(phi / r**l)
	f[4 * dim + IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_psi + Dzz_psi + (2.0 * l + 1.0) * (Dr_psi / r)
		+ D_l_alpha_D_psi + D_l_h_D_psi
		+ l * ((Dr_l_alpha / r) + (Dr_l_h / r)) * psi
		+ a2 * (w_plus_l_beta2 / alpha2 - m2) * psi
		- l * l * lambda * psi / h2));

	// u5 = lambda = (A - H) / r**2.
	// Change a2 calculation.
	a2 = h2 + r2 * lambda;
	f[5 * dim + IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_lambda + Dzz_lambda + 3.0 * (Dr_lambda / r)
		+ (Dz_l_h * Dz_lambda - Dr_l_h * Dr_lambda)
		- 4.0 * (h2 / a2) * (Dr_lambda * Dr_l_h + Dz_lambda * Dz_l_h)
		- (r2 / a2) * (Dr_lambda * Dr_lambda + Dz_lambda * Dz_lambda)
		+ (Dz_l_alpha * Dz_lambda - Dr_l_alpha * Dr_lambda)
		- 4.0 * (lambda * lambda / a2) 
		- 4.0 * (lambda / a2) * (r * Dr_lambda)
		- 2.0 * (Dr_l_alpha / r) * lambda
		+ 2.0 * (Dr_l_h / r) * (-4.0 * (h2 / a2) + 1.0) * lambda
		- 4.0 * h2 * (Dr_l_h / r) * (Dr_l_h / r) * ((h2 / a2) + 0.5)
		+ 4.0 * (h2 / a2) * Dz_l_h * Dz_l_h * lambda
		- 2.0 * Dr_l_h * Dr_l_h * lambda
		- 4.0 * h2 * (Dr_l_h / r) * (Dr_l_alpha / r)
		- (h2 * h2 / alpha2) * (Dr_beta * Dr_beta + Dz_beta * Dz_beta)
		- (a2 * h2 / alpha2) * (Dr_beta * Dr_beta)
		+ 2.0 * lambda * (Drr_l_alpha + Dr_l_alpha * Dr_l_alpha)
		+ 2.0 * lambda * (Drr_l_h + 2.0 * Dr_l_h * Dr_l_h)
		+ Q1 * ((2.0 * h2 / alpha) * (Dr_u6 / r))
		+ Q2 * (Dr_u7 / r)
		+ 8.0 * M_PI * a2 * (m2 * lambda * phi2
			+ 2.0 * rlm1 * rlm1 * (Dr_psi / r) * (2.0 * l * psi + (r * Dr_psi)))));

	// All done.
	return;
}

void rhs_bdry(
	double *f, 
	double *u, 
	double *Dr_u,
	double *Dz_u,
	const MKL_INT NrTotal,
	const MKL_INT NzTotal,
	const MKL_INT dim,
	const MKL_INT ghost,
	const MKL_INT i,
	const MKL_INT j,
	const double dr,
	const double dz,
	const MKL_INT l,
	const double m,
	const double w,
	const double M,
	const double J,
	const double rescale
)
{
	// Auxiliary doubles.
	double r = dr * (i + 0.5 - ghost);
	double z = dz * (j + 0.5 - ghost);
	//double dzodr = dz / dr;
	double rr2 = r * r + z * z;
	double rr = sqrt(rr2);
	double scale = dr * dz / rr2;

	// Omega.
	double w2 = w * w;
	double m2 = m * m;
	double chi = sqrt(m2 - w2);

	// Fetch values.
	double l_alpha     = u[IDX(i, j)];
	double Dr_l_alpha  = Dr_u[IDX(i, j)];
	double Dz_l_alpha  = Dz_u[IDX(i, j)];
	double beta      = u[dim + IDX(i, j)];
	double Dr_beta   = Dr_u[dim + IDX(i, j)];
	double Dz_beta   = Dz_u[dim + IDX(i, j)];
	double l_h     = u[2 * dim + IDX(i, j)];
	double Dr_l_h  = Dr_u[2 * dim + IDX(i, j)];
	double Dz_l_h  = Dz_u[2 * dim + IDX(i, j)];
	double l_a     = u[3 * dim + IDX(i, j)];
	double Dr_l_a  = Dr_u[3 * dim + IDX(i, j)];
	double Dz_l_a  = Dz_u[3 * dim + IDX(i, j)];
	double psi     = u[4 * dim + IDX(i, j)];
	double Dr_psi  = Dr_u[4 * dim + IDX(i, j)];
	double Dz_psi  = Dz_u[4 * dim + IDX(i, j)];
	double lambda    = u[5 * dim + IDX(i, j)];
	double Dr_lambda = Dr_u[5 * dim + IDX(i, j)];
	double Dz_lambda = Dz_u[5 * dim + IDX(i, j)];

	// Kerr matching.
	/*
	double a = J / M;
	double M2 = M * M;
	double a2 = a * a;
	double a4 = a2 * a2;
	double rr4 = rr2 * rr2;
	double sin_th_2 = r * r / rr2;
	double sin_th_3 = sin_th_2 * (r / rr);
	double sin_th_4 = sin_th_2 * sin_th_2;
	double sin_th_6 = sin_th_4 * sin_th_2;
	double cos_th_2 = z * z / rr2;
	double cos_th_4 = cos_th_2 * cos_th_2;
	double cos_th_6 = cos_th_4 * cos_th_2;
	double cos_2th = cos_th_2 - sin_th_2;
	double cos_4th = 1.0 - 8.0 * cos_th_2 * sin_th_2;
	double cos_6th = cos_th_6 - 15.0 * cos_th_4 * sin_th_2 + 15.0 * cos_th_2 * sin_th_4 - sin_th_6;
	*/

	/* Analytic expressions. */
#ifdef ANALYTIC
	double kerr_alpha2 = (
		(  (a2 + 2.0 * rr2 - 4.0 * M * rr + a2 * cos_2th) 
		 / (a2 + 2.0 * rr2                + a2 * cos_2th))
		* 
		(  (3.0 * a4 + 8.0 * rr4 + a2 * (+ 6.0 * M2 + 8.0 * rr2) + a2 * ((4.0 * a2 -        M2 + 8.0 * rr2) * cos_2th + (a2 - 6.0 * M2) * cos_4th + M2 * cos_6th))
		 / (3.0 * a4 + 8.0 * rr4 + a2 * (-10.0 * M2 + 8.0 * rr2) + a2 * ((4.0 * a2 + 15.0 * M2 + 8.0 * rr2) * cos_2th + (a2 - 6.0 * M2) * cos_4th + M2 * cos_6th))));

	double kerr_Omega = - ((2.0 * J / rr) * (a2 * cos_th_2 + rr2 - 2.0 * M * rr) / (-4.0 * a2 * M2 * sin_th_6 + (rr2 + a2 * cos_th_2) * (rr2 + a2 * cos_th_2)));

	double kerr_A = (rr2 + a2 * cos_th_2) / ((a2 - M2) * cos_th_2 + (M2 + rr2 - 2.0 * M * rr));

	double kerr_H = ((rr2 + a2 * cos_th_2) * (rr2 + a2 * cos_th_2) - 4.0 * a2 * M2 * sin_th_6) / ((rr2 + a2 * cos_th_2) * (rr2 - 2.0 * M * rr + a2 * cos_th_2));

	double kerr_lambda = ((-M2 / (rr2 * (rr2 + a2 * cos_th_2))) * (((rr2 + a2) * (rr2 + a2 * cos_2th) - a2 * (a2 + 2.0 * M2 - 8.0 * M * rr + 4.0 * rr2 + 2.0 * (a - M) * (a + M) * cos_2th) * sin_th_4) / ((rr2 - 2.0 * M * rr + a2 * cos_th_2) * (rr2 - 2.0 * M * rr + M2 + (a - M) * (a + M) * cos_th_2))));

	double kerr_d0 = 2.0 * (((rr2 - M * rr) / (a2 + 2.0 * rr2 - 4.0 * M * rr + a2 * cos_2th) - rr2 / (a2 + 2.0 * rr2 + a2 * cos_2th))
		 + ((4.0 * rr2 * (a2 + 2.0 * rr2 + a2 * cos_2th) / (3.0 * a4 + 8.0 * rr4 + a2 * (  6.0 * M2 + 8.0 * rr2) + a2 * ((4.0 * a2 -        M2 + 8.0 * rr2) * cos_2th + (a2 - 6.0 * M2) * cos_4th + M2 * cos_6th)))
		   -(4.0 * rr2 * (a2 + 2.0 * rr2 + a2 * cos_2th) / (3.0 * a4 + 8.0 * rr4 + a2 * (-10.0 * M2 + 8.0 * rr2) + a2 * ((4.0 * a2 + 15.0 * M2 + 8.0 * rr2) * cos_2th + (a2 - 6.0 * M2) * cos_4th + M2 * cos_6th))) ));

	double kerr_d1 = -(-0.125 * J * (a2 + 2.0 * rr2 + a2 * cos_2th) * (3.0 * a4 + 16.0 * a2 * rr2 + 8.0 * rr2 * (3.0 * rr2 - 8.0 * M * rr) + 4.0 * a2 * (a2 + 4.0 * rr2) * cos_2th + a4 * cos_4th) + 4.0 * J * J * J * (a2 - 2.0 * rr2 + a2 * cos_2th) * sin_th_6) / (rr * pow((-4.0 * J * J * sin_th_6 + pow(a2 * cos_th_2 + rr2, 2)), 2));
	double kerr_d2 = (-rr2 / (rr2 + a2 * cos_th_2) + (M * rr - rr2) / (rr2 - 2.0 * M * rr + a2 * cos_th_2) + rr2 / (rr2 + a2 * cos_th_2 - 2.0 * a * M * sin_th_3) + rr2 / (rr2 + a2 * cos_th_2 + 2.0 * a * M * sin_th_3));
	double kerr_d3 = (M * rr) * ((M - rr) * rr + (a2 - M * rr) * cos_th_2) / ((rr2 + a2 * cos_th_2) * (M2 + rr2 - 2.0 * M * rr + (a2 - M2) * cos_th_2));
	double kerr_d5 = 2.0 * kerr_lambda * (-1.0 + (M * rr - rr2) / (rr2 - 2.0 * M * rr + a2 * cos_th_2) + (M * rr - rr2) / (M2 - 2.0 * M * rr + rr2 + (a2 - M2) * cos_th_2) + rr2 / (rr2 + a2 * cos_th_2) + (rr2 * (a2 + 2.0 * rr2 + a2 * cos_2th) + 4.0 * a2 * (M * rr - rr2) * sin_th_4) / ((a2 + rr2) * (rr2 + a2 * cos_2th) - a2 * (a2 + 2.0 * M2 - 8.0 * M * rr + 4.0 * rr2 + 2.0 * (a2 - M2) * cos_2th) * sin_th_4));
#endif

	// Robin and exponential decay boundary conditions.
	f[IDX(i, j)] = rescale * scale * (r * Dr_l_alpha + z * Dz_l_alpha + l_alpha
#ifdef ANALYTIC
		-(kerr_d0 + 0.5 * log(kerr_alpha2))
#else
		//-(M2 / rr2)
#endif
		);
	
	f[dim + IDX(i, j)] = rescale * scale * (r * Dr_beta + z * Dz_beta + 3.0 * beta
#ifdef ANALYTIC
		-(kerr_d1 + 3.0 * kerr_Omega)
#else
		//-(-4.0 * J * M / rr4)
#endif
		);

	f[2 * dim + IDX(i, j)] = rescale * scale * (r * Dr_l_h + z * Dz_l_h + l_h
#ifdef ANALYTIC
		-(kerr_d2 + 0.5 * log(kerr_H))
#else
		//-(-M2 / rr2)
#endif
		);

	f[3 * dim + IDX(i, j)] = rescale * scale * (r * Dr_l_a + z * Dz_l_a + l_a
#ifdef ANALYTIC
		-(kerr_d3 + 0.5 * log(kerr_A))
#else
		//-(-0.5 * M2 * (1.0 + cos_th_2) / rr2)
#endif
		);

	f[4 * dim + IDX(i, j)] = rescale * scale * (r * Dr_psi + z * Dz_psi + (rr * chi + l + 1.0) * psi);

	f[5 * dim + IDX(i, j)] = rescale * scale * (r * Dr_lambda + z * Dz_lambda + 4.0 * lambda
#ifdef ANALYTIC
		-(kerr_d5 + 4.0 * kerr_lambda)
#else
		//-(4.0 * M * M2 / (rr * rr4))
#endif
		);

	// All done.
	return;
}