#include "tools.h"
#include "omega_calc.h"

void rhs_vars(
	double *f, 
	const double *u, 
	const double *Dr_u,
	const double *Dz_u,
	const double *Drr_u, 
	const double *Dzz_u,
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
	const double rescale
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
	double alpha2, h2, a2, w_plus_l_beta, w_plus_l_beta2;
	double phi, phi_over_r, phi2, phi2_over_r2;

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

	// Coordinates.
	r = dr * (i + 0.5 - ghost);
	r2 = r * r;
	//z = dz * (j + 0.5 - ghost);
	//z2 = z * z;
	//rr = sqrt(r2 + z2);
	rlm1 = (l ==1) ? 1.0 : pow(r, l - 1);
	rl = rlm1 * r;

	// Auxiliary variables.
	alpha2 = exp(2.0 * l_alpha);
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

	// u1 = log(alpha).
	f[IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_l_alpha + Dzz_l_alpha + (Dr_l_alpha / r)
		+ D_l_alpha_D_l_alpha + D_l_alpha_D_l_h 
		- 0.5 * r2_h2_over_alpha2_D_beta_D_beta 
		+ 4.0 * M_PI * a2 * (m2 - 2.0 * w_plus_l_beta2 / alpha2) * phi2));

	// u2 = beta.
	f[dim + IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_beta + Dzz_beta + 3.0 * (Dr_beta / r)
		- D_l_alpha_D_beta + 3.0 * D_beta_D_l_h
		- 16.0 * M_PI * a2 * l * w_plus_l_beta * phi2_over_r2 / h2));

	// u3 = log(h).
	f[2 * dim + IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_l_h + Dzz_l_h + 2.0 * (Dr_l_h / r)
		+ D_l_h_D_l_h + D_l_alpha_D_l_h + 0.5 * r2_h2_over_alpha2_D_beta_D_beta
		+ (Dr_l_alpha / r) 
		+ 4.0 * M_PI * a2 * (r2 * m2 + 2.0 * l * l / h2) * phi2_over_r2));

	// u4 = log(a).
	f[3 * dim + IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_l_a + Dzz_l_a
		- D_l_alpha_D_l_h - 0.25 * r2_h2_over_alpha2_D_beta_D_beta 
		- (Dr_l_alpha / r)
		+ 4.0 * M_PI * ((l * l * (1.0 - a2 / h2) + a2 * r2 * w_plus_l_beta2 / alpha2) * phi2_over_r2
			+ (2.0 * l * r * psi * Dr_psi + r2 * D_psi_D_psi) * (rlm1 * rlm1))));

	// u5 = log(phi / r**l)
	f[4 * dim + IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_psi + Dzz_psi + (2.0 * l + 1.0) * (Dr_psi / r)
		+ D_l_alpha_D_psi + D_l_h_D_psi
		+ l * ((Dr_l_alpha / r) + (Dr_l_h / r)) * psi
		+ a2 * (w_plus_l_beta2 / alpha2 - m2) * psi)
		- dzodr * l * l * ((a2 - h2) / (r2 / (dr * dr))) * psi / h2);

	/*
	// u4 = log(a).
	f[3 * dim + IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_l_a + Dzz_l_a
		- D_l_alpha_D_l_h - 0.25 * r2_h2_over_alpha2_D_beta_D_beta 
		- (Dr_l_alpha / r)
		+ 4.0 * M_PI * (a2 * (r2 * w_plus_l_beta2 / alpha2 - l * l / h2)
			+ r2 * (D_psi_D_psi + chi * (chi - 2.0 * Dx_psi))
			+ 2.0 * l * r * Dr_psi + l * l) * phi2_over_r2));

	// u5 = log(exp(+chi * rr) * phi / r**l)
	f[4 * dim + IDX(i, j)] = rescale * (dr * dr * dzodr * (Drr_psi + Dzz_psi + (2.0 * l + 1.0) * (Dr_psi / r)
		+ D_psi_D_psi + D_l_alpha_D_psi + D_l_h_D_psi
		+ l * ((Dr_l_alpha / r) + (Dr_l_h / r))
		+ a2 * (w_plus_l_beta2 / alpha2 - m2)
		+ chi * (chi - (2.0 * (l + 1.0)) / rr - 2.0 * Dx_psi - Dx_l_alpha - Dx_l_h))
		- dzodr * l * l * ((a2 - h2) / (r2 / (dr * dr))) / h2);
	*/

	// All done.
	return;
}

void rhs_bdry(
	double *f, 
	const double *u, 
	const double *Dr_u,
	const double *Dz_u,
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
	const double rescale
)
{
	// Auxiliary doubles.
	double r = dr * (i + 0.5 - ghost);
	double z = dz * (j + 0.5 - ghost);
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

	// Robin and exponential decay boundary conditions.
	f[IDX(i, j)] = rescale * scale * (r * Dr_l_alpha + z * Dz_l_alpha + l_alpha);
	
	f[dim + IDX(i, j)] = rescale * scale * (r * Dr_beta + z * Dz_beta + 3.0 * beta);

	f[2 * dim + IDX(i, j)] = rescale * scale * (r * Dr_l_h + z * Dz_l_h + l_h);

	f[3 * dim + IDX(i, j)] = rescale * scale * (r * Dr_l_a + z * Dz_l_a + l_a);

	f[4 * dim + IDX(i, j)] = rescale * scale * (r * Dr_psi + z * Dz_psi + (rr * chi + l + 1.0) * psi);
	//f[4 * dim + IDX(i, j)] = rescale * scale * (r * Dr_psi + z * Dz_psi + (l + 1.0));

	// All done.
	return;
}