// Include headers.
#include "tools.h"
#include "omega_calc.h"

const double D10 = -0.5;
const double D11 = 0.0;
const double D12 = +0.5;

const double D20 = +1.0;
const double D21 = -2.0;
const double D22 = +1.0;


// Calculate u1 = log(alpha) part.
void f1(double *a, MKL_INT *ia, MKL_INT *ja,
		const MKL_INT offset, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
		const MKL_INT i, const MKL_INT j, const double dr, const double dz,
		const MKL_INT l, const double m, const double chi,
		const double u101, const double u110, const double u111, const double u112, const double u121,
		const double u201, const double u210, const double u211, const double u212, const double u221,
		const double u301, const double u310, const double u311, const double u312, const double u321,
		const double u401, const double u410, const double u411, const double u412, const double u421,
		const double u501, const double u510, const double u511, const double u512, const double u521)

{
	// Omega.
	double w = omega_calc(chi, m);
	// Set rho normalized value. Called "r integer".
	// True rho value is r = dr * ri.
	double ri = (double)i - 0.5;
	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	// Some other constants.
	double dr2 = dr * dr;
	double r2 = ri * ri * dr2;
	double m2 = m * m;
	double rlm1 = (l == 1) ? 1.0 : pow(ri * dr, l - 1);
	double rl = rlm1 * (ri * dr);
	double alpha2 = exp(2.0 * u111);
	double h2 = exp(2.0 * u311);
	double a2 = exp(2.0 * u411);
	double wplOmega = w + l * u211;
	double wplOmega2 = wplOmega * wplOmega;
	double phior = rlm1 * u511;
	double phi = phior * (ri * dr);
	//double phi2or2 = phior * phior;
	double phi2 = phi * phi;
	// Finite differences.
	double dRu1 = 0.5 * (u121 - u101);
	double dRu2 = 0.5 * (u221 - u201);
	double dRu3 = 0.5 * (u321 - u301);
	//double dRu4 = 0.5 * (u421 - u401);
	//double dRu5 = 0.5 * (u521 - u501);
	double dZu1 = 0.5 * (u112 - u110);
	double dZu2 = 0.5 * (u212 - u210);
	double dZu3 = 0.5 * (u312 - u310);
	//double dZu4 = 0.5 * (u412 - u410);
	//double dZu5 = 0.5 * (u512 - u510);
	// Quadratic omega finite difference term.
	double r2h2oalpha2dOmega2 = r2 * h2 * (dzodr * dRu2 * dRu2 + drodz * dZu2 * dZu2) / alpha2;

	// Row start at offset. This is grid 1.
	ia[IDX(i, j)] = BASE + offset;

	// Set values.
	a[offset     ] = dzodr*((D20) + (D10)*(1.0/ri + 2.0*dRu1 + dRu3));
	a[offset +  1] = drodz*((D20) + (D10)*(2.0*dZu1 + dZu3));
	a[offset +  2] = (D21)*(dzodr + drodz) + r2h2oalpha2dOmega2 
		+ 16.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	a[offset +  3] = drodz*2.0*(D22) - a[offset + 1];
	a[offset +  4] = dzodr*2.0*(D22) - a[offset    ];
	a[offset +  5] = dzodr*((D10)*(-r2*h2*dRu2/alpha2));
	a[offset +  6] = drodz*((D10)*(-r2*h2*dZu2/alpha2));
	a[offset +  7] = -16.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	a[offset +  8] = -a[offset + 6];
	a[offset +  9] = -a[offset + 5];
	a[offset + 10] = dzodr*((D10)*dRu1);
	a[offset + 11] = drodz*((D10)*dZu1);
	a[offset + 12] = -r2h2oalpha2dOmega2;
	a[offset + 13] = -a[offset + 11];
	a[offset + 14] = -a[offset + 10];
	a[offset + 15] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;
	a[offset + 16] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*rl*phi;
	a[offset + 17] = dw_du(chi, m) * (-16.0*M_PI*dr2*dzodr*a2*wplOmega*phi2/alpha2);

	// Set column indices.
	ja[offset     ] = BASE +           IDX(i - 1, j    );
	ja[offset +  1] = BASE +           IDX(i    , j - 1);
	ja[offset +  2] = BASE +           IDX(i    , j    );
	ja[offset +  3] = BASE +           IDX(i    , j + 1);
	ja[offset +  4] = BASE +           IDX(i + 1, j    );
	ja[offset +  5] = BASE +     dim + IDX(i - 1, j    );
	ja[offset +  6] = BASE +     dim + IDX(i    , j - 1);
	ja[offset +  7] = BASE +     dim + IDX(i    , j    );
	ja[offset +  8] = BASE +     dim + IDX(i    , j + 1);
	ja[offset +  9] = BASE +     dim + IDX(i + 1, j    );
	ja[offset + 10] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset + 11] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset + 12] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset + 13] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset + 14] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset + 15] = BASE + 3 * dim + IDX(i    , j    );
	ja[offset + 16] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset + 17] = BASE + 5 * dim;

	// All done.
	return;
}

// Calculate u2 = Omega part.
void f2(double *a, MKL_INT *ia, MKL_INT *ja,
		const MKL_INT offset, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
		const MKL_INT i, const MKL_INT j, const double dr, const double dz,
		const MKL_INT l, const double m, const double chi,
		const double u101, const double u110, const double u111, const double u112, const double u121,
		const double u201, const double u210, const double u211, const double u212, const double u221,
		const double u301, const double u310, const double u311, const double u312, const double u321,
		const double u401, const double u410, const double u411, const double u412, const double u421,
		const double u501, const double u510, const double u511, const double u512, const double u521)

{
	// Omega.
	double w = omega_calc(chi, m);
	// Set rho normalized value. Called "r integer".
	// True rho value is r = dr * ri.
	double ri = (double)i - 0.5;
	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	// Some other constants.
	double dr2 = dr * dr;
	//double r2 = ri * ri * dr2;
	//double m2 = m * m;
	double rlm1 = (l == 1) ? 1.0 : pow(ri * dr, l - 1);
	//double rl = rlm1 * (ri * dr);
	//double alpha2 = exp(2.0 * u111);
	double h2 = exp(2.0 * u311);
	double a2 = exp(2.0 * u411);
	double wplOmega = w + l * u211;
	//double wplOmega2 = wplOmega * wplOmega;
	double phior = rlm1 * u511;
	//double phi = phior * (ri * dr);
	double phi2or2 = phior * phior;
	//double phi2 = phi * phi;
	// Finite differences.
	double dRu1 = 0.5 * (u121 - u101);
	double dRu2 = 0.5 * (u221 - u201);
	double dRu3 = 0.5 * (u321 - u301);
	//double dRu4 = 0.5 * (u421 - u401);
	//double dRu5 = 0.5 * (u521 - u501);
	double dZu1 = 0.5 * (u112 - u110);
	double dZu2 = 0.5 * (u212 - u210);
	double dZu3 = 0.5 * (u312 - u310);
	//double dZu4 = 0.5 * (u412 - u410);
	//double dZu5 = 0.5 * (u512 - u510);
	// Quadratic omega finite difference term.
	//double r2h2oalpha2dOmega2 = r2 * h2 * (dzodr * dRu2 * dRu2 + drodz * dZu2 * dZu2) / alpha2;

	// Row start at offset. This is grid 2.
	ia[dim + IDX(i, j)] = BASE + offset;

	// Set values.
	a[offset     ] = dzodr*((D10)*(-dRu2));
	a[offset +  1] = drodz*((D10)*(-dZu2));
	a[offset +  2] = -a[offset + 1];
	a[offset +  3] = -a[offset    ];
	a[offset +  4] = dzodr*((D20) + (D10)*(3.0/ri - dRu1 + 3.0*dRu3));
	a[offset +  5] = drodz*((D20) + (D10)*(-dZu1 + 3.0*dZu3));
	a[offset +  6] = (D21)*(drodz + dzodr) - 16.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	a[offset +  7] = drodz*2.0*(D22) -a[offset + 5];
	a[offset +  8] = dzodr*2.0*(D22) -a[offset + 4];
	a[offset +  9] = dzodr*((D10)*(3.0*dRu2));
	a[offset + 10] = drodz*((D10)*(3.0*dZu2));
	a[offset + 11] = 32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;
	a[offset + 12] = -a[offset + 10];
	a[offset + 13] = -a[offset +  9];
	a[offset + 14] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;
	a[offset + 15] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phior*rlm1/h2;
	a[offset + 16] = dw_du(chi, m) * (-16.0*M_PI*dr2*dzodr*a2*l*phi2or2/h2);

	// Set column indices.
	ja[offset     ] = BASE +           IDX(i - 1, j    );
	ja[offset +  1] = BASE +           IDX(i    , j - 1);
	ja[offset +  2] = BASE +           IDX(i    , j + 1);
	ja[offset +  3] = BASE +           IDX(i + 1, j    );
	ja[offset +  4] = BASE +     dim + IDX(i - 1, j    );
	ja[offset +  5] = BASE +     dim + IDX(i    , j - 1);
	ja[offset +  6] = BASE +     dim + IDX(i    , j    );
	ja[offset +  7] = BASE +     dim + IDX(i    , j + 1);
	ja[offset +  8] = BASE +     dim + IDX(i + 1, j    );
	ja[offset +  9] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset + 10] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset + 11] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset + 12] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset + 13] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset + 14] = BASE + 3 * dim + IDX(i    , j    );
	ja[offset + 15] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset + 16] = BASE + 5 * dim;

	// All done.
	return;
}

// Calculate u3 = log(h) part.
void f3(double *a, MKL_INT *ia, MKL_INT *ja,
		const MKL_INT offset, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
		const MKL_INT i, const MKL_INT j, const double dr, const double dz,
		const MKL_INT l, const double m, const double chi,
		const double u101, const double u110, const double u111, const double u112, const double u121,
		const double u201, const double u210, const double u211, const double u212, const double u221,
		const double u301, const double u310, const double u311, const double u312, const double u321,
		const double u401, const double u410, const double u411, const double u412, const double u421,
		const double u501, const double u510, const double u511, const double u512, const double u521)

{
	// Omega.
	//double w = omega_calc(chi, m);
	// Set rho normalized value. Called "r integer".
	// True rho value is r = dr * ri.
	double ri = (double)i - 0.5;
	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	// Some other constants.
	double dr2 = dr * dr;
	double r2 = ri * ri * dr2;
	double m2 = m * m;
	double rlm1 = (l == 1) ? 1.0 : pow(ri * dr, l - 1);
	//double rl = rlm1 * (ri * dr);
	double alpha2 = exp(2.0 * u111);
	double h2 = exp(2.0 * u311);
	double a2 = exp(2.0 * u411);
	//double wplOmega = w + l * u211;
	//double wplOmega2 = wplOmega * wplOmega;
	double phior = rlm1 * u511;
	//double phi = phior * (ri * dr);
	double phi2or2 = phior * phior;
	//double phi2 = phi * phi;
	// Finite differences.
	double dRu1 = 0.5 * (u121 - u101);
	double dRu2 = 0.5 * (u221 - u201);
	double dRu3 = 0.5 * (u321 - u301);
	//double dRu4 = 0.5 * (u421 - u401);
	//double dRu5 = 0.5 * (u521 - u501);
	double dZu1 = 0.5 * (u112 - u110);
	double dZu2 = 0.5 * (u212 - u210);
	double dZu3 = 0.5 * (u312 - u310);
	//double dZu4 = 0.5 * (u412 - u410);
	//double dZu5 = 0.5 * (u512 - u510);
	// Quadratic omega finite difference term.
	double r2h2oalpha2dOmega2 = r2 * h2 * (dzodr * dRu2 * dRu2 + drodz * dZu2 * dZu2) / alpha2;

	// Row start at offset. This is grid 3.
	ia[2 * dim + IDX(i, j)] = BASE + offset;

	// Set values.
	a[offset     ] = dzodr*((D10)*(1.0/ri + dRu3));
	a[offset +  1] = drodz*((D10)*dZu3);
	a[offset +  2] = -r2h2oalpha2dOmega2;
	a[offset +  3] = -a[offset + 1];
	a[offset +  4] = -a[offset    ];
	a[offset +  5] = dzodr*((D10)*(r2*h2*dRu2/alpha2));
	a[offset +  6] = drodz*((D10)*(r2*h2*dZu2/alpha2));
	a[offset +  7] = -a[offset + 6];
	a[offset +  8] = -a[offset + 5];
	a[offset +  9] = dzodr*((D20) + (D10)*(2.0/ri + dRu1 + 2.0*dRu3));
	a[offset + 10] = drodz*((D20) + (D10)*(dZu1 + 2.0*dZu3));
	a[offset + 11] = (D21)*(drodz + dzodr) + r2h2oalpha2dOmega2 
		- 16.0*M_PI*dr2*dzodr*a2*l*l*phi2or2/h2;
	a[offset + 12] = drodz*2.0*(D22) -a[offset + 10];
	a[offset + 13] = dzodr*2.0*(D22) -a[offset +  9];
	a[offset + 14] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;
	a[offset + 15] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phior*rlm1;

	// Set column indices.
	ja[offset     ] = BASE +           IDX(i - 1, j    );
	ja[offset +  1] = BASE +           IDX(i    , j - 1);
	ja[offset +  2] = BASE +           IDX(i    , j    );
	ja[offset +  3] = BASE +           IDX(i    , j + 1);
	ja[offset +  4] = BASE +           IDX(i + 1, j    );
	ja[offset +  5] = BASE +     dim + IDX(i - 1, j    );
	ja[offset +  6] = BASE +     dim + IDX(i    , j - 1);
	ja[offset +  7] = BASE +     dim + IDX(i    , j + 1);
	ja[offset +  8] = BASE +     dim + IDX(i + 1, j    );
	ja[offset +  9] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset + 10] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset + 11] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset + 12] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset + 13] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset + 14] = BASE + 3 * dim + IDX(i    , j    );
	ja[offset + 15] = BASE + 4 * dim + IDX(i    , j    );

	// All done.
	return;
}

// Calculate u4 = log(a) part.
void f4(double *a, MKL_INT *ia, MKL_INT *ja,
		const MKL_INT offset, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
		const MKL_INT i, const MKL_INT j, const double dr, const double dz,
		const MKL_INT l, const double m, const double chi,
		const double u101, const double u110, const double u111, const double u112, const double u121,
		const double u201, const double u210, const double u211, const double u212, const double u221,
		const double u301, const double u310, const double u311, const double u312, const double u321,
		const double u401, const double u410, const double u411, const double u412, const double u421,
		const double u501, const double u510, const double u511, const double u512, const double u521)

{
	// Omega.
	double w = omega_calc(chi, m);
	// Set rho normalized value. Called "r integer".
	// True rho value is r = dr * ri.
	double ri = (double)i - 0.5;
	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	// Some other constants.
	double dr2 = dr * dr;
	double r2 = ri * ri * dr2;
	//double m2 = m * m;
	double rlm1 = (l == 1) ? 1.0 : pow(ri * dr, l - 1);
	double rl = rlm1 * (ri * dr);
	double alpha2 = exp(2.0 * u111);
	double h2 = exp(2.0 * u311);
	double a2 = exp(2.0 * u411);
	double wplOmega = w + l * u211;
	double wplOmega2 = wplOmega * wplOmega;
	double psi = u511;
	double phior = rlm1 * psi;
	double phi = phior * (ri * dr);
	double phi2or2 = phior * phior;
	double phi2 = phi * phi;
	// Finite differences.
	double dRu1 = 0.5 * (u121 - u101);
	double dRu2 = 0.5 * (u221 - u201);
	double dRu3 = 0.5 * (u321 - u301);
	//double dRu4 = 0.5 * (u421 - u401);
	double dRu5 = 0.5 * (u521 - u501);
	double dZu1 = 0.5 * (u112 - u110);
	double dZu2 = 0.5 * (u212 - u210);
	double dZu3 = 0.5 * (u312 - u310);
	//double dZu4 = 0.5 * (u412 - u410);
	double dZu5 = 0.5 * (u512 - u510);
	// Quadratic omega finite difference term.
	double r2h2oalpha2dOmega2 = r2 * h2 * (dzodr * dRu2 * dRu2 + drodz * dZu2 * dZu2) / alpha2;

	// Row start at offset. This is grid 4.
	ia[3 * dim + IDX(i, j)] = BASE + offset;

	// Set values.
	a[offset     ] = dzodr*((D10)*(-1.0/ri - dRu3));
	a[offset +  1] = drodz*((D10)*(-dZu3));
	a[offset +  2] = 0.5*r2h2oalpha2dOmega2 - 8.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	a[offset +  3] = -a[offset + 1];
	a[offset +  4] = -a[offset    ];
	a[offset +  5] = dzodr*((D10)*(-0.5*r2*h2*dRu2/alpha2));
	a[offset +  6] = drodz*((D10)*(-0.5*r2*h2*dZu2/alpha2));
	a[offset +  7] = 8.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	a[offset +  8] = -a[offset + 6];
	a[offset +  9] = -a[offset + 5];
	a[offset + 10] = dzodr*((D10)*(-dRu1));
	a[offset + 11] = drodz*((D10)*(-dZu1));
	a[offset + 12] = -0.5*r2h2oalpha2dOmega2 + 8.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	a[offset + 13] = -a[offset + 11];
	a[offset + 14] = -a[offset + 10];
	a[offset + 15] = dzodr*(D20);// CONSTANT!
	a[offset + 16] = drodz*(D20);// CONSTANT!
	a[offset + 17] = (D21)*(drodz + dzodr) + 8.0*M_PI*dr2*dzodr*(-l*l/h2 + r2*wplOmega2/alpha2)*a2*phi2or2;
	a[offset + 18] = drodz*(D22);// CONSTANT!
	a[offset + 19] = dzodr*(D22);// CONSTANT!
	a[offset + 20] = dzodr*(D10)*(8.0*M_PI*rl*rl*(l*psi/ri + dRu5));
	a[offset + 21] = drodz*(D10)*(8.0*M_PI*rl*rl*dZu5);
	a[offset + 22] = 8.0*M_PI*dr2*dzodr*rlm1*rlm1*(l*ri*dRu5 + (a2*wplOmega2*r2/alpha2 + l*l*(1.0 - a2/h2))*psi);
	a[offset + 23] = -a[offset + 21];
	a[offset + 24] = -a[offset + 20];
	a[offset + 25] = dw_du(chi, m) * (8.0*M_PI*dr2*dzodr*a2*wplOmega*phi2/alpha2);

	// Set column indices.
	ja[offset     ] = BASE +           IDX(i - 1, j    );
	ja[offset +  1] = BASE +           IDX(i    , j - 1);
	ja[offset +  2] = BASE +           IDX(i    , j    );
	ja[offset +  3] = BASE +           IDX(i    , j + 1);
	ja[offset +  4] = BASE +           IDX(i + 1, j    );
	ja[offset +  5] = BASE +     dim + IDX(i - 1, j    );
	ja[offset +  6] = BASE +     dim + IDX(i    , j - 1);
	ja[offset +  7] = BASE +     dim + IDX(i    , j    );
	ja[offset +  8] = BASE +     dim + IDX(i    , j + 1);
	ja[offset +  9] = BASE +     dim + IDX(i + 1, j    );
	ja[offset + 10] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset + 11] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset + 12] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset + 13] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset + 14] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset + 15] = BASE + 3 * dim + IDX(i - 1, j    );
	ja[offset + 16] = BASE + 3 * dim + IDX(i    , j - 1);
	ja[offset + 17] = BASE + 3 * dim + IDX(i    , j    );
	ja[offset + 18] = BASE + 3 * dim + IDX(i    , j + 1);
	ja[offset + 19] = BASE + 3 * dim + IDX(i + 1, j    );
	ja[offset + 20] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset + 21] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset + 22] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset + 23] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset + 24] = BASE + 4 * dim + IDX(i + 1, j    );
	ja[offset + 25] = BASE + 5 * dim;

	// All done.
	return;
}

// Calculate u5 = phi / r**l part.
void f5(double *a, MKL_INT *ia, MKL_INT *ja,
		const MKL_INT offset, const MKL_INT NrTotal, const MKL_INT NzTotal, const MKL_INT dim,
		const MKL_INT i, const MKL_INT j, const double dr, const double dz,
		const MKL_INT l, const double m, const double chi,
		const double u101, const double u110, const double u111, const double u112, const double u121,
		const double u201, const double u210, const double u211, const double u212, const double u221,
		const double u301, const double u310, const double u311, const double u312, const double u321,
		const double u401, const double u410, const double u411, const double u412, const double u421,
		const double u501, const double u510, const double u511, const double u512, const double u521)

{
	// Omega.
	double w = omega_calc(chi, m);
	// Set rho normalized value. Called "r integer".
	// True rho value is r = dr * ri.
	double ri = (double)i - 0.5;
	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	// Some other constants.
	double dr2 = dr * dr;
	//double r2 = ri * ri * dr2;
	double m2 = m * m;
	//double rlm1 = (l == 1) ? 1.0 : pow(ri * dr, l - 1);
	//double rl = rlm1 * (ri * dr);
	double alpha2 = exp(2.0 * u111);
	double h2 = exp(2.0 * u311);
	double a2 = exp(2.0 * u411);
	double wplOmega = w + l * u211;
	double wplOmega2 = wplOmega * wplOmega;
	double psi = u511;
	//double phior = rlm1 * psi;
	//double phi = phior * (ri * dr);
	//double phi2or2 = phior * phior;
	//double phi2 = phi * phi;
	// Finite differences.
	double dRu1 = 0.5 * (u121 - u101);
	//double dRu2 = 0.5 * (u221 - u201);
	double dRu3 = 0.5 * (u321 - u301);
	//double dRu4 = 0.5 * (u421 - u401);
	double dRu5 = 0.5 * (u521 - u501);
	double dZu1 = 0.5 * (u112 - u110);
	//double dZu2 = 0.5 * (u212 - u210);
	double dZu3 = 0.5 * (u312 - u310);
	//double dZu4 = 0.5 * (u412 - u410);
	double dZu5 = 0.5 * (u512 - u510);
	// Quadratic omega finite difference term.
	//double r2h2oalpha2dOmega2 = r2 * h2 * (dzodr * dRu2 * dRu2 + drodz * dZu2 * dZu2) / alpha2;
	
	// Row start at offset. This is grid 5.
	ia[4 * dim + IDX(i, j)] = BASE + offset;

	// Set values.
	a[offset     ] = dzodr*((D10)*(l*psi/ri + dRu5));
	a[offset +  1] = drodz*((D10)*dZu5);
	a[offset +  2] = -2.0*dr2*dzodr*a2*wplOmega2*psi/alpha2;
	a[offset +  3] = -a[offset + 1];
	a[offset +  4] = -a[offset    ];
	a[offset +  5] = 2.0*dr2*dzodr*l*a2*wplOmega*psi/alpha2;
	a[offset +  6] = dzodr*((D10)*(l*psi/ri + dRu5));
	a[offset +  7] = drodz*((D10)*dZu5);
	a[offset +  8] = 2.0*dzodr*l*l*(a2/h2)*psi/(ri*ri);
	a[offset +  9] = -a[offset + 7];
	a[offset + 10] = -a[offset + 6];
	a[offset + 11] = 2.0*dzodr*(dr2*a2*(wplOmega2/alpha2 - m2) - l*l*(a2/h2)/(ri*ri))*psi;
	a[offset + 12] = dzodr*((D20) + (D10)*((2.0*l + 1.0)/ri + dRu1 + dRu3));
	a[offset + 13] = drodz*((D20) + (D10)*(dZu1 + dZu3));
	a[offset + 14] = (D21)*(drodz + dzodr) + dzodr * (l * (dRu1 / ri + dRu3 / ri) 
			+ (dr2 * a2 * (wplOmega2 / alpha2 - m2) - l * l * ((a2 - h2) / (ri * ri)) / h2));
	a[offset + 15] = drodz*2.0*(D22) -a[offset + 13];
	a[offset + 16] = dzodr*2.0*(D22) -a[offset + 12];
	a[offset + 17] = dw_du(chi, m) * (2.0*dr2*dzodr*a2*wplOmega*psi/alpha2);

	// Set column indices.
	ja[offset     ] = BASE +           IDX(i - 1, j    );
	ja[offset +  1] = BASE +           IDX(i    , j - 1);
	ja[offset +  2] = BASE +           IDX(i    , j    );
	ja[offset +  3] = BASE +           IDX(i    , j + 1);
	ja[offset +  4] = BASE +           IDX(i + 1, j    );
	ja[offset +  5] = BASE +     dim + IDX(i    , j    );
	ja[offset +  6] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset +  7] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset +  8] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset +  9] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset + 10] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset + 11] = BASE + 3 * dim + IDX(i    , j    );
	ja[offset + 12] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset + 13] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset + 14] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset + 15] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset + 16] = BASE + 4 * dim + IDX(i + 1, j    );
	ja[offset + 17] = BASE + 5 * dim;

	// All done.
	return;
}
