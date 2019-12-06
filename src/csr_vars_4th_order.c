#include "tools.h"
#include "omega_calc.h"

const double D10 = 1.0 / 12.0;
const double D11 = -2.0 / 3.0;
const double D12 = 0.0;
const double D13 = +2.0 / 3.0;
const double D14 = -1.0 / 12.0;

const double D20 = -1.0 / 12.0;
const double D21 = 4.0 / 3.0;
const double D22 = -2.5;
const double D23 = 4.0 / 3.0;
const double D24 = -1.0 / 12.0;

const double S10 = 0.0;
const double S11 = -1.0 / 12.0;
const double S12 = +0.5;
const double S13 = -1.5;
const double S14 = +5.0 / 6.0;
const double S15 = +0.25;

const double S20 = 1.0 / 12.0;
const double S21 = -0.5;
const double S22 = +7.0 / 6.0;
const double S23 = -1.0 / 3.0;
const double S24 = -1.25;
const double S25 = +5.0 / 6.0;

// Jacobian for centered-centered 4th order stencil and variable omega.
void jacobian_4th_order_variable_omega_cc
(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings. 
	MKL_INT *ja,		// CSR array for columns.
	const MKL_INT NrTotal, 	// Grid total dimension in r.
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const double dr, 	// Spatial step in r.
	const double dz,	// Spatial step in z.
	const MKL_INT l, 	// Scalar field rotation number.
	const double m, 	// Scalar field mass.
	const double xi,	// Scalar field frequency variable.
	// Now come the grid variables. For cc stencil, each grid function has 9 variables.
	const double u102, const double u112, const double u120, const double u121, const double u122, const double u123, const double u124, const double u132, const double u142,
	const double u202, const double u212, const double u220, const double u221, const double u222, const double u223, const double u224, const double u232, const double u242,
	const double u302, const double u312, const double u320, const double u321, const double u322, const double u323, const double u324, const double u332, const double u342,
	const double u402, const double u412, const double u420, const double u421, const double u422, const double u423, const double u424, const double u432, const double u442,
	const double u502, const double u512, const double u520, const double u521, const double u522, const double u523, const double u524, const double u532, const double u542,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5 	// Number of elements filled before filling function 5.
)
{
	// Physical variables.
	double alpha = exp(u122);
	double Omega = u222;
	double h    = exp(u322);
	double a    = exp(u422);
	double psi  = u522;

	// Coordinates.
	double ri = (double)i - 1.5;
	double zi = (double)j - 1.5;
	double r = ri * dr;
	double z = zi * dz;
	double r2 = r * r;
	double z2 = z * z;
	double rr = sqrt(r2 + z2);

	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	double dr2 = dr * dr;
	double dz2 = dz * dz;

	// Scalar field mass and frequency.
	double w = omega_calc(xi, m);
	double m2 = m * m;
	double w2 = w * w;
	double chi = sqrt(m2 - w2);

	// Scalar field.
	double rlm1 = (l == 1) ? 1.0 : pow(r, l - 1);
	double rl = rlm1 * r;
	double phior = rlm1 * exp(-chi * rr) * exp(psi);
	double phi = r * phior;
	double phi2or2 = phior * phior;
	double phi2 = phi * phi;

	/// Shift combined with scalar field rotation.
	double wplOmega = w + l * Omega;
	double wplOmega2 = wplOmega * wplOmega;

	// Finite differences.
	double dRu1 = D10 * u102 + D11 * u112 + D13 * u132 + D14 * u142;
	double dRu2 = D10 * u202 + D11 * u212 + D13 * u232 + D14 * u242;
	double dRu3 = D10 * u302 + D11 * u312 + D13 * u332 + D14 * u342;
	double dRu4 = D10 * u402 + D11 * u412 + D13 * u432 + D14 * u442;
	double dRu5 = D10 * u502 + D11 * u512 + D13 * u532 + D14 * u542;

	double dZu1 = D10 * u120 + D11 * u121 + D13 * u123 + D14 * u124;
	double dZu2 = D10 * u220 + D11 * u221 + D13 * u223 + D14 * u224;
	double dZu3 = D10 * u320 + D11 * u321 + D13 * u323 + D14 * u324;
	double dZu4 = D10 * u420 + D11 * u421 + D13 * u423 + D14 * u424;
	double dZu5 = D10 * u520 + D11 * u521 + D13 * u523 + D14 * u524;

	// Radial derivatives.
	double dXu5 = (ri * dRu5 + zi * dZu5) / rr;
	double dXu1 = (ri * dRu1 + zi * dZu1) / rr;
	double dXu3 = (ri * dRu3 + zi * dZu3) / rr;

	// Squared variables.
	double alpha2 = alpha * alpha;
	double h2 = h * h;
	double a2 = a * a;

	// Common term.
	double r2h2oalpha2dOmega2 = r2 * h2 * (dzodr * dRu2 * dRu2 + drodz * dZu2 * dZu2) / alpha2;

	// Alpha: grid number 0.
	ia[IDX(i, j)] = BASE + offset1;

	// Values.
	aa[offset1 +  0] = dzodr*((D20) + (D10)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  1] = dzodr*((D21) + (D11)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  2] = drodz*((D20) + (D10)*(2.0*dZu1 + dZu3));
	aa[offset1 +  3] = drodz*((D21) + (D11)*(2.0*dZu1 + dZu3));
	aa[offset1 +  4] = (D22)*(dzodr + drodz) + r2h2oalpha2dOmega2 + 16.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset1 +  5] = 2.0*(D21)*drodz - (aa[offset1 +  3]);
	aa[offset1 +  6] = 2.0*(D20)*drodz - (aa[offset1 +  2]);
	aa[offset1 +  7] = 2.0*(D21)*dzodr - (aa[offset1 +  1]);
	aa[offset1 +  8] = 2.0*(D20)*dzodr - (aa[offset1 +  0]);

	aa[offset1 +  9] = dzodr*((D10)*(-r2*h2*dRu2/alpha2));
	aa[offset1 + 10] = dzodr*((D11)*(-r2*h2*dRu2/alpha2));
	aa[offset1 + 11] = drodz*((D10)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 12] = drodz*((D11)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 13] = -16.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset1 + 14] = -aa[offset1 + 12];
	aa[offset1 + 15] = -aa[offset1 + 11];
	aa[offset1 + 16] = -aa[offset1 + 10];
	aa[offset1 + 17] = -aa[offset1 +  9];

	aa[offset1 + 18] = dzodr*((D10)*dRu1);
	aa[offset1 + 19] = dzodr*((D11)*dRu1);
	aa[offset1 + 20] = drodz*((D10)*dZu1);
	aa[offset1 + 21] = drodz*((D11)*dZu1);
	aa[offset1 + 22] = -r2h2oalpha2dOmega2;
	aa[offset1 + 23] = -aa[offset1 + 21];
	aa[offset1 + 24] = -aa[offset1 + 20];
	aa[offset1 + 25] = -aa[offset1 + 19];
	aa[offset1 + 26] = -aa[offset1 + 18];

	aa[offset1 + 27] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;

	aa[offset1 + 28] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;

	aa[offset1 + 29] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*wplOmega*phi2/alpha2 + 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2*(rr*w/chi));

	// Columns.
	ja[offset1 +  0] = BASE +           IDX(i - 2, j    );
	ja[offset1 +  1] = BASE +           IDX(i - 1, j    );
	ja[offset1 +  2] = BASE +           IDX(i    , j - 2);
	ja[offset1 +  3] = BASE +           IDX(i    , j - 1);
	ja[offset1 +  4] = BASE +           IDX(i    , j    );
	ja[offset1 +  5] = BASE +           IDX(i    , j + 1);
	ja[offset1 +  6] = BASE +           IDX(i    , j + 2);
	ja[offset1 +  7] = BASE +           IDX(i + 1, j    );
	ja[offset1 +  8] = BASE +           IDX(i + 2, j    );

	ja[offset1 +  9] = BASE +     dim + IDX(i - 2, j    );
	ja[offset1 + 10] = BASE +     dim + IDX(i - 1, j    );
	ja[offset1 + 11] = BASE +     dim + IDX(i    , j - 2);
	ja[offset1 + 12] = BASE +     dim + IDX(i    , j - 1);
	ja[offset1 + 13] = BASE +     dim + IDX(i    , j    );
	ja[offset1 + 14] = BASE +     dim + IDX(i    , j + 1);
	ja[offset1 + 15] = BASE +     dim + IDX(i    , j + 2);
	ja[offset1 + 16] = BASE +     dim + IDX(i + 1, j    );
	ja[offset1 + 17] = BASE +     dim + IDX(i + 2, j    );

	ja[offset1 + 18] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset1 + 19] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset1 + 20] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset1 + 21] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset1 + 22] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset1 + 23] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset1 + 24] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset1 + 25] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset1 + 26] = BASE + 2 * dim + IDX(i + 2, j    );

	ja[offset1 + 27] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset1 + 28] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset1 + 29] = BASE + 5 * dim;


	// Beta: grid number 1.
	ia[dim + IDX(i, j)] = BASE + offset2;

	// Values.
	aa[offset2 +  0] = dzodr*((D10)*(-dRu2));
	aa[offset2 +  1] = dzodr*((D11)*(-dRu2));
	aa[offset2 +  2] = drodz*((D10)*(-dZu2));
	aa[offset2 +  3] = drodz*((D11)*(-dZu2));
	aa[offset2 +  4] = -aa[offset2 +  3];
	aa[offset2 +  5] = -aa[offset2 +  2];
	aa[offset2 +  6] = -aa[offset2 +  1];
	aa[offset2 +  7] = -aa[offset2 +  0];
	
	aa[offset2 +  8] = dzodr*((D20) + (D10)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 +  9] = dzodr*((D21) + (D11)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 + 10] = drodz*((D20) + (D10)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 11] = drodz*((D21) + (D11)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 12] = (D22)*(drodz + dzodr) - 16.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset2 + 13] = 2.0*(D21)*drodz - aa[offset2 + 11];
	aa[offset2 + 14] = 2.0*(D20)*drodz - aa[offset2 + 10];
	aa[offset2 + 15] = 2.0*(D21)*dzodr - aa[offset2 +  9];
	aa[offset2 + 16] = 2.0*(D20)*dzodr - aa[offset2 +  8];

	aa[offset2 + 17] = dzodr*((D10)*(3.0*dRu2));
	aa[offset2 + 18] = dzodr*((D11)*(3.0*dRu2));
	aa[offset2 + 19] = drodz*((D10)*(3.0*dZu2));
	aa[offset2 + 20] = drodz*((D11)*(3.0*dZu2));
	aa[offset2 + 21] = 32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;
	aa[offset2 + 22] = -aa[offset2 + 20];
	aa[offset2 + 23] = -aa[offset2 + 19];
	aa[offset2 + 24] = -aa[offset2 + 18];
	aa[offset2 + 25] = -aa[offset2 + 17];

	aa[offset2 + 26] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;

	aa[offset2 + 27] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;

	aa[offset2 + 28] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*l*phi2or2/h2 - 32.0*M_PI*dr2*dzodr*(a2/h2)*l*wplOmega*phi2or2*(rr*w/chi));

	// Columns.
	ja[offset2 +  0] = BASE +           IDX(i - 2, j    );
	ja[offset2 +  1] = BASE +           IDX(i - 1, j    );
	ja[offset2 +  2] = BASE +           IDX(i    , j - 2);
	ja[offset2 +  3] = BASE +           IDX(i    , j - 1);
	ja[offset2 +  4] = BASE +           IDX(i    , j + 1);
	ja[offset2 +  5] = BASE +           IDX(i    , j + 2);
	ja[offset2 +  6] = BASE +           IDX(i + 1, j    );
	ja[offset2 +  7] = BASE +           IDX(i + 2, j    );

	ja[offset2 +  8] = BASE +     dim + IDX(i - 2, j    );
	ja[offset2 +  9] = BASE +     dim + IDX(i - 1, j    );
	ja[offset2 + 10] = BASE +     dim + IDX(i    , j - 2);
	ja[offset2 + 11] = BASE +     dim + IDX(i    , j - 1);
	ja[offset2 + 12] = BASE +     dim + IDX(i    , j    );
	ja[offset2 + 13] = BASE +     dim + IDX(i    , j + 1);
	ja[offset2 + 14] = BASE +     dim + IDX(i    , j + 2);
	ja[offset2 + 15] = BASE +     dim + IDX(i + 1, j    );
	ja[offset2 + 16] = BASE +     dim + IDX(i + 2, j    );

	ja[offset2 + 17] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset2 + 18] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset2 + 19] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset2 + 20] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset2 + 21] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset2 + 22] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset2 + 23] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset2 + 24] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset2 + 25] = BASE + 2 * dim + IDX(i + 2, j    );

	ja[offset2 + 26] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset2 + 27] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset2 + 28] = BASE + 5 * dim;


	// H: grid number 2.
	ia[2 * dim + IDX(i, j)] = BASE + offset3;

	// Values.
	aa[offset3 +  0] = dzodr*((D10)*(1.0/ri + dRu3));
	aa[offset3 +  1] = dzodr*((D11)*(1.0/ri + dRu3));
	aa[offset3 +  2] = drodz*((D10)*dZu3);
	aa[offset3 +  3] = drodz*((D11)*dZu3);
	aa[offset3 +  4] = -r2h2oalpha2dOmega2;
	aa[offset3 +  5] = -aa[offset3 +  3];
	aa[offset3 +  6] = -aa[offset3 +  2];
	aa[offset3 +  7] = -aa[offset3 +  1];
	aa[offset3 +  8] = -aa[offset3 +  0];

	aa[offset3 +  9] = dzodr*((D10)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 10] = dzodr*((D11)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 11] = drodz*((D10)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 12] = drodz*((D11)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 13] = -aa[offset3 + 12];
	aa[offset3 + 14] = -aa[offset3 + 11];
	aa[offset3 + 15] = -aa[offset3 + 10];
	aa[offset3 + 16] = -aa[offset3 +  9];

	aa[offset3 + 17] = dzodr*((D20) + (D10)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 18] = dzodr*((D21) + (D11)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 19] = drodz*((D20) + (D10)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 20] = drodz*((D21) + (D11)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 21] = (D22)*(drodz + dzodr) + r2h2oalpha2dOmega2 - 16.0*M_PI*dr2*dzodr*a2*l*l*phi2or2/h2;
	aa[offset3 + 22] = 2.0*(D21)*drodz - aa[offset3 + 20];
	aa[offset3 + 23] = 2.0*(D20)*drodz - aa[offset3 + 19];
	aa[offset3 + 24] = 2.0*(D21)*dzodr - aa[offset3 + 18];
	aa[offset3 + 25] = 2.0*(D20)*dzodr - aa[offset3 + 17];

	aa[offset3 + 26] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;

	aa[offset3 + 27] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;

	aa[offset3 + 28] = dw_du(xi, m) * (8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2*(rr*w/chi));

	// Columns.
	ja[offset3 +  0] = BASE +           IDX(i - 2, j    );
	ja[offset3 +  1] = BASE +           IDX(i - 1, j    );
	ja[offset3 +  2] = BASE +           IDX(i    , j - 2);
	ja[offset3 +  3] = BASE +           IDX(i    , j - 1);
	ja[offset3 +  4] = BASE +           IDX(i    , j    );
	ja[offset3 +  5] = BASE +           IDX(i    , j + 1);
	ja[offset3 +  6] = BASE +           IDX(i    , j + 2);
	ja[offset3 +  7] = BASE +           IDX(i + 1, j    );
	ja[offset3 +  8] = BASE +           IDX(i + 2, j    );

	ja[offset3 +  9] = BASE +     dim + IDX(i - 2, j    );
	ja[offset3 + 10] = BASE +     dim + IDX(i - 1, j    );
	ja[offset3 + 11] = BASE +     dim + IDX(i    , j - 2);
	ja[offset3 + 12] = BASE +     dim + IDX(i    , j - 1);
	ja[offset3 + 13] = BASE +     dim + IDX(i    , j + 1);
	ja[offset3 + 14] = BASE +     dim + IDX(i    , j + 2);
	ja[offset3 + 15] = BASE +     dim + IDX(i + 1, j    );
	ja[offset3 + 16] = BASE +     dim + IDX(i + 2, j    );

	ja[offset3 + 17] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset3 + 18] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset3 + 19] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset3 + 20] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset3 + 21] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset3 + 22] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset3 + 23] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset3 + 24] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset3 + 25] = BASE + 2 * dim + IDX(i + 2, j    );

	ja[offset3 + 26] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset3 + 27] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset3 + 28] = BASE + 5 * dim;


	// A: grid number 3.
	ia[3 * dim + IDX(i, j)] = BASE + offset4;

	// Values.
	aa[offset4 +  0] = dzodr*((D10)*(-1.0/ri - dRu3));
	aa[offset4 +  1] = dzodr*((D11)*(-1.0/ri - dRu3));
	aa[offset4 +  2] = drodz*((D10)*(-dZu3));
	aa[offset4 +  3] = drodz*((D11)*(-dZu3));
	aa[offset4 +  4] = 0.5*r2h2oalpha2dOmega2 - 8.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset4 +  5] = -aa[offset4 +  3];
	aa[offset4 +  6] = -aa[offset4 +  2];
	aa[offset4 +  7] = -aa[offset4 +  1];
	aa[offset4 +  8] = -aa[offset4 +  0];

	aa[offset4 +  9] = dzodr*((D10)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 + 10] = dzodr*((D11)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 + 11] = drodz*((D10)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 12] = drodz*((D11)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 13] = 8.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset4 + 14] = -aa[offset4 + 12];
	aa[offset4 + 15] = -aa[offset4 + 11];
	aa[offset4 + 16] = -aa[offset4 + 10];
	aa[offset4 + 17] = -aa[offset4 +  9];

	aa[offset4 + 18] = dzodr*((D10)*(-dRu1));
	aa[offset4 + 19] = dzodr*((D11)*(-dRu1));
	aa[offset4 + 20] = drodz*((D10)*(-dZu1));
	aa[offset4 + 21] = drodz*((D11)*(-dZu1));
	aa[offset4 + 22] = -0.5*r2h2oalpha2dOmega2 + 8.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset4 + 23] = -aa[offset4 + 21];
	aa[offset4 + 24] = -aa[offset4 + 20];
	aa[offset4 + 25] = -aa[offset4 + 19];
	aa[offset4 + 26] = -aa[offset4 + 18];

	aa[offset4 + 27] = (D20)*dzodr; // CONSTANT!
	aa[offset4 + 28] = (D21)*dzodr; // CONSTANT!
	aa[offset4 + 29] = (D20)*drodz; // CONSTANT!
	aa[offset4 + 30] = (D21)*drodz; // CONSTANT!
	aa[offset4 + 31] = (D22)*(drodz + dzodr) + 8.0*M_PI*dr2*dzodr*(-l*l/h2 + r2*wplOmega2/alpha2)*a2*phi2or2;
	aa[offset4 + 32] = (D21)*drodz; // CONSTANT!
	aa[offset4 + 33] = (D20)*drodz; // CONSTANT!
	aa[offset4 + 34] = (D21)*dzodr; // CONSTANT!
	aa[offset4 + 35] = (D20)*dzodr; // CONSTANT!

	aa[offset4 + 36] = dzodr*((D10)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 37] = dzodr*((D11)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 38] = drodz*((D10)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 39] = drodz*((D11)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 40] = 8.0*M_PI*phi2or2*(dr2*dzodr*(l*l + 2.0*l*ri*dRu5 + a2*(-l*l/h2 + r2*wplOmega2/alpha2) + r2*(chi*(chi - 2.0*dXu5))) + (r2*(dzodr*dRu5*dRu5 + drodz*dZu5*dZu5)));
	aa[offset4 + 41] = -aa[offset4 + 39];
	aa[offset4 + 42] = -aa[offset4 + 38];
	aa[offset4 + 43] = -aa[offset4 + 37];
	aa[offset4 + 44] = -aa[offset4 + 36];

	aa[offset4 + 45] = dw_du(xi, m) * (8.0*M_PI*phi2or2*(dr2*dzodr*(a2*r2*wplOmega/alpha2 + r2*w*(dXu5/chi - 1.0) + (rr*w/chi)*(l*l + 2.0*l*ri*dRu5 + a2*(-l*l/h2 + r2*wplOmega2/alpha2) + r2*(chi*(chi - 2.0*dXu5)))) + (rr*w/chi)*(r2*(dzodr*dRu5*dRu5 + drodz*dZu5*dZu5))));

	// Columns.
	ja[offset4 +  0] = BASE +           IDX(i - 2, j    );
	ja[offset4 +  1] = BASE +           IDX(i - 1, j    );
	ja[offset4 +  2] = BASE +           IDX(i    , j - 2);
	ja[offset4 +  3] = BASE +           IDX(i    , j - 1);
	ja[offset4 +  4] = BASE +           IDX(i    , j    );
	ja[offset4 +  5] = BASE +           IDX(i    , j + 1);
	ja[offset4 +  6] = BASE +           IDX(i    , j + 2);
	ja[offset4 +  7] = BASE +           IDX(i + 1, j    );
	ja[offset4 +  8] = BASE +           IDX(i + 2, j    );

	ja[offset4 +  9] = BASE +     dim + IDX(i - 2, j    );
	ja[offset4 + 10] = BASE +     dim + IDX(i - 1, j    );
	ja[offset4 + 11] = BASE +     dim + IDX(i    , j - 2);
	ja[offset4 + 12] = BASE +     dim + IDX(i    , j - 1);
	ja[offset4 + 13] = BASE +     dim + IDX(i    , j    );
	ja[offset4 + 14] = BASE +     dim + IDX(i    , j + 1);
	ja[offset4 + 15] = BASE +     dim + IDX(i    , j + 2);
	ja[offset4 + 16] = BASE +     dim + IDX(i + 1, j    );
	ja[offset4 + 17] = BASE +     dim + IDX(i + 2, j    );

	ja[offset4 + 18] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset4 + 19] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset4 + 20] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset4 + 21] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset4 + 22] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset4 + 23] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset4 + 24] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset4 + 25] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset4 + 26] = BASE + 2 * dim + IDX(i + 2, j    );

	ja[offset4 + 27] = BASE + 3 * dim + IDX(i - 2, j    );
	ja[offset4 + 28] = BASE + 3 * dim + IDX(i - 1, j    );
	ja[offset4 + 29] = BASE + 3 * dim + IDX(i    , j - 2);
	ja[offset4 + 30] = BASE + 3 * dim + IDX(i    , j - 1);
	ja[offset4 + 31] = BASE + 3 * dim + IDX(i    , j    );
	ja[offset4 + 32] = BASE + 3 * dim + IDX(i    , j + 1);
	ja[offset4 + 33] = BASE + 3 * dim + IDX(i    , j + 2);
	ja[offset4 + 34] = BASE + 3 * dim + IDX(i + 1, j    );
	ja[offset4 + 35] = BASE + 3 * dim + IDX(i + 2, j    );

	ja[offset4 + 36] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset4 + 37] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset4 + 38] = BASE + 4 * dim + IDX(i    , j - 2);
	ja[offset4 + 39] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset4 + 40] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset4 + 41] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset4 + 42] = BASE + 4 * dim + IDX(i    , j + 2);
	ja[offset4 + 43] = BASE + 4 * dim + IDX(i + 1, j    );
	ja[offset4 + 44] = BASE + 4 * dim + IDX(i + 2, j    );

	ja[offset4 + 45] = BASE + 5 * dim;


	// Psi: grid number 4.
	ia[4 * dim + IDX(i, j)] = BASE + offset5;

	// Values.
	aa[offset5 +  0] = dzodr*((D10)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  1] = dzodr*((D11)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  2] = drodz*((D10)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  3] = drodz*((D11)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  4] = -2.0*dr2*dzodr*a2*wplOmega2/alpha2;
	aa[offset5 +  5] = -aa[offset5 +  3];
	aa[offset5 +  6] = -aa[offset5 +  2];
	aa[offset5 +  7] = -aa[offset5 +  1];
	aa[offset5 +  8] = -aa[offset5 +  0];

	aa[offset5 +  9] = 2.0*dr2*dzodr*a2*l*wplOmega/alpha2;

	aa[offset5 + 10] = dzodr*((D10)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 + 11] = dzodr*((D11)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 + 12] = drodz*((D10)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 13] = drodz*((D11)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 14] = 2.0*l*l*dzodr*(a2/h2)*(1.0/(ri*ri));
	aa[offset5 + 15] = -aa[offset5 + 13];
	aa[offset5 + 16] = -aa[offset5 + 12];
	aa[offset5 + 17] = -aa[offset5 + 11];
	aa[offset5 + 18] = -aa[offset5 + 10];

	aa[offset5 + 19] = 2.0*dzodr*(dr2*a2*(wplOmega2/alpha2 - m2) - l*l*(a2/h2)*(1.0/(ri*ri)));

	aa[offset5 + 20] = (D20)*dzodr + dzodr*((D10)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 21] = (D21)*dzodr + dzodr*((D11)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 22] = (D20)*drodz + drodz*((D10)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 23] = (D21)*drodz + drodz*((D11)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 24] = (D22)*(drodz + dzodr); // CONSTANT!
	aa[offset5 + 25] = 2.0*(D21)*drodz - aa[offset5 + 23];
	aa[offset5 + 26] = 2.0*(D20)*drodz - aa[offset5 + 22];
	aa[offset5 + 27] = 2.0*(D21)*dzodr - aa[offset5 + 21];
	aa[offset5 + 28] = 2.0*(D20)*dzodr - aa[offset5 + 20];

	aa[offset5 + 29] = dw_du(xi, m) * (dr2*dzodr*(2.0*a2*wplOmega/alpha2 + (-w/chi)*(2.0*chi - 2.0*(l + 1.0)/rr - 2.0*dXu5 - dXu1 - dXu3)));

	// Columns.
	ja[offset5 +  0] = BASE +           IDX(i - 2, j    );
	ja[offset5 +  1] = BASE +           IDX(i - 1, j    );
	ja[offset5 +  2] = BASE +           IDX(i    , j - 2);
	ja[offset5 +  3] = BASE +           IDX(i    , j - 1);
	ja[offset5 +  4] = BASE +           IDX(i    , j    );
	ja[offset5 +  5] = BASE +           IDX(i    , j + 1);
	ja[offset5 +  6] = BASE +           IDX(i    , j + 2);
	ja[offset5 +  7] = BASE +           IDX(i + 1, j    );
	ja[offset5 +  8] = BASE +           IDX(i + 2, j    );

	ja[offset5 + 19] = BASE +     dim + IDX(i    , j    );

	ja[offset5 + 10] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset5 + 11] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset5 + 12] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset5 + 13] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset5 + 14] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset5 + 15] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset5 + 16] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset5 + 17] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset5 + 18] = BASE + 2 * dim + IDX(i + 2, j    );

	ja[offset5 + 19] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset5 + 20] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset5 + 21] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset5 + 22] = BASE + 4 * dim + IDX(i    , j - 2);
	ja[offset5 + 23] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset5 + 24] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset5 + 25] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset5 + 26] = BASE + 4 * dim + IDX(i    , j + 2);
	ja[offset5 + 27] = BASE + 4 * dim + IDX(i + 1, j    );
	ja[offset5 + 28] = BASE + 4 * dim + IDX(i + 2, j    );

	ja[offset5 + 29] = BASE + 5 * dim;


	// All done.
	return;
}

// Jacobian for centered-semionesided 4th order stencil and variable omega.
void jacobian_4th_order_variable_omega_cs
(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings. 
	MKL_INT *ja,		// CSR array for columns.
	const MKL_INT NrTotal, 	// Grid total dimension in r.
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const double dr, 	// Spatial step in r.
	const double dz,	// Spatial step in z.
	const MKL_INT l, 	// Scalar field rotation number.
	const double m, 	// Scalar field mass.
	const double xi,	// Scalar field frequency variable.
	// Now come the grid variables. For cs stencil, each grid function has 10 variables.
	const double u104, const double u114, const double u120, const double u121, const double u122, const double u123, const double u124, const double u125, const double u134, const double u144,
	const double u204, const double u214, const double u220, const double u221, const double u222, const double u223, const double u224, const double u225, const double u234, const double u244,
	const double u304, const double u314, const double u320, const double u321, const double u322, const double u323, const double u324, const double u325, const double u334, const double u344,
	const double u404, const double u414, const double u420, const double u421, const double u422, const double u423, const double u424, const double u425, const double u434, const double u444,
	const double u504, const double u514, const double u520, const double u521, const double u522, const double u523, const double u524, const double u525, const double u534, const double u544,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5 	// Number of elements filled before filling function 5
)
{
	// Physical variables.
	double alpha = exp(u124);
	double Omega = u224;
	double h    = exp(u324);
	double a    = exp(u424);
	double psi  = u524;
	double w = omega_calc(xi, m);

	// Coordinates.
	double ri = (double)i - 1.5;
	double zi = (double)j - 1.5;
	double r = ri * dr;
	double z = zi * dz;
	double r2 = r * r;
	double z2 = z * z;
	double rr = sqrt(r2 + z2);

	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	double dr2 = dr * dr;
	double dz2 = dz * dz;

	// Scalar field mass and frequency.
	double m2 = m * m;
	double w2 = w * w;
	double chi = sqrt(m2 - w2);

	// Scalar field.
	double rlm1 = (l == 1) ? 1.0 : pow(r, l - 1);
	double rl = rlm1 * r;
	double phior = rlm1 * exp(-chi * rr) * exp(psi);
	double phi = r * phior;
	double phi2or2 = phior * phior;
	double phi2 = phi * phi;

	// Shift combined with scalar field rotation.
	double wplOmega = w + l * Omega;
	double wplOmega2 = wplOmega * wplOmega;

	// Finite differences.
	double dRu1 = D10 * u104 + D11 * u114 + D13 * u134 + D14 * u144;
	double dRu2 = D10 * u204 + D11 * u214 + D13 * u234 + D14 * u244;
	double dRu3 = D10 * u304 + D11 * u314 + D13 * u334 + D14 * u344;
	double dRu4 = D10 * u404 + D11 * u414 + D13 * u434 + D14 * u444;
	double dRu5 = D10 * u504 + D11 * u514 + D13 * u534 + D14 * u544;

	double dZu1 = S11 * u121 + S12 * u122 + S13 * u123 + S14 * u124 + S14 * u125;
	double dZu2 = S11 * u221 + S12 * u222 + S13 * u223 + S14 * u224 + S14 * u225;
	double dZu3 = S11 * u321 + S12 * u322 + S13 * u323 + S14 * u324 + S14 * u325;
	double dZu4 = S11 * u421 + S12 * u422 + S13 * u423 + S14 * u424 + S14 * u425;
	double dZu5 = S11 * u521 + S12 * u522 + S13 * u523 + S14 * u524 + S14 * u525;

	// Radial derivatives.
	double dXu5 = (ri * dRu5 + zi * dZu5) / rr;
	double dXu1 = (ri * dRu1 + zi * dZu1) / rr;
	double dXu3 = (ri * dRu3 + zi * dZu3) / rr;

	// Squared variables.
	double alpha2 = alpha * alpha;
	double h2 = h * h;
	double a2 = a * a;

	// Common term.
	double r2h2oalpha2dOmega2 = r2 * h2 * (dzodr * dRu2 * dRu2 + drodz * dZu2 * dZu2) / alpha2;

	// Alpha: grid number 0.
	ia[IDX(i, j)] = BASE + offset1;

	// Values.
	aa[offset1 +  0] = dzodr*((D20) + (D10)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  1] = dzodr*((D21) + (D11)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  2] = drodz*((S20)); // CONSTANT!
	aa[offset1 +  3] = drodz*((S21) + (S11)*(2.0*dZu1 + dZu3));
	aa[offset1 +  4] = drodz*((S22) + (S12)*(2.0*dZu1 + dZu3));
	aa[offset1 +  5] = drodz*((S23) + (S13)*(2.0*dZu1 + dZu3));
	aa[offset1 +  6] = dzodr*(D22) + drodz*((S24) + (S14)*(2.0*dZu1 + dZu3)) + r2h2oalpha2dOmega2 + 16.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset1 +  7] = drodz*((S25) + (S15)*(2.0*dZu1 + dZu3));
	aa[offset1 +  8] = dzodr*((D23) + (D13)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  9] = dzodr*((D24) + (D14)*(1.0/ri + 2.0*dRu1 + dRu3));

	aa[offset1 + 10] = dzodr*((D10)*(-r2*h2*dRu2/alpha2));
	aa[offset1 + 11] = dzodr*((D11)*(-r2*h2*dRu2/alpha2));
	aa[offset1 + 12] = drodz*((S11)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 13] = drodz*((S12)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 14] = drodz*((S13)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 15] = drodz*((S14)*(-r2*h2*dZu2/alpha2)) - 16.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset1 + 16] = drodz*((S15)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 17] = dzodr*((D13)*(-r2*h2*dRu2/alpha2));
	aa[offset1 + 18] = dzodr*((D14)*(-r2*h2*dRu2/alpha2));

	aa[offset1 + 19] = dzodr*((D10)*dRu1);
	aa[offset1 + 20] = dzodr*((D11)*dRu1);
	aa[offset1 + 21] = drodz*((S11)*dZu1);
	aa[offset1 + 22] = drodz*((S12)*dZu1);
	aa[offset1 + 23] = drodz*((S13)*dZu1);
	aa[offset1 + 24] = drodz*((S14)*dZu1) - r2h2oalpha2dOmega2;
	aa[offset1 + 25] = drodz*((S15)*dZu1);
	aa[offset1 + 26] = dzodr*((D13)*dRu1);
	aa[offset1 + 27] = dzodr*((D14)*dRu1);

	aa[offset1 + 28] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;

	aa[offset1 + 29] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;

	aa[offset1 + 30] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*wplOmega*phi2/alpha2 + 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2*(rr*w/chi));

	// Columns.
	ja[offset1 +  0] = BASE +           IDX(i - 2, j    );
	ja[offset1 +  1] = BASE +           IDX(i - 1, j    );
	ja[offset1 +  2] = BASE +           IDX(i    , j - 4);
	ja[offset1 +  3] = BASE +           IDX(i    , j - 3);
	ja[offset1 +  4] = BASE +           IDX(i    , j - 2);
	ja[offset1 +  5] = BASE +           IDX(i    , j - 1);
	ja[offset1 +  6] = BASE +           IDX(i    , j    );
	ja[offset1 +  7] = BASE +           IDX(i    , j + 1);
	ja[offset1 +  8] = BASE +           IDX(i + 1, j    );
	ja[offset1 +  9] = BASE +           IDX(i + 2, j    );

	ja[offset1 + 10] = BASE +     dim + IDX(i - 2, j    );
	ja[offset1 + 11] = BASE +     dim + IDX(i - 1, j    );
	ja[offset1 + 12] = BASE +     dim + IDX(i    , j - 3);
	ja[offset1 + 13] = BASE +     dim + IDX(i    , j - 2);
	ja[offset1 + 14] = BASE +     dim + IDX(i    , j - 1);
	ja[offset1 + 15] = BASE +     dim + IDX(i    , j    );
	ja[offset1 + 16] = BASE +     dim + IDX(i    , j + 1);
	ja[offset1 + 17] = BASE +     dim + IDX(i + 1, j    );
	ja[offset1 + 18] = BASE +     dim + IDX(i + 2, j    );

	ja[offset1 + 19] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset1 + 20] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset1 + 21] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset1 + 22] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset1 + 23] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset1 + 24] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset1 + 25] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset1 + 26] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset1 + 27] = BASE + 2 * dim + IDX(i + 2, j    );

	ja[offset1 + 28] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset1 + 29] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset1 + 30] = BASE + 5 * dim;


	// Beta: grid number 1.
	ia[dim + IDX(i, j)] = BASE + offset2;

	// Values.
	aa[offset2 +  0] = dzodr*((D10)*(-dRu2));
	aa[offset2 +  1] = dzodr*((D11)*(-dRu2));
	aa[offset2 +  2] = drodz*((S11)*(-dZu2));
	aa[offset2 +  3] = drodz*((S12)*(-dZu2));
	aa[offset2 +  4] = drodz*((S13)*(-dZu2));
	aa[offset2 +  5] = drodz*((S14)*(-dZu2));
	aa[offset2 +  6] = drodz*((S15)*(-dZu2));
	aa[offset2 +  7] = dzodr*((D13)*(-dRu2));
	aa[offset2 +  8] = dzodr*((D14)*(-dRu2));

	aa[offset2 +  9] = dzodr*((D20) + (D10)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 + 10] = dzodr*((D21) + (D11)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 + 11] = drodz*(S20); // CONSTANT!
	aa[offset2 + 12] = drodz*((S21) + (S11)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 13] = drodz*((S22) + (S12)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 14] = drodz*((S23) + (S13)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 15] = dzodr*(D22) + drodz*((S24) + (S14)*(-dZu1 + 3.0*dZu3)) - 16.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset2 + 16] = drodz*((S25) + (S15)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 17] = dzodr*((D23) + (D13)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 + 18] = dzodr*((D24) + (D14)*(3.0/ri - dRu1 + 3.0*dRu3));
	
	aa[offset2 + 19] = dzodr*((D10)*(3.0*dRu2));
	aa[offset2 + 20] = dzodr*((D11)*(3.0*dRu2));
	aa[offset2 + 21] = drodz*((S11)*(3.0*dZu2));
	aa[offset2 + 22] = drodz*((S12)*(3.0*dZu2));
	aa[offset2 + 23] = drodz*((S13)*(3.0*dZu2));
	aa[offset2 + 24] = drodz*((S14)*(3.0*dZu2)) + 32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;
	aa[offset2 + 25] = drodz*((S15)*(3.0*dZu2));
	aa[offset2 + 26] = dzodr*((D13)*(3.0*dRu2));
	aa[offset2 + 27] = dzodr*((D14)*(3.0*dRu2));

	aa[offset2 + 28] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;
	
	aa[offset2 + 29] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;

	aa[offset2 + 30] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*l*phi2or2/h2 - 32.0*M_PI*dr2*dzodr*(a2/h2)*l*wplOmega*phi2or2*(rr*w/chi));

	// Columns.
	ja[offset2 +  0] = BASE +           IDX(i - 2, j    );
	ja[offset2 +  1] = BASE +           IDX(i - 1, j    );
	ja[offset2 +  2] = BASE +           IDX(i    , j - 3);
	ja[offset2 +  3] = BASE +           IDX(i    , j - 2);
	ja[offset2 +  4] = BASE +           IDX(i    , j - 1);
	ja[offset2 +  5] = BASE +           IDX(i    , j    );
	ja[offset2 +  6] = BASE +           IDX(i    , j + 1);
	ja[offset2 +  7] = BASE +           IDX(i + 1, j    );
	ja[offset2 +  8] = BASE +           IDX(i + 2, j    );

	ja[offset2 +  9] = BASE +     dim + IDX(i - 2, j    );
	ja[offset2 + 10] = BASE +     dim + IDX(i - 1, j    );
	ja[offset2 + 11] = BASE +     dim + IDX(i    , j - 4);
	ja[offset2 + 12] = BASE +     dim + IDX(i    , j - 3);
	ja[offset2 + 13] = BASE +     dim + IDX(i    , j - 2);
	ja[offset2 + 14] = BASE +     dim + IDX(i    , j - 1);
	ja[offset2 + 15] = BASE +     dim + IDX(i    , j    );
	ja[offset2 + 16] = BASE +     dim + IDX(i    , j + 1);
	ja[offset2 + 17] = BASE +     dim + IDX(i + 1, j    );
	ja[offset2 + 18] = BASE +     dim + IDX(i + 2, j    );

	ja[offset2 + 19] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset2 + 20] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset2 + 21] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset2 + 22] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset2 + 23] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset2 + 24] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset2 + 25] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset2 + 26] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset2 + 27] = BASE + 2 * dim + IDX(i + 2, j    );

	ja[offset2 + 28] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset2 + 29] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset2 + 30] = BASE + 5 * dim;


	// H: grid number 2.
	ia[2 * dim + IDX(i, j)] = BASE + offset3;

	// Values.
	aa[offset3 +  0] = dzodr*((D10)*(1.0/ri + dRu3));
	aa[offset3 +  1] = dzodr*((D11)*(1.0/ri + dRu3));
	aa[offset3 +  2] = drodz*((S11)*dZu3);
	aa[offset3 +  3] = drodz*((S12)*dZu3);
	aa[offset3 +  4] = drodz*((S13)*dZu3);
	aa[offset3 +  5] = drodz*((S14)*dZu3)-r2h2oalpha2dOmega2;
	aa[offset3 +  6] = drodz*((S15)*dZu3);
	aa[offset3 +  7] = dzodr*((D13)*(1.0/ri + dRu3));
	aa[offset3 +  8] = dzodr*((D14)*(1.0/ri + dRu3));

	aa[offset3 +  9] = dzodr*((D10)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 10] = dzodr*((D11)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 11] = drodz*((S11)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 12] = drodz*((S12)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 13] = drodz*((S13)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 14] = drodz*((S14)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 15] = drodz*((S15)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 16] = dzodr*((D13)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 17] = dzodr*((D14)*(r2*h2*dRu2/alpha2));

	aa[offset3 + 18] = dzodr*((D20) + (D10)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 19] = dzodr*((D21) + (D11)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 20] = drodz*(S20); // CONSTANT!
	aa[offset3 + 21] = drodz*((S21) + (S11)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 22] = drodz*((S22) + (S12)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 23] = drodz*((S23) + (S13)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 24] = dzodr*(D22) + drodz*((S24) + (S14)*(dZu1 + 2.0*dZu3)) + r2h2oalpha2dOmega2 - 16.0*M_PI*dr2*dzodr*a2*l*l*phi2or2/h2;
	aa[offset3 + 25] = drodz*((S25) + (S15)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 26] = dzodr*((D23) + (D13)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 27] = dzodr*((D24) + (D14)*(2.0/ri + dRu1 + 2.0*dRu3));

	aa[offset3 + 28] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;

	aa[offset3 + 29] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;

	aa[offset3 + 30] = dw_du(xi, m) * (8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2*(rr*w/chi));

	// Columns.
	ja[offset3 +  0] = BASE +           IDX(i - 2, j    );
	ja[offset3 +  1] = BASE +           IDX(i - 1, j    );
	ja[offset3 +  2] = BASE +           IDX(i    , j - 3);
	ja[offset3 +  3] = BASE +           IDX(i    , j - 2);
	ja[offset3 +  4] = BASE +           IDX(i    , j - 1);
	ja[offset3 +  5] = BASE +           IDX(i    , j    );
	ja[offset3 +  6] = BASE +           IDX(i    , j + 1);
	ja[offset3 +  7] = BASE +           IDX(i + 1, j    );
	ja[offset3 +  8] = BASE +           IDX(i + 2, j    );

	ja[offset3 +  9] = BASE +     dim + IDX(i - 2, j    );
	ja[offset3 + 10] = BASE +     dim + IDX(i - 1, j    );
	ja[offset3 + 11] = BASE +     dim + IDX(i    , j - 3);
	ja[offset3 + 12] = BASE +     dim + IDX(i    , j - 2);
	ja[offset3 + 13] = BASE +     dim + IDX(i    , j - 1);
	ja[offset3 + 14] = BASE +     dim + IDX(i    , j    );
	ja[offset3 + 15] = BASE +     dim + IDX(i    , j + 1);
	ja[offset3 + 16] = BASE +     dim + IDX(i + 1, j    );
	ja[offset3 + 17] = BASE +     dim + IDX(i + 2, j    );

	ja[offset3 + 18] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset3 + 19] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset3 + 20] = BASE + 2 * dim + IDX(i    , j - 4);
	ja[offset3 + 21] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset3 + 22] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset3 + 23] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset3 + 24] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset3 + 25] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset3 + 26] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset3 + 27] = BASE + 2 * dim + IDX(i + 2, j    );

	ja[offset3 + 28] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset3 + 29] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset3 + 30] = BASE + 5 * dim;


	// A: grid number 3.
	ia[3 * dim + IDX(i, j)] = BASE + offset4;

	// Values.
	aa[offset4 +  0] = dzodr*((D10)*(-1.0/ri - dRu3));
	aa[offset4 +  1] = dzodr*((D11)*(-1.0/ri - dRu3));
	aa[offset4 +  2] = drodz*((S11)*(-dZu3));
	aa[offset4 +  3] = drodz*((S12)*(-dZu3));
	aa[offset4 +  4] = drodz*((S13)*(-dZu3));
	aa[offset4 +  5] = drodz*((S14)*(-dZu3)) + 0.5*r2h2oalpha2dOmega2 - 8.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset4 +  6] = drodz*((S15)*(-dZu3));
	aa[offset4 +  7] = dzodr*((D13)*(-1.0/ri - dRu3));
	aa[offset4 +  8] = dzodr*((D14)*(-1.0/ri - dRu3));

	aa[offset4 +  9] = dzodr*((D10)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 + 10] = dzodr*((D11)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 + 11] = drodz*((S11)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 12] = drodz*((S12)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 13] = drodz*((S13)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 14] = drodz*((S14)*(-0.5*r2*h2*dZu2/alpha2)) + 8.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset4 + 15] = drodz*((S15)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 16] = dzodr*((D13)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 + 17] = dzodr*((D14)*(-0.5*r2*h2*dRu2/alpha2));

	aa[offset4 + 18] = dzodr*((D10)*(-dRu1));
	aa[offset4 + 19] = dzodr*((D11)*(-dRu1));
	aa[offset4 + 20] = drodz*((S11)*(-dZu1));
	aa[offset4 + 21] = drodz*((S12)*(-dZu1));
	aa[offset4 + 22] = drodz*((S13)*(-dZu1));
	aa[offset4 + 23] = drodz*((S14)*(-dZu1)) - 0.5*r2h2oalpha2dOmega2 + 8.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset4 + 24] = drodz*((S15)*(-dZu1));
	aa[offset4 + 25] = dzodr*((D13)*(-dRu1));
	aa[offset4 + 26] = dzodr*((D14)*(-dRu1));

	aa[offset4 + 27] = (D20)*dzodr; // CONSTANT!
	aa[offset4 + 28] = (D21)*dzodr; // CONSTANT!
	aa[offset4 + 29] = (S20)*drodz; // CONSTANT!
	aa[offset4 + 30] = (S21)*drodz; // CONSTANT!
	aa[offset4 + 31] = (S22)*drodz; // CONSTANT!
	aa[offset4 + 32] = (S23)*drodz; // CONSTANT!
	aa[offset4 + 33] = (S24)*drodz + (D22)*dzodr + 8.0*M_PI*dr2*dzodr*(-l*l/h2 + r2*wplOmega2/alpha2)*a2*phi2or2;
	aa[offset4 + 34] = (S25)*drodz; // CONSTANT!
	aa[offset4 + 35] = (D21)*dzodr; // CONSTANT!
	aa[offset4 + 36] = (D20)*dzodr; // CONSTANT!
	
	aa[offset4 + 37] = dzodr*((D10)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 38] = dzodr*((D11)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 39] = drodz*((S11)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 40] = drodz*((S12)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 41] = drodz*((S13)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 42] = drodz*((S14)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr))) + 8.0*M_PI*phi2or2*(dr2*dzodr*(l*l + 2.0*l*ri*dRu5 + a2*(-l*l/h2 + r2*wplOmega2/alpha2) + r2*(chi*(chi - 2.0*dXu5))) + (r2*(dzodr*dRu5*dRu5 + drodz*dZu5*dZu5)));
	aa[offset4 + 43] = drodz*((S15)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 44] = dzodr*((D13)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 45] = dzodr*((D14)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));

	aa[offset4 + 46] = dw_du(xi, m) * (8.0*M_PI*phi2or2*(dr2*dzodr*(a2*r2*wplOmega/alpha2 + r2*w*(dXu5/chi - 1.0) + (rr*w/chi)*(l*l + 2.0*l*ri*dRu5 + a2*(-l*l/h2 + r2*wplOmega2/alpha2) + r2*(chi*(chi - 2.0*dXu5)))) + (rr*w/chi)*(r2*(dzodr*dRu5*dRu5 + drodz*dZu5*dZu5))));

	// Columns.
	ja[offset4 +  0] = BASE +           IDX(i - 2, j    );
	ja[offset4 +  1] = BASE +           IDX(i - 1, j    );
	ja[offset4 +  2] = BASE +           IDX(i    , j - 3);
	ja[offset4 +  3] = BASE +           IDX(i    , j - 2);
	ja[offset4 +  4] = BASE +           IDX(i    , j - 1);
	ja[offset4 +  5] = BASE +           IDX(i    , j    );
	ja[offset4 +  6] = BASE +           IDX(i    , j + 1);
	ja[offset4 +  7] = BASE +           IDX(i + 1, j    );
	ja[offset4 +  8] = BASE +           IDX(i + 2, j    );

	ja[offset4 +  9] = BASE +     dim + IDX(i - 2, j    );
	ja[offset4 + 10] = BASE +     dim + IDX(i - 1, j    );
	ja[offset4 + 11] = BASE +     dim + IDX(i    , j - 3);
	ja[offset4 + 12] = BASE +     dim + IDX(i    , j - 2);
	ja[offset4 + 13] = BASE +     dim + IDX(i    , j - 1);
	ja[offset4 + 14] = BASE +     dim + IDX(i    , j    );
	ja[offset4 + 15] = BASE +     dim + IDX(i    , j + 1);
	ja[offset4 + 16] = BASE +     dim + IDX(i + 1, j    );
	ja[offset4 + 17] = BASE +     dim + IDX(i + 2, j    );

	ja[offset4 + 18] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset4 + 19] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset4 + 20] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset4 + 21] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset4 + 22] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset4 + 23] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset4 + 24] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset4 + 25] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset4 + 26] = BASE + 2 * dim + IDX(i + 2, j    );

	ja[offset4 + 27] = BASE + 3 * dim + IDX(i - 2, j    );
	ja[offset4 + 28] = BASE + 3 * dim + IDX(i - 1, j    );
	ja[offset4 + 29] = BASE + 3 * dim + IDX(i    , j - 4);
	ja[offset4 + 30] = BASE + 3 * dim + IDX(i    , j - 3);
	ja[offset4 + 31] = BASE + 3 * dim + IDX(i    , j - 2);
	ja[offset4 + 32] = BASE + 3 * dim + IDX(i    , j - 1);
	ja[offset4 + 33] = BASE + 3 * dim + IDX(i    , j    );
	ja[offset4 + 34] = BASE + 3 * dim + IDX(i    , j + 1);
	ja[offset4 + 35] = BASE + 3 * dim + IDX(i + 1, j    );
	ja[offset4 + 36] = BASE + 3 * dim + IDX(i + 2, j    );

	ja[offset4 + 37] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset4 + 38] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset4 + 39] = BASE + 4 * dim + IDX(i    , j - 3);
	ja[offset4 + 40] = BASE + 4 * dim + IDX(i    , j - 2);
	ja[offset4 + 41] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset4 + 42] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset4 + 43] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset4 + 44] = BASE + 4 * dim + IDX(i + 1, j    );
	ja[offset4 + 45] = BASE + 4 * dim + IDX(i + 2, j    );

	ja[offset4 + 46] = BASE + 5 * dim;

	// Psi: grid number 4.
	ia[4 * dim + IDX(i, j)] = BASE + offset5;

	// Values.
	aa[offset5 +  0] = dzodr*((D10)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  1] = dzodr*((D11)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  2] = drodz*((S11)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  3] = drodz*((S12)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  4] = drodz*((S13)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  5] = drodz*((S14)*(dZu5 - chi*dz2*zi/rr))-2.0*dr2*dzodr*a2*wplOmega2/alpha2;
	aa[offset5 +  6] = drodz*((S15)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  7] = dzodr*((D13)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  8] = dzodr*((D14)*(dRu5 + l/ri - chi*dr2*ri/rr));

	aa[offset5 +  9] = 2.0*dr2*dzodr*a2*l*wplOmega/alpha2;

	aa[offset5 + 10] = dzodr*((D10)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 + 11] = dzodr*((D11)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 + 12] = drodz*((S11)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 13] = drodz*((S12)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 14] = drodz*((S13)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 15] = drodz*((S14)*(dZu5 - chi*dz2*zi/rr))+2.0*l*l*dzodr*(a2/h2)*(1.0/(ri*ri));
	aa[offset5 + 16] = drodz*((S15)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 17] = dzodr*((D13)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 + 18] = dzodr*((D14)*(dRu5 + l/ri - chi*dr2*ri/rr));

	aa[offset5 + 19] = 2.0*dzodr*(dr2*a2*(wplOmega2/alpha2 - m2) - l*l*(a2/h2)*(1.0/(ri*ri)));

	aa[offset5 + 20] = (D20)*dzodr + dzodr*((D10)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 21] = (D21)*dzodr + dzodr*((D11)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 22] = (S20)*drodz; // CONSTANT!
	aa[offset5 + 23] = (S21)*drodz + drodz*((S11)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 24] = (S22)*drodz + drodz*((S12)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 25] = (S23)*drodz + drodz*((S13)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 26] = (D22)*dzodr + (S24)*drodz + drodz*((S14)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 27] = (S25)*drodz + drodz*((S15)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 28] = (D23)*dzodr + dzodr*((D13)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 29] = (D24)*dzodr + dzodr*((D14)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));

	aa[offset5 + 30] = dw_du(xi, m) * (dr2*dzodr*(2.0*a2*wplOmega/alpha2 + (-w/chi)*(2.0*chi - 2.0*(l + 1.0)/rr - 2.0*dXu5 - dXu1 - dXu3)));

	// Columns.
	ja[offset5 +  0] = BASE +           IDX(i - 2, j    );
	ja[offset5 +  1] = BASE +           IDX(i - 1, j    );
	ja[offset5 +  2] = BASE +           IDX(i    , j - 3);
	ja[offset5 +  3] = BASE +           IDX(i    , j - 2);
	ja[offset5 +  4] = BASE +           IDX(i    , j - 1);
	ja[offset5 +  5] = BASE +           IDX(i    , j    );
	ja[offset5 +  6] = BASE +           IDX(i    , j + 1);
	ja[offset5 +  7] = BASE +           IDX(i + 1, j    );
	ja[offset5 +  8] = BASE +           IDX(i + 2, j    );

	ja[offset5 +  9] = BASE +     dim + IDX(i    , j    );

	ja[offset5 + 10] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset5 + 11] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset5 + 12] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset5 + 13] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset5 + 14] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset5 + 15] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset5 + 16] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset5 + 17] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset5 + 18] = BASE + 2 * dim + IDX(i + 2, j    );

	ja[offset5 + 19] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset5 + 20] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset5 + 21] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset5 + 22] = BASE + 4 * dim + IDX(i    , j - 4);
	ja[offset5 + 23] = BASE + 4 * dim + IDX(i    , j - 3);
	ja[offset5 + 24] = BASE + 4 * dim + IDX(i    , j - 2);
	ja[offset5 + 25] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset5 + 26] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset5 + 27] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset5 + 28] = BASE + 4 * dim + IDX(i + 1, j    );
	ja[offset5 + 29] = BASE + 4 * dim + IDX(i + 2, j    );

	ja[offset5 + 30] = BASE + 5 * dim;


	// All done.
	return;
}

// Jacobian for semionesided-centered 4th order stencil and variable omega.
void jacobian_4th_order_variable_omega_sc
(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings. 
	MKL_INT *ja,		// CSR array for columns.
	const MKL_INT NrTotal, 	// Grid total dimension in r.
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const double dr, 	// Spatial step in r.
	const double dz,	// Spatial step in z.
	const MKL_INT l, 	// Scalar field rotation number.
	const double m, 	// Scalar field mass.
	const double xi,	// Scalar field frequency variable.
	// Now come the grid variables. For sc stencil, each grid function has 10 variables.
	const double u102, const double u112, const double u122, const double u132, const double u140, const double u141, const double u142, const double u143, const double u144, const double u152,
	const double u202, const double u212, const double u222, const double u232, const double u240, const double u241, const double u242, const double u243, const double u244, const double u252,
	const double u302, const double u312, const double u322, const double u332, const double u340, const double u341, const double u342, const double u343, const double u344, const double u352,
	const double u402, const double u412, const double u422, const double u432, const double u440, const double u441, const double u442, const double u443, const double u444, const double u452,
	const double u502, const double u512, const double u522, const double u532, const double u540, const double u541, const double u542, const double u543, const double u544, const double u552,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5 	// Number of elements filled before filling function 5
)
{
	// Physical variables.
	double alpha = exp(u142);
	double Omega = u242;
	double h    = exp(u342);
	double a    = exp(u442);
	double psi  = u542;
	double w = omega_calc(xi, m);

	// Coordinates.
	double ri = (double)i - 1.5;
	double zi = (double)j - 1.5;
	double r = ri * dr;
	double z = zi * dz;
	double r2 = r * r;
	double z2 = z * z;
	double rr = sqrt(r2 + z2);

	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	double dr2 = dr * dr;
	double dz2 = dz * dz;

	// Scalar field mass and frequency.
	double m2 = m * m;
	double w2 = w * w;
	double chi = sqrt(m2 - w2);

	// Scalar field.
	double rlm1 = (l == 1) ? 1.0 : pow(r, l - 1);
	double rl = rlm1 * r;
	double phior = rlm1 * exp(-chi * rr) * exp(psi);
	double phi = r * phior;
	double phi2or2 = phior * phior;
	double phi2 = phi * phi;

	// Shift combined with scalar field rotation.
	double wplOmega = w + l * Omega;
	double wplOmega2 = wplOmega * wplOmega;

	// Finite differences.
	double dRu1 = S11 * u112 + S12 * u122 + S13 * u132 + S14 * u142 + S14 * u152;
	double dRu2 = S11 * u212 + S12 * u222 + S13 * u232 + S14 * u242 + S14 * u252;
	double dRu3 = S11 * u312 + S12 * u322 + S13 * u332 + S14 * u342 + S14 * u352;
	double dRu4 = S11 * u412 + S12 * u422 + S13 * u432 + S14 * u442 + S14 * u452;
	double dRu5 = S11 * u512 + S12 * u522 + S13 * u532 + S14 * u542 + S14 * u552;

	double dZu1 = D10 * u141 + D11 * u142 + D13 * u143 + D14 * u144;
	double dZu2 = D10 * u241 + D11 * u242 + D13 * u243 + D14 * u244;
	double dZu3 = D10 * u341 + D11 * u342 + D13 * u343 + D14 * u344;
	double dZu4 = D10 * u441 + D11 * u442 + D13 * u443 + D14 * u444;
	double dZu5 = D10 * u541 + D11 * u542 + D13 * u543 + D14 * u544;

	// Radial derivatives.
	double dXu5 = (ri * dRu5 + zi * dZu5) / rr;
	double dXu1 = (ri * dRu1 + zi * dZu1) / rr;
	double dXu3 = (ri * dRu3 + zi * dZu3) / rr;

	// Squared variables.
	double alpha2 = alpha * alpha;
	double h2 = h * h;
	double a2 = a * a;

	// Common term.
	double r2h2oalpha2dOmega2 = r2 * h2 * (dzodr * dRu2 * dRu2 + drodz * dZu2 * dZu2) / alpha2;

	// Alpha: grid number 0.
	ia[IDX(i, j)] = BASE + offset1;

	// Values.
	aa[offset1 +  0] = dzodr*(S20); // CONSTANT!
	aa[offset1 +  1] = dzodr*((S21) + (S11)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  2] = dzodr*((S22) + (S12)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  3] = dzodr*((S23) + (S13)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  4] = drodz*((D20) + (D10)*(2.0*dZu1 + dZu3));
	aa[offset1 +  5] = drodz*((D21) + (D11)*(2.0*dZu1 + dZu3));
	aa[offset1 +  6] = drodz*(D22) + dzodr*((S24) + (S14)*(1.0/ri + 2.0*dRu1 + dRu3)) + r2h2oalpha2dOmega2 + 16.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset1 +  7] = drodz*((D23) + (D13)*(2.0*dZu1 + dZu3));
	aa[offset1 +  8] = drodz*((D24) + (D14)*(2.0*dZu1 + dZu3));
	aa[offset1 +  9] = dzodr*((S25) + (S15)*(1.0/ri + 2.0*dRu1 + dRu3));

	aa[offset1 + 10] = dzodr*((S11)*(-r2*h2*dRu2/alpha2));
	aa[offset1 + 11] = dzodr*((S12)*(-r2*h2*dRu2/alpha2));
	aa[offset1 + 12] = dzodr*((S13)*(-r2*h2*dRu2/alpha2));
	aa[offset1 + 13] = drodz*((D10)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 14] = drodz*((D11)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 15] = dzodr*((S14)*(-r2*h2*dRu2/alpha2)) - 16.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset1 + 16] = drodz*((D13)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 17] = drodz*((D14)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 18] = dzodr*((S15)*(-r2*h2*dRu2/alpha2));

	aa[offset1 + 19] = dzodr*((S11)*dRu1);
	aa[offset1 + 20] = dzodr*((S12)*dRu1);
	aa[offset1 + 21] = dzodr*((S13)*dRu1);
	aa[offset1 + 22] = drodz*((D10)*dZu1);
	aa[offset1 + 23] = drodz*((D11)*dZu1);
	aa[offset1 + 24] = dzodr*((S14)*dRu1) - r2h2oalpha2dOmega2;
	aa[offset1 + 25] = drodz*((D13)*dZu1);
	aa[offset1 + 26] = drodz*((D14)*dZu1);
	aa[offset1 + 27] = dzodr*((S15)*dRu1);

	aa[offset1 + 28] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;

	aa[offset1 + 29] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;

	aa[offset1 + 30] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*wplOmega*phi2/alpha2 + 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2*(rr*w/chi));

	// Columns.
	ja[offset1 +  0] = BASE +           IDX(i - 4, j    );
	ja[offset1 +  1] = BASE +           IDX(i - 3, j    );
	ja[offset1 +  2] = BASE +           IDX(i - 2, j    );
	ja[offset1 +  3] = BASE +           IDX(i - 1, j    );
	ja[offset1 +  4] = BASE +           IDX(i    , j - 2);
	ja[offset1 +  5] = BASE +           IDX(i    , j - 1);
	ja[offset1 +  6] = BASE +           IDX(i    , j    );
	ja[offset1 +  7] = BASE +           IDX(i    , j + 1);
	ja[offset1 +  8] = BASE +           IDX(i    , j + 2);
	ja[offset1 +  9] = BASE +           IDX(i + 1, j    );

	ja[offset1 + 10] = BASE +     dim + IDX(i - 3, j    );
	ja[offset1 + 11] = BASE +     dim + IDX(i - 2, j    );
	ja[offset1 + 12] = BASE +     dim + IDX(i - 1, j    );
	ja[offset1 + 13] = BASE +     dim + IDX(i    , j - 2);
	ja[offset1 + 14] = BASE +     dim + IDX(i    , j - 1);
	ja[offset1 + 15] = BASE +     dim + IDX(i    , j    );
	ja[offset1 + 16] = BASE +     dim + IDX(i    , j + 1);
	ja[offset1 + 17] = BASE +     dim + IDX(i    , j + 2);
	ja[offset1 + 18] = BASE +     dim + IDX(i + 1, j    );

	ja[offset1 + 19] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset1 + 20] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset1 + 21] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset1 + 22] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset1 + 23] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset1 + 24] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset1 + 25] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset1 + 26] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset1 + 27] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset1 + 28] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset1 + 29] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset1 + 30] = BASE + 5 * dim;


	// Beta: grid number 1.
	ia[dim + IDX(i, j)] = BASE + offset2;

	// Values.
	aa[offset2 +  0] = dzodr*((S11)*(-dRu2));
	aa[offset2 +  1] = dzodr*((S12)*(-dRu2));
	aa[offset2 +  2] = dzodr*((S13)*(-dRu2));
	aa[offset2 +  3] = drodz*((S11)*(-dZu2));
	aa[offset2 +  4] = drodz*((S12)*(-dZu2));
	aa[offset2 +  5] = drodz*((S13)*(-dZu2));
	aa[offset2 +  6] = dzodr*((S14)*(-dRu2)) + drodz*((S14)*(-dZu2));
	aa[offset2 +  7] = drodz*((S15)*(-dZu2));
	aa[offset2 +  8] = dzodr*((S15)*(-dRu2));

	aa[offset2 +  9] = dzodr*(S20); // CONSTANT!
	aa[offset2 + 10] = dzodr*((S21) + (S11)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 + 11] = dzodr*((S22) + (S12)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 + 12] = dzodr*((S23) + (S13)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 + 13] = drodz*(S20); // CONSTANT!
	aa[offset2 + 14] = drodz*((S21) + (S11)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 15] = drodz*((S22) + (S12)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 16] = drodz*((S23) + (S13)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 17] = dzodr*((S24) + (S14)*(3.0/ri - dRu1 + 3.0*dRu3)) + drodz*((S24) + (S14)*(-dZu1 + 3.0*dZu3)) - 16.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset2 + 18] = drodz*((S25) + (S15)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 19] = dzodr*((S25) + (S15)*(3.0/ri - dRu1 + 3.0*dRu3));
	
	aa[offset2 + 20] = dzodr*((S11)*(3.0*dRu2));
	aa[offset2 + 21] = dzodr*((S12)*(3.0*dRu2));
	aa[offset2 + 22] = dzodr*((S13)*(3.0*dRu2));
	aa[offset2 + 23] = drodz*((S11)*(3.0*dZu2));
	aa[offset2 + 24] = drodz*((S12)*(3.0*dZu2));
	aa[offset2 + 25] = drodz*((S13)*(3.0*dZu2));
	aa[offset2 + 26] = dzodr*((S14)*(3.0*dRu2)) + drodz*((S14)*(3.0*dZu2)) + 32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;
	aa[offset2 + 27] = drodz*((S15)*(3.0*dZu2));
	aa[offset2 + 28] = dzodr*((S15)*(3.0*dRu2));

	aa[offset2 + 29] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;
	
	aa[offset2 + 30] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;

	aa[offset2 + 31] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*l*phi2or2/h2 - 32.0*M_PI*dr2*dzodr*(a2/h2)*l*wplOmega*phi2or2*(rr*w/chi));

	// Columns.
	ja[offset2 +  0] = BASE +           IDX(i - 3, j    );
	ja[offset2 +  1] = BASE +           IDX(i - 2, j    );
	ja[offset2 +  2] = BASE +           IDX(i - 1, j    );
	ja[offset2 +  3] = BASE +           IDX(i    , j - 3);
	ja[offset2 +  4] = BASE +           IDX(i    , j - 2);
	ja[offset2 +  5] = BASE +           IDX(i    , j - 1);
	ja[offset2 +  6] = BASE +           IDX(i    , j    );
	ja[offset2 +  7] = BASE +           IDX(i    , j + 1);
	ja[offset2 +  8] = BASE +           IDX(i + 1, j    );

	ja[offset2 +  9] = BASE +     dim + IDX(i - 4, j    );
	ja[offset2 + 10] = BASE +     dim + IDX(i - 3, j    );
	ja[offset2 + 11] = BASE +     dim + IDX(i - 2, j    );
	ja[offset2 + 12] = BASE +     dim + IDX(i - 1, j    );
	ja[offset2 + 13] = BASE +     dim + IDX(i    , j - 4);
	ja[offset2 + 14] = BASE +     dim + IDX(i    , j - 3);
	ja[offset2 + 15] = BASE +     dim + IDX(i    , j - 2);
	ja[offset2 + 16] = BASE +     dim + IDX(i    , j - 1);
	ja[offset2 + 17] = BASE +     dim + IDX(i    , j    );
	ja[offset2 + 18] = BASE +     dim + IDX(i    , j + 1);
	ja[offset2 + 19] = BASE +     dim + IDX(i + 1, j    );

	ja[offset2 + 20] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset2 + 21] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset2 + 22] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset2 + 23] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset2 + 24] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset2 + 25] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset2 + 26] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset2 + 27] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset2 + 28] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset2 + 29] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset2 + 30] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset2 + 31] = BASE + 5 * dim;


	// H: grid number 2.
	ia[2 * dim + IDX(i, j)] = BASE + offset3;

	// Values.
	aa[offset3 +  0] = dzodr*((S11)*(1.0/ri + dRu3));
	aa[offset3 +  1] = dzodr*((S12)*(1.0/ri + dRu3));
	aa[offset3 +  2] = dzodr*((S13)*(1.0/ri + dRu3));
	aa[offset3 +  3] = drodz*((D10)*dZu3);
	aa[offset3 +  4] = drodz*((D11)*dZu3);
	aa[offset3 +  5] = dzodr*((S14)*(1.0/ri + dRu3))-r2h2oalpha2dOmega2;
	aa[offset3 +  6] = drodz*((D13)*dZu3);
	aa[offset3 +  7] = drodz*((D14)*dZu3);
	aa[offset3 +  8] = dzodr*((S15)*(1.0/ri + dRu3));

	aa[offset3 +  9] = dzodr*((S11)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 10] = dzodr*((S12)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 11] = dzodr*((S13)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 12] = drodz*((D10)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 13] = drodz*((D11)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 14] = dzodr*((S14)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 15] = drodz*((D13)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 16] = drodz*((D14)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 17] = dzodr*((S15)*(r2*h2*dRu2/alpha2));

	aa[offset3 + 18] = dzodr*(S20); // CONSTANT!
	aa[offset3 + 19] = dzodr*((S21) + (S11)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 20] = dzodr*((S22) + (S12)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 21] = dzodr*((S23) + (S13)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 22] = drodz*((D20) + (D10)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 23] = drodz*((D21) + (D11)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 24] = dzodr*((S24) + (S14)*(2.0/ri + dRu1 + 2.0*dRu3)) + drodz*(D22) + r2h2oalpha2dOmega2 - 16.0*M_PI*dr2*dzodr*a2*l*l*phi2or2/h2;
	aa[offset3 + 25] = drodz*((D23) + (D13)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 26] = drodz*((D24) + (D14)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 27] = dzodr*((S25) + (S15)*(2.0/ri + dRu1 + 2.0*dRu3));

	aa[offset3 + 28] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;

	aa[offset3 + 29] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;

	aa[offset3 + 30] = dw_du(xi, m) * (8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2*(rr*w/chi));

	// Columns.
	ja[offset3 +  0] = BASE +           IDX(i - 3, j    );
	ja[offset3 +  1] = BASE +           IDX(i - 2, j    );
	ja[offset3 +  2] = BASE +           IDX(i - 1, j    );
	ja[offset3 +  3] = BASE +           IDX(i    , j - 2);
	ja[offset3 +  4] = BASE +           IDX(i    , j - 1);
	ja[offset3 +  5] = BASE +           IDX(i    , j    );
	ja[offset3 +  6] = BASE +           IDX(i    , j + 1);
	ja[offset3 +  7] = BASE +           IDX(i    , j + 2);
	ja[offset3 +  8] = BASE +           IDX(i + 1, j    );

	ja[offset3 +  9] = BASE +     dim + IDX(i - 3, j    );
	ja[offset3 + 10] = BASE +     dim + IDX(i - 2, j    );
	ja[offset3 + 11] = BASE +     dim + IDX(i - 1, j    );
	ja[offset3 + 12] = BASE +     dim + IDX(i    , j - 2);
	ja[offset3 + 13] = BASE +     dim + IDX(i    , j - 1);
	ja[offset3 + 14] = BASE +     dim + IDX(i    , j    );
	ja[offset3 + 15] = BASE +     dim + IDX(i    , j + 1);
	ja[offset3 + 16] = BASE +     dim + IDX(i    , j + 2);
	ja[offset3 + 17] = BASE +     dim + IDX(i + 1, j    );

	ja[offset3 + 18] = BASE + 2 * dim + IDX(i - 4, j    );
	ja[offset3 + 19] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset3 + 20] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset3 + 21] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset3 + 22] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset3 + 23] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset3 + 24] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset3 + 25] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset3 + 26] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset3 + 27] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset3 + 28] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset3 + 29] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset3 + 30] = BASE + 5 * dim;


	// A: grid number 3.
	ia[3 * dim + IDX(i, j)] = BASE + offset4;

	// Values.
	aa[offset4 +  0] = dzodr*((S11)*(-1.0/ri - dRu3));
	aa[offset4 +  1] = dzodr*((S12)*(-1.0/ri - dRu3));
	aa[offset4 +  2] = dzodr*((S13)*(-1.0/ri - dRu3));
	aa[offset4 +  3] = drodz*((D10)*(-dZu3));
	aa[offset4 +  4] = drodz*((D11)*(-dZu3));
	aa[offset4 +  5] = dzodr*((S13)*(-1.0/ri - dRu3)) + 0.5*r2h2oalpha2dOmega2 - 8.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset4 +  6] = drodz*((D13)*(-dZu3));
	aa[offset4 +  7] = drodz*((D14)*(-dZu3));
	aa[offset4 +  8] = dzodr*((S15)*(-1.0/ri - dRu3));

	aa[offset4 +  9] = dzodr*((S11)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 + 10] = dzodr*((S12)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 + 11] = dzodr*((S13)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 + 12] = drodz*((D10)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 13] = drodz*((D11)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 14] = dzodr*((S14)*(-0.5*r2*h2*dRu2/alpha2)) + 8.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset4 + 15] = drodz*((D13)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 16] = drodz*((D14)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 17] = dzodr*((S15)*(-0.5*r2*h2*dRu2/alpha2));

	aa[offset4 + 18] = dzodr*((S11)*(-dRu1));
	aa[offset4 + 19] = dzodr*((S12)*(-dRu1));
	aa[offset4 + 20] = dzodr*((S13)*(-dRu1));
	aa[offset4 + 21] = drodz*((D10)*(-dZu1));
	aa[offset4 + 22] = drodz*((D11)*(-dZu1));
	aa[offset4 + 23] = dzodr*((S14)*(-dRu1)) - 0.5*r2h2oalpha2dOmega2 + 8.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset4 + 24] = drodz*((D13)*(-dZu1));
	aa[offset4 + 25] = drodz*((D14)*(-dZu1));
	aa[offset4 + 26] = dzodr*((S15)*(-dRu1));

	aa[offset4 + 27] = (S20)*dzodr; // CONSTANT!
	aa[offset4 + 28] = (S21)*dzodr; // CONSTANT!
	aa[offset4 + 29] = (S22)*dzodr; // CONSTANT!
	aa[offset4 + 30] = (S23)*dzodr; // CONSTANT!
	aa[offset4 + 31] = (D20)*drodz; // CONSTANT!
	aa[offset4 + 32] = (D21)*drodz; // CONSTANT!
	aa[offset4 + 33] = (D22)*drodz + (S24)*dzodr + 8.0*M_PI*dr2*dzodr*(-l*l/h2 + r2*wplOmega2/alpha2)*a2*phi2or2;
	aa[offset4 + 34] = (D23)*drodz; // CONSTANT!
	aa[offset4 + 35] = (D24)*drodz; // CONSTANT!
	aa[offset4 + 36] = (S25)*dzodr; // CONSTANT!
	
	aa[offset4 + 37] = dzodr*((S11)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 38] = dzodr*((S12)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 39] = dzodr*((S13)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 40] = drodz*((D10)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 41] = drodz*((D11)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 42] = dzodr*((S14)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri))) + 8.0*M_PI*phi2or2*(dr2*dzodr*(l*l + 2.0*l*ri*dRu5 + a2*(-l*l/h2 + r2*wplOmega2/alpha2) + r2*(chi*(chi - 2.0*dXu5))) + (r2*(dzodr*dRu5*dRu5 + drodz*dZu5*dZu5)));
	aa[offset4 + 43] = drodz*((D13)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 44] = drodz*((D14)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 45] = dzodr*((S15)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));

	aa[offset4 + 46] = dw_du(xi, m) * (8.0*M_PI*phi2or2*(dr2*dzodr*(a2*r2*wplOmega/alpha2 + r2*w*(dXu5/chi - 1.0) + (rr*w/chi)*(l*l + 2.0*l*ri*dRu5 + a2*(-l*l/h2 + r2*wplOmega2/alpha2) + r2*(chi*(chi - 2.0*dXu5)))) + (rr*w/chi)*(r2*(dzodr*dRu5*dRu5 + drodz*dZu5*dZu5))));

	// Columns.
	ja[offset4 +  0] = BASE +           IDX(i - 3, j    );
	ja[offset4 +  1] = BASE +           IDX(i - 2, j    );
	ja[offset4 +  2] = BASE +           IDX(i - 1, j    );
	ja[offset4 +  3] = BASE +           IDX(i    , j - 2);
	ja[offset4 +  4] = BASE +           IDX(i    , j - 1);
	ja[offset4 +  5] = BASE +           IDX(i    , j    );
	ja[offset4 +  6] = BASE +           IDX(i    , j + 1);
	ja[offset4 +  7] = BASE +           IDX(i    , j + 2);
	ja[offset4 +  8] = BASE +           IDX(i + 1, j    );

	ja[offset4 +  9] = BASE +     dim + IDX(i - 3, j    );
	ja[offset4 + 10] = BASE +     dim + IDX(i - 2, j    );
	ja[offset4 + 11] = BASE +     dim + IDX(i - 1, j    );
	ja[offset4 + 12] = BASE +     dim + IDX(i    , j - 2);
	ja[offset4 + 13] = BASE +     dim + IDX(i    , j - 1);
	ja[offset4 + 14] = BASE +     dim + IDX(i    , j    );
	ja[offset4 + 15] = BASE +     dim + IDX(i    , j + 1);
	ja[offset4 + 16] = BASE +     dim + IDX(i    , j + 2);
	ja[offset4 + 17] = BASE +     dim + IDX(i + 1, j    );

	ja[offset4 + 18] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset4 + 19] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset4 + 20] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset4 + 21] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset4 + 22] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset4 + 23] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset4 + 24] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset4 + 25] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset4 + 26] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset4 + 27] = BASE + 3 * dim + IDX(i - 4, j    );
	ja[offset4 + 28] = BASE + 3 * dim + IDX(i - 3, j    );
	ja[offset4 + 29] = BASE + 3 * dim + IDX(i - 2, j    );
	ja[offset4 + 30] = BASE + 3 * dim + IDX(i - 1, j    );
	ja[offset4 + 31] = BASE + 3 * dim + IDX(i    , j - 2);
	ja[offset4 + 32] = BASE + 3 * dim + IDX(i    , j - 1);
	ja[offset4 + 33] = BASE + 3 * dim + IDX(i    , j    );
	ja[offset4 + 34] = BASE + 3 * dim + IDX(i    , j + 1);
	ja[offset4 + 35] = BASE + 3 * dim + IDX(i    , j + 2);
	ja[offset4 + 36] = BASE + 3 * dim + IDX(i + 1, j    );

	ja[offset4 + 37] = BASE + 4 * dim + IDX(i - 3, j    );
	ja[offset4 + 38] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset4 + 39] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset4 + 40] = BASE + 4 * dim + IDX(i    , j - 2);
	ja[offset4 + 41] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset4 + 42] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset4 + 43] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset4 + 44] = BASE + 4 * dim + IDX(i    , j + 2);
	ja[offset4 + 45] = BASE + 4 * dim + IDX(i + 1, j    );

	ja[offset4 + 46] = BASE + 5 * dim;

	// Psi: grid number 4.
	ia[4 * dim + IDX(i, j)] = BASE + offset5;

	// Values.
	aa[offset5 +  0] = dzodr*((S11)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  1] = dzodr*((S12)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  2] = dzodr*((S13)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  3] = drodz*((D10)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  4] = drodz*((D11)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  5] = dzodr*((S14)*(dRu5 + l/ri - chi*dr2*ri/rr))-2.0*dr2*dzodr*a2*wplOmega2/alpha2;
	aa[offset5 +  6] = drodz*((D13)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  7] = drodz*((D14)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  8] = dzodr*((S15)*(dRu5 + l/ri - chi*dr2*ri/rr));

	aa[offset5 +  9] = 2.0*dr2*dzodr*a2*l*wplOmega/alpha2;

	aa[offset5 + 10] = dzodr*((S11)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 + 11] = dzodr*((S12)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 + 12] = dzodr*((S13)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 + 13] = drodz*((D10)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 14] = drodz*((D11)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 15] = dzodr*((S14)*(dRu5 + l/ri - chi*dr2*ri/rr))+2.0*l*l*dzodr*(a2/h2)*(1.0/(ri*ri));
	aa[offset5 + 16] = drodz*((D13)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 17] = drodz*((D14)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 18] = dzodr*((S15)*(dRu5 + l/ri - chi*dr2*ri/rr));

	aa[offset5 + 19] = 2.0*dzodr*(dr2*a2*(wplOmega2/alpha2 - m2) - l*l*(a2/h2)*(1.0/(ri*ri)));

	aa[offset5 + 20] = (S20)*dzodr; // CONSTANT!
	aa[offset5 + 21] = (S21)*dzodr + dzodr*((S11)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 22] = (S22)*dzodr + dzodr*((S12)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 23] = (S23)*dzodr + dzodr*((S13)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 24] = (D20)*drodz + drodz*((D10)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 25] = (D21)*drodz + drodz*((D11)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 26] = (D22)*drodz + (S24)*drodz + drodz*((S14)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 27] = (D23)*drodz + drodz*((D13)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 28] = (D24)*drodz + drodz*((D14)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 29] = (S25)*dzodr + dzodr*((S15)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));

	aa[offset5 + 30] = dw_du(xi, m) * (dr2*dzodr*(2.0*a2*wplOmega/alpha2 + (-w/chi)*(2.0*chi - 2.0*(l + 1.0)/rr - 2.0*dXu5 - dXu1 - dXu3)));

	// Columns.
	ja[offset5 +  0] = BASE +           IDX(i - 3, j    );
	ja[offset5 +  1] = BASE +           IDX(i - 2, j    );
	ja[offset5 +  2] = BASE +           IDX(i - 1, j    );
	ja[offset5 +  3] = BASE +           IDX(i    , j - 2);
	ja[offset5 +  4] = BASE +           IDX(i    , j - 1);
	ja[offset5 +  5] = BASE +           IDX(i    , j    );
	ja[offset5 +  6] = BASE +           IDX(i    , j + 1);
	ja[offset5 +  7] = BASE +           IDX(i    , j + 2);
	ja[offset5 +  8] = BASE +           IDX(i + 1, j    );

	ja[offset5 +  9] = BASE +     dim + IDX(i    , j    );

	ja[offset5 + 10] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset5 + 11] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset5 + 12] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset5 + 13] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset5 + 14] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset5 + 15] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset5 + 16] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset5 + 17] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset5 + 18] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset5 + 19] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset5 + 20] = BASE + 4 * dim + IDX(i - 4, j    );
	ja[offset5 + 21] = BASE + 4 * dim + IDX(i - 3, j    );
	ja[offset5 + 22] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset5 + 23] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset5 + 24] = BASE + 4 * dim + IDX(i    , j - 2);
	ja[offset5 + 25] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset5 + 26] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset5 + 27] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset5 + 28] = BASE + 4 * dim + IDX(i    , j + 2);
	ja[offset5 + 29] = BASE + 4 * dim + IDX(i + 1, j    );

	ja[offset5 + 30] = BASE + 5 * dim;


	// All done.
	return;
}

// Jacobian for semionesided-semionesided 4th order stencil and variable omega.
void jacobian_4th_order_variable_omega_ss
(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings. 
	MKL_INT *ja,		// CSR array for columns.
	const MKL_INT NrTotal, 	// Grid total dimension in r.
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT i, 	// Integer coordinate for r: 0 <= i < NrTotal.
	const MKL_INT j, 	// Integer coordinate for z: 0 <= j < NzTotal.
	const double dr, 	// Spatial step in r.
	const double dz,	// Spatial step in z.
	const MKL_INT l, 	// Scalar field rotation number.
	const double m, 	// Scalar field mass.
	const double xi,	// Scalar field frequency variable.
	// Now come the grid variables. For ss stencil, each grid function has 11 variables.
	const double u104, const double u114, const double u124, const double u134, const double u140, const double u141, const double u142, const double u143, const double u144, const double u145, const double u154,
	const double u204, const double u214, const double u224, const double u234, const double u240, const double u241, const double u242, const double u243, const double u244, const double u245, const double u254,
	const double u304, const double u314, const double u324, const double u334, const double u340, const double u341, const double u342, const double u343, const double u344, const double u345, const double u354,
	const double u404, const double u414, const double u424, const double u434, const double u440, const double u441, const double u442, const double u443, const double u444, const double u445, const double u454,
	const double u504, const double u514, const double u524, const double u534, const double u540, const double u541, const double u542, const double u543, const double u544, const double u545, const double u554,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5 	// Number of elements filled before filling function 5
)
{
	// Physical variables.
	double alpha = exp(u144);
	double Omega = u244;
	double h    = exp(u344);
	double a    = exp(u444);
	double psi  = u544;
	double w = omega_calc(xi, m);

	// Coordinates.
	double ri = (double)i - 1.5;
	double zi = (double)j - 1.5;
	double r = ri * dr;
	double z = zi * dz;
	double r2 = r * r;
	double z2 = z * z;
	double rr = sqrt(r2 + z2);

	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	double dr2 = dr * dr;
	double dz2 = dz * dz;

	// Scalar field mass and frequency.
	double m2 = m * m;
	double w2 = w * w;
	double chi = sqrt(m2 - w2);

	// Scalar field.
	double rlm1 = (l == 1) ? 1.0 : pow(r, l - 1);
	double rl = rlm1 * r;
	double phior = rlm1 * exp(-chi * rr) * exp(psi);
	double phi = r * phior;
	double phi2or2 = phior * phior;
	double phi2 = phi * phi;

	// Shift combined with scalar field rotation.
	double wplOmega = w + l * Omega;
	double wplOmega2 = wplOmega * wplOmega;

	// Finite differences.
	double dRu1 = S11 * u114 + S12 * u124 + S13 * u134 + S14 * u144 + S14 * u154;
	double dRu2 = S11 * u214 + S12 * u224 + S13 * u234 + S14 * u244 + S14 * u254;
	double dRu3 = S11 * u314 + S12 * u324 + S13 * u334 + S14 * u344 + S14 * u354;
	double dRu4 = S11 * u414 + S12 * u424 + S13 * u434 + S14 * u444 + S14 * u454;
	double dRu5 = S11 * u514 + S12 * u524 + S13 * u534 + S14 * u544 + S14 * u554;

	double dZu1 = S11 * u141 + S12 * u142 + S13 * u143 + S14 * u144 + S14 * u145;
	double dZu2 = S11 * u241 + S12 * u242 + S13 * u243 + S14 * u244 + S14 * u245;
	double dZu3 = S11 * u341 + S12 * u342 + S13 * u343 + S14 * u344 + S14 * u345;
	double dZu4 = S11 * u441 + S12 * u442 + S13 * u443 + S14 * u444 + S14 * u445;
	double dZu5 = S11 * u541 + S12 * u542 + S13 * u543 + S14 * u544 + S14 * u545;

	// Radial derivatives.
	double dXu5 = (ri * dRu5 + zi * dZu5) / rr;
	double dXu1 = (ri * dRu1 + zi * dZu1) / rr;
	double dXu3 = (ri * dRu3 + zi * dZu3) / rr;

	// Squared variables.
	double alpha2 = alpha * alpha;
	double h2 = h * h;
	double a2 = a * a;

	// Common term.
	double r2h2oalpha2dOmega2 = r2 * h2 * (dzodr * dRu2 * dRu2 + drodz * dZu2 * dZu2) / alpha2;

	// Alpha: grid number 0.
	ia[IDX(i, j)] = BASE + offset1;

	// Values.
	aa[offset1 +  0] = dzodr*((S20)); // CONSTANT!
	aa[offset1 +  1] = dzodr*((S21) + (S11)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  2] = dzodr*((S22) + (S12)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  3] = dzodr*((S23) + (S13)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  4] = drodz*((S20)); // CONSTANT!
	aa[offset1 +  5] = drodz*((S21) + (S11)*(2.0*dZu1 + dZu3));
	aa[offset1 +  6] = drodz*((S22) + (S12)*(2.0*dZu1 + dZu3));
	aa[offset1 +  7] = drodz*((S23) + (S13)*(2.0*dZu1 + dZu3));
	aa[offset1 +  8] = dzodr*((S24) + (S14)*(1.0/ri + 2.0*dRu1 + dRu3)) + drodz*((S24) + (S14)*(2.0*dZu1 + dZu3)) + r2h2oalpha2dOmega2 + 16.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset1 +  9] = drodz*((S25) + (S15)*(2.0*dZu1 + dZu3));
	aa[offset1 + 10] = dzodr*((S25) + (S15)*(1.0/ri + 2.0*dRu1 + dRu3));

	aa[offset1 + 11] = dzodr*((S11)*(-r2*h2*dRu2/alpha2));
	aa[offset1 + 12] = dzodr*((S12)*(-r2*h2*dRu2/alpha2));
	aa[offset1 + 13] = dzodr*((S13)*(-r2*h2*dRu2/alpha2));
	aa[offset1 + 14] = drodz*((S11)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 15] = drodz*((S12)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 16] = drodz*((S13)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 17] = dzodr*((S14)*(-r2*h2*dRu2/alpha2)) + drodz*((S14)*(-r2*h2*dZu2/alpha2)) - 16.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset1 + 18] = drodz*((S15)*(-r2*h2*dZu2/alpha2));
	aa[offset1 + 19] = dzodr*((S15)*(-r2*h2*dRu2/alpha2));

	aa[offset1 + 20] = dzodr*((S11)*dRu1);
	aa[offset1 + 21] = dzodr*((S12)*dRu1);
	aa[offset1 + 22] = dzodr*((S13)*dRu1);
	aa[offset1 + 23] = drodz*((S11)*dZu1);
	aa[offset1 + 24] = drodz*((S12)*dZu1);
	aa[offset1 + 25] = drodz*((S13)*dZu1);
	aa[offset1 + 26] = dzodr*((S14)*dRu1) + drodz*((S14)*dZu1) - r2h2oalpha2dOmega2;
	aa[offset1 + 27] = drodz*((S15)*dZu1);
	aa[offset1 + 28] = dzodr*((S15)*dRu1);

	aa[offset1 + 29] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;

	aa[offset1 + 30] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;

	aa[offset1 + 31] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*wplOmega*phi2/alpha2 + 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2*(rr*w/chi));

	// Columns.
	ja[offset1 +  0] = BASE +           IDX(i - 4, j    );
	ja[offset1 +  1] = BASE +           IDX(i - 3, j    );
	ja[offset1 +  2] = BASE +           IDX(i - 2, j    );
	ja[offset1 +  3] = BASE +           IDX(i - 1, j    );
	ja[offset1 +  4] = BASE +           IDX(i    , j - 4);
	ja[offset1 +  5] = BASE +           IDX(i    , j - 3);
	ja[offset1 +  6] = BASE +           IDX(i    , j - 2);
	ja[offset1 +  7] = BASE +           IDX(i    , j - 1);
	ja[offset1 +  8] = BASE +           IDX(i    , j    );
	ja[offset1 +  9] = BASE +           IDX(i    , j + 1);
	ja[offset1 + 10] = BASE +           IDX(i + 1, j    );

	ja[offset1 + 11] = BASE +     dim + IDX(i - 3, j    );
	ja[offset1 + 12] = BASE +     dim + IDX(i - 2, j    );
	ja[offset1 + 13] = BASE +     dim + IDX(i - 1, j    );
	ja[offset1 + 14] = BASE +     dim + IDX(i    , j - 3);
	ja[offset1 + 15] = BASE +     dim + IDX(i    , j - 2);
	ja[offset1 + 16] = BASE +     dim + IDX(i    , j - 1);
	ja[offset1 + 17] = BASE +     dim + IDX(i    , j    );
	ja[offset1 + 18] = BASE +     dim + IDX(i    , j + 1);
	ja[offset1 + 19] = BASE +     dim + IDX(i + 1, j    );

	ja[offset1 + 20] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset1 + 21] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset1 + 22] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset1 + 23] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset1 + 24] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset1 + 25] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset1 + 26] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset1 + 27] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset1 + 28] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset1 + 29] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset1 + 30] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset1 + 31] = BASE + 5 * dim;


	// Beta: grid number 1.
	ia[dim + IDX(i, j)] = BASE + offset2;

	// Values.
	aa[offset2 +  0] = dzodr*((S11)*(-dRu2));
	aa[offset2 +  1] = dzodr*((S12)*(-dRu2));
	aa[offset2 +  2] = dzodr*((S13)*(-dRu2));
	aa[offset2 +  3] = drodz*((S11)*(-dZu2));
	aa[offset2 +  4] = drodz*((S12)*(-dZu2));
	aa[offset2 +  5] = drodz*((S13)*(-dZu2));
	aa[offset2 +  6] = dzodr*((S14)*(-dRu2)) + drodz*((S14)*(-dZu2));
	aa[offset2 +  7] = drodz*((S15)*(-dZu2));
	aa[offset2 +  8] = dzodr*((S15)*(-dRu2));

	aa[offset2 +  9] = dzodr*(S20); // CONSTANT!
	aa[offset2 + 10] = dzodr*((S21) + (S11)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 + 11] = dzodr*((S22) + (S12)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 + 12] = dzodr*((S23) + (S13)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 + 13] = drodz*(S20); // CONSTANT!
	aa[offset2 + 14] = drodz*((S21) + (S11)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 15] = drodz*((S22) + (S12)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 16] = drodz*((S23) + (S13)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 17] = dzodr*((S24) + (S14)*(3.0/ri - dRu1 + 3.0*dRu3)) + drodz*((S24) + (S14)*(-dZu1 + 3.0*dZu3)) - 16.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset2 + 18] = drodz*((S25) + (S15)*(-dZu1 + 3.0*dZu3));
	aa[offset2 + 19] = dzodr*((S25) + (S15)*(3.0/ri - dRu1 + 3.0*dRu3));
	
	aa[offset2 + 20] = dzodr*((S11)*(3.0*dRu2));
	aa[offset2 + 21] = dzodr*((S12)*(3.0*dRu2));
	aa[offset2 + 22] = dzodr*((S13)*(3.0*dRu2));
	aa[offset2 + 23] = drodz*((S11)*(3.0*dZu2));
	aa[offset2 + 24] = drodz*((S12)*(3.0*dZu2));
	aa[offset2 + 25] = drodz*((S13)*(3.0*dZu2));
	aa[offset2 + 26] = dzodr*((S14)*(3.0*dRu2)) + drodz*((S14)*(3.0*dZu2)) + 32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;
	aa[offset2 + 27] = drodz*((S15)*(3.0*dZu2));
	aa[offset2 + 28] = dzodr*((S15)*(3.0*dRu2));

	aa[offset2 + 29] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;
	
	aa[offset2 + 30] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;

	aa[offset2 + 31] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*l*phi2or2/h2 - 32.0*M_PI*dr2*dzodr*(a2/h2)*l*wplOmega*phi2or2*(rr*w/chi));

	// Columns.
	ja[offset2 +  0] = BASE +           IDX(i - 3, j    );
	ja[offset2 +  1] = BASE +           IDX(i - 2, j    );
	ja[offset2 +  2] = BASE +           IDX(i - 1, j    );
	ja[offset2 +  3] = BASE +           IDX(i    , j - 3);
	ja[offset2 +  4] = BASE +           IDX(i    , j - 2);
	ja[offset2 +  5] = BASE +           IDX(i    , j - 1);
	ja[offset2 +  6] = BASE +           IDX(i    , j    );
	ja[offset2 +  7] = BASE +           IDX(i    , j + 1);
	ja[offset2 +  8] = BASE +           IDX(i + 1, j    );

	ja[offset2 +  9] = BASE +     dim + IDX(i - 4, j    );
	ja[offset2 + 10] = BASE +     dim + IDX(i - 3, j    );
	ja[offset2 + 11] = BASE +     dim + IDX(i - 2, j    );
	ja[offset2 + 12] = BASE +     dim + IDX(i - 1, j    );
	ja[offset2 + 13] = BASE +     dim + IDX(i    , j - 4);
	ja[offset2 + 14] = BASE +     dim + IDX(i    , j - 3);
	ja[offset2 + 15] = BASE +     dim + IDX(i    , j - 2);
	ja[offset2 + 16] = BASE +     dim + IDX(i    , j - 1);
	ja[offset2 + 17] = BASE +     dim + IDX(i    , j    );
	ja[offset2 + 18] = BASE +     dim + IDX(i    , j + 1);
	ja[offset2 + 19] = BASE +     dim + IDX(i + 1, j    );

	ja[offset2 + 20] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset2 + 21] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset2 + 22] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset2 + 23] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset2 + 24] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset2 + 25] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset2 + 26] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset2 + 27] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset2 + 28] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset2 + 29] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset2 + 30] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset2 + 31] = BASE + 5 * dim;


	// H: grid number 2.
	ia[2 * dim + IDX(i, j)] = BASE + offset3;

	// Values.
	aa[offset3 +  0] = dzodr*((S11)*(1.0/ri + dRu3));
	aa[offset3 +  1] = dzodr*((S12)*(1.0/ri + dRu3));
	aa[offset3 +  2] = dzodr*((S13)*(1.0/ri + dRu3));
	aa[offset3 +  3] = drodz*((S11)*dZu3);
	aa[offset3 +  4] = drodz*((S12)*dZu3);
	aa[offset3 +  5] = drodz*((S13)*dZu3);
	aa[offset3 +  6] = dzodr*((S14)*(1.0/ri + dRu3))+drodz*((S14)*dZu3)-r2h2oalpha2dOmega2;
	aa[offset3 +  7] = drodz*((S15)*dZu3);
	aa[offset3 +  8] = dzodr*((S15)*(1.0/ri + dRu3));

	aa[offset3 +  9] = dzodr*((S11)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 10] = dzodr*((S12)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 11] = dzodr*((S13)*(r2*h2*dRu2/alpha2));
	aa[offset3 + 12] = drodz*((S11)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 13] = drodz*((S12)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 14] = drodz*((S13)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 15] = dzodr*((S14)*(r2*h2*dRu2/alpha2))+drodz*((S14)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 16] = drodz*((S15)*(r2*h2*dZu2/alpha2));
	aa[offset3 + 17] = dzodr*((S15)*(r2*h2*dRu2/alpha2));

	aa[offset3 + 18] = dzodr*(S20); // CONSTANT!
	aa[offset3 + 19] = dzodr*((S21) + (S11)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 20] = dzodr*((S22) + (S12)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 21] = dzodr*((S23) + (S13)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 22] = drodz*(S20); // CONSTANT!
	aa[offset3 + 23] = drodz*((S21) + (S11)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 24] = drodz*((S22) + (S12)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 25] = drodz*((S23) + (S13)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 26] = dzodr*((S24) + (S14)*(2.0/ri + dRu1 + 2.0*dRu3)) + drodz*((S24) + (S14)*(dZu1 + 2.0*dZu3)) + r2h2oalpha2dOmega2 - 16.0*M_PI*dr2*dzodr*a2*l*l*phi2or2/h2;
	aa[offset3 + 27] = drodz*((S25) + (S15)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 28] = dzodr*((S25) + (S15)*(2.0/ri + dRu1 + 2.0*dRu3));

	aa[offset3 + 29] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;

	aa[offset3 + 30] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;

	aa[offset3 + 31] = dw_du(xi, m) * (8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2*(rr*w/chi));

	// Columns.
	ja[offset3 +  0] = BASE +           IDX(i - 3, j    );
	ja[offset3 +  1] = BASE +           IDX(i - 2, j    );
	ja[offset3 +  2] = BASE +           IDX(i - 1, j    );
	ja[offset3 +  3] = BASE +           IDX(i    , j - 3);
	ja[offset3 +  4] = BASE +           IDX(i    , j - 2);
	ja[offset3 +  5] = BASE +           IDX(i    , j - 1);
	ja[offset3 +  6] = BASE +           IDX(i    , j    );
	ja[offset3 +  7] = BASE +           IDX(i    , j + 1);
	ja[offset3 +  8] = BASE +           IDX(i + 1, j    );

	ja[offset3 +  9] = BASE +     dim + IDX(i - 3, j    );
	ja[offset3 + 10] = BASE +     dim + IDX(i - 2, j    );
	ja[offset3 + 11] = BASE +     dim + IDX(i - 1, j    );
	ja[offset3 + 12] = BASE +     dim + IDX(i    , j - 3);
	ja[offset3 + 13] = BASE +     dim + IDX(i    , j - 2);
	ja[offset3 + 14] = BASE +     dim + IDX(i    , j - 1);
	ja[offset3 + 15] = BASE +     dim + IDX(i    , j    );
	ja[offset3 + 16] = BASE +     dim + IDX(i    , j + 1);
	ja[offset3 + 17] = BASE +     dim + IDX(i + 1, j    );

	ja[offset3 + 18] = BASE + 2 * dim + IDX(i - 4, j    );
	ja[offset3 + 19] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset3 + 20] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset3 + 21] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset3 + 22] = BASE + 2 * dim + IDX(i    , j - 4);
	ja[offset3 + 23] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset3 + 24] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset3 + 25] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset3 + 26] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset3 + 27] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset3 + 28] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset3 + 29] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset3 + 30] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset3 + 31] = BASE + 5 * dim;


	// A: grid number 3.
	ia[3 * dim + IDX(i, j)] = BASE + offset4;

	// Values.
	aa[offset4 +  0] = dzodr*((S11)*(-1.0/ri - dRu3));
	aa[offset4 +  1] = dzodr*((S12)*(-1.0/ri - dRu3));
	aa[offset4 +  2] = dzodr*((S13)*(-1.0/ri - dRu3));
	aa[offset4 +  3] = drodz*((S11)*(-dZu3));
	aa[offset4 +  4] = drodz*((S12)*(-dZu3));
	aa[offset4 +  5] = drodz*((S13)*(-dZu3));
	aa[offset4 +  6] = dzodr*((S13)*(-1.0/ri - dRu3)) + drodz*((S14)*(-dZu3)) + 0.5*r2h2oalpha2dOmega2 - 8.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset4 +  7] = drodz*((S15)*(-dZu3));
	aa[offset4 +  8] = dzodr*((S15)*(-1.0/ri - dRu3));

	aa[offset4 +  9] = dzodr*((S11)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 + 10] = dzodr*((S12)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 + 11] = dzodr*((S13)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 + 12] = drodz*((S11)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 13] = drodz*((S12)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 14] = drodz*((S13)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 15] = dzodr*((S14)*(-0.5*r2*h2*dRu2/alpha2)) + drodz*((S14)*(-0.5*r2*h2*dZu2/alpha2)) + 8.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset4 + 16] = drodz*((S15)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 + 17] = dzodr*((S15)*(-0.5*r2*h2*dRu2/alpha2));

	aa[offset4 + 18] = dzodr*((S11)*(-dRu1));
	aa[offset4 + 19] = dzodr*((S12)*(-dRu1));
	aa[offset4 + 20] = dzodr*((S13)*(-dRu1));
	aa[offset4 + 21] = drodz*((S11)*(-dZu1));
	aa[offset4 + 22] = drodz*((S12)*(-dZu1));
	aa[offset4 + 23] = drodz*((S13)*(-dZu1));
	aa[offset4 + 24] = dzodr*((S14)*(-dRu1)) + drodz*((S14)*(-dZu1)) - 0.5*r2h2oalpha2dOmega2 + 8.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset4 + 25] = drodz*((S15)*(-dZu1));
	aa[offset4 + 26] = dzodr*((S15)*(-dRu1));

	aa[offset4 + 27] = (S20)*dzodr; // CONSTANT!
	aa[offset4 + 28] = (S21)*dzodr; // CONSTANT!
	aa[offset4 + 29] = (S22)*dzodr; // CONSTANT!
	aa[offset4 + 30] = (S23)*dzodr; // CONSTANT!
	aa[offset4 + 31] = (S20)*drodz; // CONSTANT!
	aa[offset4 + 32] = (S21)*drodz; // CONSTANT!
	aa[offset4 + 33] = (S22)*drodz; // CONSTANT!
	aa[offset4 + 34] = (S23)*drodz; // CONSTANT!
	aa[offset4 + 35] = (S24)*drodz + (S24)*dzodr + 8.0*M_PI*dr2*dzodr*(-l*l/h2 + r2*wplOmega2/alpha2)*a2*phi2or2;
	aa[offset4 + 36] = (S25)*drodz; // CONSTANT!
	aa[offset4 + 37] = (S25)*dzodr; // CONSTANT!
	
	aa[offset4 + 38] = dzodr*((S11)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 39] = dzodr*((S12)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 40] = dzodr*((S13)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 41] = drodz*((S11)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 42] = drodz*((S12)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 43] = drodz*((S13)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 44] = dzodr*((S14)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri))) + drodz*((S14)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr))) + 8.0*M_PI*phi2or2*(dr2*dzodr*(l*l + 2.0*l*ri*dRu5 + a2*(-l*l/h2 + r2*wplOmega2/alpha2) + r2*(chi*(chi - 2.0*dXu5))) + (r2*(dzodr*dRu5*dRu5 + drodz*dZu5*dZu5)));
	aa[offset4 + 45] = drodz*((S15)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 46] = dzodr*((S15)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));

	aa[offset4 + 47] = dw_du(xi, m) * (8.0*M_PI*phi2or2*(dr2*dzodr*(a2*r2*wplOmega/alpha2 + r2*w*(dXu5/chi - 1.0) + (rr*w/chi)*(l*l + 2.0*l*ri*dRu5 + a2*(-l*l/h2 + r2*wplOmega2/alpha2) + r2*(chi*(chi - 2.0*dXu5)))) + (rr*w/chi)*(r2*(dzodr*dRu5*dRu5 + drodz*dZu5*dZu5))));

	// Columns.
	ja[offset4 +  0] = BASE +           IDX(i - 3, j    );
	ja[offset4 +  1] = BASE +           IDX(i - 2, j    );
	ja[offset4 +  2] = BASE +           IDX(i - 1, j    );
	ja[offset4 +  3] = BASE +           IDX(i    , j - 3);
	ja[offset4 +  4] = BASE +           IDX(i    , j - 2);
	ja[offset4 +  5] = BASE +           IDX(i    , j - 1);
	ja[offset4 +  6] = BASE +           IDX(i    , j    );
	ja[offset4 +  7] = BASE +           IDX(i    , j + 1);
	ja[offset4 +  8] = BASE +           IDX(i + 1, j    );

	ja[offset4 +  9] = BASE +     dim + IDX(i - 3, j    );
	ja[offset4 + 10] = BASE +     dim + IDX(i - 2, j    );
	ja[offset4 + 11] = BASE +     dim + IDX(i - 1, j    );
	ja[offset4 + 12] = BASE +     dim + IDX(i    , j - 3);
	ja[offset4 + 13] = BASE +     dim + IDX(i    , j - 2);
	ja[offset4 + 14] = BASE +     dim + IDX(i    , j - 1);
	ja[offset4 + 15] = BASE +     dim + IDX(i    , j    );
	ja[offset4 + 16] = BASE +     dim + IDX(i    , j + 1);
	ja[offset4 + 17] = BASE +     dim + IDX(i + 1, j    );

	ja[offset4 + 18] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset4 + 19] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset4 + 20] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset4 + 21] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset4 + 22] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset4 + 23] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset4 + 24] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset4 + 25] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset4 + 26] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset4 + 27] = BASE + 3 * dim + IDX(i - 4, j    );
	ja[offset4 + 28] = BASE + 3 * dim + IDX(i - 3, j    );
	ja[offset4 + 29] = BASE + 3 * dim + IDX(i - 2, j    );
	ja[offset4 + 30] = BASE + 3 * dim + IDX(i - 1, j    );
	ja[offset4 + 31] = BASE + 3 * dim + IDX(i    , j - 4);
	ja[offset4 + 32] = BASE + 3 * dim + IDX(i    , j - 3);
	ja[offset4 + 33] = BASE + 3 * dim + IDX(i    , j - 2);
	ja[offset4 + 34] = BASE + 3 * dim + IDX(i    , j - 1);
	ja[offset4 + 35] = BASE + 3 * dim + IDX(i    , j    );
	ja[offset4 + 36] = BASE + 3 * dim + IDX(i    , j + 1);
	ja[offset4 + 37] = BASE + 3 * dim + IDX(i + 1, j    );

	ja[offset4 + 38] = BASE + 4 * dim + IDX(i - 3, j    );
	ja[offset4 + 39] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset4 + 40] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset4 + 41] = BASE + 4 * dim + IDX(i    , j - 3);
	ja[offset4 + 42] = BASE + 4 * dim + IDX(i    , j - 2);
	ja[offset4 + 43] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset4 + 44] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset4 + 45] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset4 + 46] = BASE + 4 * dim + IDX(i + 1, j    );

	ja[offset4 + 47] = BASE + 5 * dim;

	// Psi: grid number 4.
	ia[4 * dim + IDX(i, j)] = BASE + offset5;

	// Values.
	aa[offset5 +  0] = dzodr*((S11)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  1] = dzodr*((S12)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  2] = dzodr*((S13)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  3] = drodz*((S11)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  4] = drodz*((S12)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  5] = drodz*((S13)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  6] = dzodr*((S14)*(dRu5 + l/ri - chi*dr2*ri/rr))+drodz*((S14)*(dZu5 - chi*dz2*zi/rr))-2.0*dr2*dzodr*a2*wplOmega2/alpha2;
	aa[offset5 +  7] = drodz*((S15)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  8] = dzodr*((S15)*(dRu5 + l/ri - chi*dr2*ri/rr));

	aa[offset5 +  9] = 2.0*dr2*dzodr*a2*l*wplOmega/alpha2;

	aa[offset5 + 10] = dzodr*((S11)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 + 11] = dzodr*((S12)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 + 12] = dzodr*((S13)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 + 13] = drodz*((S11)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 14] = drodz*((S12)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 15] = drodz*((S13)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 16] = dzodr*((S14)*(dRu5 + l/ri - chi*dr2*ri/rr))+drodz*((S14)*(dZu5 - chi*dz2*zi/rr))+2.0*l*l*dzodr*(a2/h2)*(1.0/(ri*ri));
	aa[offset5 + 17] = drodz*((S15)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 + 18] = dzodr*((S15)*(dRu5 + l/ri - chi*dr2*ri/rr));

	aa[offset5 + 19] = 2.0*dzodr*(dr2*a2*(wplOmega2/alpha2 - m2) - l*l*(a2/h2)*(1.0/(ri*ri)));

	aa[offset5 + 20] = (S20)*dzodr; // CONSTANT!
	aa[offset5 + 21] = (S21)*dzodr + dzodr*((S11)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 22] = (S22)*dzodr + dzodr*((S12)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 23] = (S23)*dzodr + dzodr*((S13)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 24] = (S20)*drodz; // CONSTANT!
	aa[offset5 + 25] = (S21)*drodz + drodz*((S11)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 26] = (S22)*drodz + drodz*((S12)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 27] = (S23)*drodz + drodz*((S13)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 28] = (S24)*drodz + drodz*((S14)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr)) + (S24)*drodz + drodz*((S14)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 29] = (S25)*drodz + drodz*((S15)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 30] = (S25)*dzodr + dzodr*((S15)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));

	aa[offset5 + 31] = dw_du(xi, m) * (dr2*dzodr*(2.0*a2*wplOmega/alpha2 + (-w/chi)*(2.0*chi - 2.0*(l + 1.0)/rr - 2.0*dXu5 - dXu1 - dXu3)));

	// Columns.
	ja[offset5 +  0] = BASE +           IDX(i - 3, j    );
	ja[offset5 +  1] = BASE +           IDX(i - 2, j    );
	ja[offset5 +  2] = BASE +           IDX(i - 1, j    );
	ja[offset5 +  3] = BASE +           IDX(i    , j - 3);
	ja[offset5 +  4] = BASE +           IDX(i    , j - 2);
	ja[offset5 +  5] = BASE +           IDX(i    , j - 1);
	ja[offset5 +  6] = BASE +           IDX(i    , j    );
	ja[offset5 +  7] = BASE +           IDX(i    , j + 1);
	ja[offset5 +  8] = BASE +           IDX(i + 1, j    );

	ja[offset5 +  9] = BASE +     dim + IDX(i    , j    );

	ja[offset5 + 10] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset5 + 11] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset5 + 12] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset5 + 13] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset5 + 14] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset5 + 15] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset5 + 16] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset5 + 17] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset5 + 18] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset5 + 19] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset5 + 20] = BASE + 4 * dim + IDX(i - 4, j    );
	ja[offset5 + 21] = BASE + 4 * dim + IDX(i - 3, j    );
	ja[offset5 + 22] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset5 + 23] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset5 + 24] = BASE + 4 * dim + IDX(i    , j - 4);
	ja[offset5 + 25] = BASE + 4 * dim + IDX(i    , j - 3);
	ja[offset5 + 26] = BASE + 4 * dim + IDX(i    , j - 2);
	ja[offset5 + 27] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset5 + 28] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset5 + 29] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset5 + 30] = BASE + 4 * dim + IDX(i + 1, j    );

	ja[offset5 + 31] = BASE + 5 * dim;


	// All done.
	return;
}