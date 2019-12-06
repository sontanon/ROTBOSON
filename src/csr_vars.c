// Include headers.
#include "tools.h"
#include "omega_calc.h"

// Finite difference coefficients for 2nd order.
const double D10 = -0.5;
const double D11 = 0.0;
const double D12 = +0.5;

const double D20 = +1.0;
const double D21 = -2.0;
const double D22 = +1.0;

// Jacobian for centered-centered 2nd order stencil and variable omega.
void jacobian_2nd_order_variable_omega_cc
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
	// Now come the grid variables. For cc stencil, each grid function has 5 variables.
	const double u101, const double u110, const double u111, const double u112, const double u121,
	const double u201, const double u210, const double u211, const double u212, const double u221,
	const double u301, const double u310, const double u311, const double u312, const double u321,
	const double u401, const double u410, const double u411, const double u412, const double u421,
	const double u501, const double u510, const double u511, const double u512, const double u521,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5 	// Number of elements filled before filling function 5.
)
{
	// Physical variables.
	double alpha = exp(u111);
	double Omega = u211;
	double h    = exp(u311);
	double a    = exp(u411);
	double psi  = u511;

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
	double dRu1 = D10 * u101 + D12 * u121;
	double dRu2 = D10 * u201 + D12 * u221;
	double dRu3 = D10 * u301 + D12 * u321;
	double dRu4 = D10 * u401 + D12 * u421;
	double dRu5 = D10 * u501 + D12 * u521;

	double dZu1 = D10 * u110 + D12 * u112;
	double dZu2 = D10 * u210 + D12 * u212;
	double dZu3 = D10 * u310 + D12 * u312;
	double dZu4 = D10 * u410 + D12 * u412;
	double dZu5 = D10 * u510 + D12 * u512;

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
	aa[offset1 +  1] = drodz*((D20) + (D10)*(2.0*dZu1 + dZu3));
	aa[offset1 +  2] = (D21)*(dzodr + drodz) + r2h2oalpha2dOmega2 + 16.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset1 +  3] = 2.0*(D20)*drodz - (aa[offset1 +  1]);
	aa[offset1 +  4] = 2.0*(D20)*dzodr - (aa[offset1 +  0]);

	aa[offset1 +  5] = dzodr*((D10)*(-r2*h2*dRu2/alpha2));
	aa[offset1 +  6] = drodz*((D10)*(-r2*h2*dZu2/alpha2));
	aa[offset1 +  7] = -16.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset1 +  8] = -aa[offset1 +  6];
	aa[offset1 +  9] = -aa[offset1 +  5];

	aa[offset1 + 10] = dzodr*((D10)*dRu1);
	aa[offset1 + 11] = drodz*((D10)*dZu1);
	aa[offset1 + 12] = -r2h2oalpha2dOmega2;
	aa[offset1 + 13] = -aa[offset1 + 11];
	aa[offset1 + 14] = -aa[offset1 + 10];

	aa[offset1 + 15] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;

	aa[offset1 + 16] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;

	aa[offset1 + 17] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*wplOmega*phi2/alpha2 + 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2*(rr*w/chi));

	// Columns.
	ja[offset1 +  0] = BASE +           IDX(i - 1, j    );
	ja[offset1 +  1] = BASE +           IDX(i    , j - 1);
	ja[offset1 +  2] = BASE +           IDX(i    , j    );
	ja[offset1 +  3] = BASE +           IDX(i    , j + 1);
	ja[offset1 +  4] = BASE +           IDX(i + 1, j    );

	ja[offset1 +  5] = BASE +     dim + IDX(i - 1, j    );
	ja[offset1 +  6] = BASE +     dim + IDX(i    , j - 1);
	ja[offset1 +  7] = BASE +     dim + IDX(i    , j    );
	ja[offset1 +  8] = BASE +     dim + IDX(i    , j + 1);
	ja[offset1 +  9] = BASE +     dim + IDX(i + 1, j    );

	ja[offset1 + 10] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset1 + 11] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset1 + 12] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset1 + 13] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset1 + 14] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset1 + 15] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset1 + 16] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset1 + 17] = BASE + 5 * dim;


	// Beta: grid number 1.
	ia[dim + IDX(i, j)] = BASE + offset2;

	// Values.
	aa[offset2 +  0] = dzodr*((D10)*(-dRu2));
	aa[offset2 +  1] = drodz*((D10)*(-dZu2));
	aa[offset2 +  2] = -aa[offset2 +  1];
	aa[offset2 +  3] = -aa[offset2 +  0];
	
	aa[offset2 +  4] = dzodr*((D20) + (D10)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 +  5] = drodz*((D20) + (D10)*(-dZu1 + 3.0*dZu3));
	aa[offset2 +  6] = (D21)*(drodz + dzodr) - 16.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset2 +  7] = 2.0*(D20)*drodz - aa[offset2 +  5];
	aa[offset2 +  8] = 2.0*(D20)*dzodr - aa[offset2 +  4];

	aa[offset2 +  9] = dzodr*((D10)*(3.0*dRu2));
	aa[offset2 + 10] = drodz*((D10)*(3.0*dZu2));
	aa[offset2 + 11] = 32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;
	aa[offset2 + 12] = -aa[offset2 + 10];
	aa[offset2 + 13] = -aa[offset2 +  9];

	aa[offset2 + 14] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;

	aa[offset2 + 15] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;

	aa[offset2 + 16] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*l*phi2or2/h2 - 32.0*M_PI*dr2*dzodr*(a2/h2)*l*wplOmega*phi2or2*(rr*w/chi));

	// Columns.
	ja[offset2 +  0] = BASE +           IDX(i - 1, j    );
	ja[offset2 +  1] = BASE +           IDX(i    , j - 1);
	ja[offset2 +  2] = BASE +           IDX(i    , j + 1);
	ja[offset2 +  3] = BASE +           IDX(i + 1, j    );

	ja[offset2 +  4] = BASE +     dim + IDX(i - 1, j    );
	ja[offset2 +  5] = BASE +     dim + IDX(i    , j - 1);
	ja[offset2 +  6] = BASE +     dim + IDX(i    , j    );
	ja[offset2 +  7] = BASE +     dim + IDX(i    , j + 1);
	ja[offset2 +  8] = BASE +     dim + IDX(i + 1, j    );

	ja[offset2 +  9] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset2 + 10] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset2 + 11] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset2 + 12] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset2 + 13] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset2 + 14] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset2 + 15] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset2 + 16] = BASE + 5 * dim;


	// H: grid number 2.
	ia[2 * dim + IDX(i, j)] = BASE + offset3;

	// Values.
	aa[offset3 +  0] = dzodr*((D10)*(1.0/ri + dRu3));
	aa[offset3 +  1] = drodz*((D10)*dZu3);
	aa[offset3 +  2] = -r2h2oalpha2dOmega2;
	aa[offset3 +  3] = -aa[offset3 +  1];
	aa[offset3 +  4] = -aa[offset3 +  0];

	aa[offset3 +  5] = dzodr*((D10)*(r2*h2*dRu2/alpha2));
	aa[offset3 +  6] = drodz*((D10)*(r2*h2*dZu2/alpha2));
	aa[offset3 +  7] = -aa[offset3 +  6];
	aa[offset3 +  8] = -aa[offset3 +  5];

	aa[offset3 +  9] = dzodr*((D20) + (D10)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 10] = drodz*((D20) + (D10)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 11] = (D21)*(drodz + dzodr) + r2h2oalpha2dOmega2 - 16.0*M_PI*dr2*dzodr*a2*l*l*phi2or2/h2;
	aa[offset3 + 12] = 2.0*(D20)*drodz - aa[offset3 + 10];
	aa[offset3 + 13] = 2.0*(D20)*dzodr - aa[offset3 +  9];

	aa[offset3 + 14] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;

	aa[offset3 + 15] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;

	aa[offset3 + 16] = dw_du(xi, m) * (8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2*(rr*w/chi));

	// Columns.
	ja[offset3 +  0] = BASE +           IDX(i - 1, j    );
	ja[offset3 +  1] = BASE +           IDX(i    , j - 1);
	ja[offset3 +  2] = BASE +           IDX(i    , j    );
	ja[offset3 +  3] = BASE +           IDX(i    , j + 1);
	ja[offset3 +  4] = BASE +           IDX(i + 1, j    );

	ja[offset3 +  5] = BASE +     dim + IDX(i - 1, j    );
	ja[offset3 +  6] = BASE +     dim + IDX(i    , j - 1);
	ja[offset3 +  7] = BASE +     dim + IDX(i    , j + 1);
	ja[offset3 +  8] = BASE +     dim + IDX(i + 1, j    );

	ja[offset3 +  9] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset3 + 10] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset3 + 11] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset3 + 12] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset3 + 13] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset3 + 14] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset3 + 15] = BASE + 4 * dim + IDX(i    , j    );

	ja[offset3 + 16] = BASE + 5 * dim;


	// A: grid number 3.
	ia[3 * dim + IDX(i, j)] = BASE + offset4;

	// Values.
	aa[offset4 +  0] = dzodr*((D10)*(-1.0/ri - dRu3));
	aa[offset4 +  1] = drodz*((D10)*(-dZu3));
	aa[offset4 +  2] = 0.5*r2h2oalpha2dOmega2 - 8.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset4 +  3] = -aa[offset4 +  1];
	aa[offset4 +  4] = -aa[offset4 +  0];

	aa[offset4 +  5] = dzodr*((D10)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 +  6] = drodz*((D10)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 +  7] = 8.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset4 +  8] = -aa[offset4 +  6];
	aa[offset4 +  9] = -aa[offset4 +  5];

	aa[offset4 + 10] = dzodr*((D10)*(-dRu1));
	aa[offset4 + 11] = drodz*((D10)*(-dZu1));
	aa[offset4 + 12] = -0.5*r2h2oalpha2dOmega2 + 8.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset4 + 13] = -aa[offset4 + 11];
	aa[offset4 + 14] = -aa[offset4 + 10];

	aa[offset4 + 15] = (D20)*dzodr; // CONSTANT!
	aa[offset4 + 16] = (D20)*drodz; // CONSTANT!
	aa[offset4 + 17] = (D21)*(drodz + dzodr) + 8.0*M_PI*dr2*dzodr*(-l*l/h2 + r2*wplOmega2/alpha2)*a2*phi2or2;
	aa[offset4 + 18] = (D20)*drodz; // CONSTANT!
	aa[offset4 + 19] = (D20)*dzodr; // CONSTANT!

	aa[offset4 + 20] = dzodr*((D10)*(8.0*M_PI*phi2*(dRu5 - (chi*dr2*ri)/rr + l/ri)));
	aa[offset4 + 21] = drodz*((D10)*(8.0*M_PI*phi2*(dZu5 - (chi*dz2*zi)/rr)));
	aa[offset4 + 22] = 8.0*M_PI*phi2or2*(dr2*dzodr*(l*l + 2.0*l*ri*dRu5 + a2*(-l*l/h2 + r2*wplOmega2/alpha2) + r2*(chi*(chi - 2.0*dXu5))) + (r2*(dzodr*dRu5*dRu5 + drodz*dZu5*dZu5)));
	aa[offset4 + 23] = -aa[offset4 + 21];
	aa[offset4 + 24] = -aa[offset4 + 20];

	aa[offset4 + 25] = dw_du(xi, m) * (8.0*M_PI*phi2or2*(dr2*dzodr*(a2*r2*wplOmega/alpha2 + r2*w*(dXu5/chi - 1.0) + (rr*w/chi)*(l*l + 2.0*l*ri*dRu5 + a2*(-l*l/h2 + r2*wplOmega2/alpha2) + r2*(chi*(chi - 2.0*dXu5)))) + (rr*w/chi)*(r2*(dzodr*dRu5*dRu5 + drodz*dZu5*dZu5))));

	// Columns.
	ja[offset4 +  0] = BASE +           IDX(i - 1, j    );
	ja[offset4 +  1] = BASE +           IDX(i    , j - 1);
	ja[offset4 +  2] = BASE +           IDX(i    , j    );
	ja[offset4 +  3] = BASE +           IDX(i    , j + 1);
	ja[offset4 +  4] = BASE +           IDX(i + 1, j    );

	ja[offset4 +  5] = BASE +     dim + IDX(i - 1, j    );
	ja[offset4 +  6] = BASE +     dim + IDX(i    , j - 1);
	ja[offset4 +  7] = BASE +     dim + IDX(i    , j    );
	ja[offset4 +  8] = BASE +     dim + IDX(i    , j + 1);
	ja[offset4 +  9] = BASE +     dim + IDX(i + 1, j    );

	ja[offset4 + 10] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset4 + 11] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset4 + 12] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset4 + 13] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset4 + 14] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset4 + 15] = BASE + 3 * dim + IDX(i - 1, j    );
	ja[offset4 + 16] = BASE + 3 * dim + IDX(i    , j - 1);
	ja[offset4 + 17] = BASE + 3 * dim + IDX(i    , j    );
	ja[offset4 + 18] = BASE + 3 * dim + IDX(i    , j + 1);
	ja[offset4 + 19] = BASE + 3 * dim + IDX(i + 1, j    );

	ja[offset4 + 20] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset4 + 21] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset4 + 22] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset4 + 23] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset4 + 24] = BASE + 4 * dim + IDX(i + 1, j    );

	ja[offset4 + 25] = BASE + 5 * dim;


	// Psi: grid number 4.
	ia[4 * dim + IDX(i, j)] = BASE + offset5;

	// Values.
	aa[offset5 +  0] = dzodr*((D10)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  1] = drodz*((D10)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  2] = -2.0*dr2*dzodr*a2*wplOmega2/alpha2;
	aa[offset5 +  3] = -aa[offset5 +  1];
	aa[offset5 +  4] = -aa[offset5 +  0];

	aa[offset5 +  5] = 2.0*dr2*dzodr*a2*l*wplOmega/alpha2;

	aa[offset5 +  6] = dzodr*((D10)*(dRu5 + l/ri - chi*dr2*ri/rr));
	aa[offset5 +  7] = drodz*((D10)*(dZu5 - chi*dz2*zi/rr));
	aa[offset5 +  8] = 2.0*l*l*dzodr*(a2/h2)*(1.0/(ri*ri));
	aa[offset5 +  9] = -aa[offset5 +  7];
	aa[offset5 + 10] = -aa[offset5 +  6];

	aa[offset5 + 11] = 2.0*dzodr*(dr2*a2*(wplOmega2/alpha2 - m2) - l*l*(a2/h2)*(1.0/(ri*ri)));

	aa[offset5 + 12] = (D20)*dzodr + dzodr*((D10)*((2.0*l + 1.0)/ri + 2.0*dRu5 + dRu1 + dRu3 - 2.0*chi*dr2*ri/rr));
	aa[offset5 + 13] = (D20)*drodz + drodz*((D10)*(2.0*dZu5 + dZu1 + dZu3 - 2.0*chi*dz2*zi/rr));
	aa[offset5 + 14] = (D21)*(drodz + dzodr); // CONSTANT!
	aa[offset5 + 15] = 2.0*(D20)*drodz - aa[offset5 + 13];
	aa[offset5 + 16] = 2.0*(D20)*dzodr - aa[offset5 + 12];

	aa[offset5 + 17] = dw_du(xi, m) * (dr2*dzodr*(2.0*a2*wplOmega/alpha2 + (-w/chi)*(2.0*chi - 2.0*(l + 1.0)/rr - 2.0*dXu5 - dXu1 - dXu3)));

	// Columns.
	ja[offset5 +  0] = BASE +           IDX(i - 1, j    );
	ja[offset5 +  1] = BASE +           IDX(i    , j - 1);
	ja[offset5 +  2] = BASE +           IDX(i    , j    );
	ja[offset5 +  3] = BASE +           IDX(i    , j + 1);
	ja[offset5 +  4] = BASE +           IDX(i + 1, j    );

	ja[offset5 +  5] = BASE +     dim + IDX(i    , j    );

	ja[offset5 +  6] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset5 +  7] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset5 +  8] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset5 +  9] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset5 + 10] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset5 + 11] = BASE + 3 * dim + IDX(i    , j    );

	ja[offset5 + 12] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset5 + 13] = BASE + 4 * dim + IDX(i    , j - 1);
	ja[offset5 + 14] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset5 + 15] = BASE + 4 * dim + IDX(i    , j + 1);
	ja[offset5 + 16] = BASE + 4 * dim + IDX(i + 1, j    );

	ja[offset5 + 17] = BASE + 5 * dim;


	// All done.
	return;
}

/* OLD DEPRECATED SUBROUTINES
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
*/