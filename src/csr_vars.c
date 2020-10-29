// Include headers.
#include "tools.h"
#include "omega_calc.h"

// Finite difference coefficients for 2nd order.
#define D_2_10 (-0.5)
#define D_2_11 (+0.0)
#define D_2_12 (+0.5)

#define D_2_20 (+1.0)
#define D_2_21 (-2.0)
#define D_2_22 (+1.0)

#define Q1 1.0
#define Q2 1.0

#define GNUM 6



// Jacobian for centered-centered 2nd order stencil and variable omega.
void jacobian_2nd_order_variable_omega_cc
(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings. 
	MKL_INT *ja,		// CSR array for columns.
	const MKL_INT NrTotal, 	// Grid total dimension in r.
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost,	// Number of ghost zones.
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
	const double u601, const double u610, const double u611, const double u612, const double u621,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5,	// Number of elements filled before filling function 5.
	const MKL_INT offset6	// Number of elements filled before filling function 5.
)
{
	// Physical variables.
	double alpha = exp(u111);
	double Omega = u211;
	double h    = exp(u311);
	double a    = exp(u411);
	double psi  = u511;
	double lambda = u611;

	// Coordinates.
	double ri = (double)i + 0.5 - ghost;
	//double zi = (double)j + 0.5 - ghost;
	double r = ri * dr;
	//double z = zi * dz;
	double r2 = r * r;
	//double z2 = z * z;
	//double rr = sqrt(r2 + z2);

	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	double dr2 = dr * dr;
	//double dz2 = dz * dz;

	// Scalar field mass and frequency.
	double w = omega_calc(xi, m);
	double m2 = m * m;
	//double w2 = w * w;
	//double chi = sqrt(m2 - w2);

	// Scalar field.
	double rlm1 = (l == 1) ? 1.0 : pow(r, l - 1);
	double rl = rlm1 * r;
	double phior = rlm1 * psi;
	double phi = r * phior;
	double phi2or2 = phior * phior;
	double phi2 = phi * phi;

	/// Shift combined with scalar field rotation.
	double wplOmega = w + l * Omega;
	double wplOmega2 = wplOmega * wplOmega;

	// Finite differences.
	double dRu1 = D_2_10 * u101 + D_2_12 * u121;
	double dRu2 = D_2_10 * u201 + D_2_12 * u221;
	double dRu3 = D_2_10 * u301 + D_2_12 * u321;
	//double dRu4 = D_2_10 * u401 + D_2_12 * u421;
	double dRu5 = D_2_10 * u501 + D_2_12 * u521;
	double dRu6 = D_2_10 * u601 + D_2_12 * u621;

	double dZu1 = D_2_10 * u110 + D_2_12 * u112;
	double dZu2 = D_2_10 * u210 + D_2_12 * u212;
	double dZu3 = D_2_10 * u310 + D_2_12 * u312;
	//double dZu4 = D_2_10 * u410 + D_2_12 * u412;
	double dZu5 = D_2_10 * u510 + D_2_12 * u512;
	double dZu6 = D_2_10 * u610 + D_2_12 * u612;

	// Second derivatives.
	double dRRu1 = D_2_20 * u101 + D_2_21 * u111 + D_2_22 * u121;
	double dRRu3 = D_2_20 * u301 + D_2_21 * u311 + D_2_22 * u321;

	// Radial derivatives.
	//double dXu5 = (ri * dRu5 + zi * dZu5) / rr;
	//double dXu1 = (ri * dRu1 + zi * dZu1) / rr;
	//double dXu3 = (ri * dRu3 + zi * dZu3) / rr;

	// Squared variables.
	double alpha2 = alpha * alpha;
	double h2 = h * h;
	double a2 = a * a;

	// Common term.
	double r2h2oalpha2dOmega2 = r2 * h2 * (dzodr * dRu2 * dRu2 + drodz * dZu2 * dZu2) / alpha2;

	// Alpha: grid number 0.
	ia[IDX(i, j)] = BASE + offset1;

	// Set values.
	aa[offset1     ] = dzodr*((D_2_20) + (D_2_10)*(1.0/ri + 2.0*dRu1 + dRu3));
	aa[offset1 +  1] = drodz*((D_2_20) + (D_2_10)*(2.0*dZu1 + dZu3));
	aa[offset1 +  2] = (D_2_21)*(dzodr + drodz) + r2h2oalpha2dOmega2 + 16.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset1 +  3] = drodz*2.0*(D_2_22) - aa[offset1 + 1];
	aa[offset1 +  4] = dzodr*2.0*(D_2_22) - aa[offset1    ];

	aa[offset1 +  5] = dzodr*((D_2_10)*(-r2*h2*dRu2/alpha2));
	aa[offset1 +  6] = drodz*((D_2_10)*(-r2*h2*dZu2/alpha2));
	aa[offset1 +  7] = -16.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset1 +  8] = -aa[offset1 + 6];
	aa[offset1 +  9] = -aa[offset1 + 5];

	aa[offset1 + 10] = dzodr*((D_2_10)*dRu1);
	aa[offset1 + 11] = drodz*((D_2_10)*dZu1);
	aa[offset1 + 12] = -r2h2oalpha2dOmega2;
	aa[offset1 + 13] = -aa[offset1 + 11];
	aa[offset1 + 14] = -aa[offset1 + 10];

	aa[offset1 + 15] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;

	aa[offset1 + 16] = 8.0*M_PI*dr2*dzodr*a2*(m2 - 2.0*wplOmega2/alpha2)*rl*phi;

	aa[offset1 + 17] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*wplOmega*phi2/alpha2);

	// Set column indices.
	ja[offset1     ] = BASE +           IDX(i - 1, j    );
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

	ja[offset1 + 17] = BASE + GNUM * dim;


	// Row start at offset. This is grid 2.
	ia[dim + IDX(i, j)] = BASE + offset2;

	// Set values.
	aa[offset2     ] = dzodr*((D_2_10)*(-dRu2));
	aa[offset2 +  1] = drodz*((D_2_10)*(-dZu2));
	aa[offset2 +  2] = -aa[offset2 + 1];
	aa[offset2 +  3] = -aa[offset2    ];

	aa[offset2 +  4] = dzodr*((D_2_20) + (D_2_10)*(3.0/ri - dRu1 + 3.0*dRu3));
	aa[offset2 +  5] = drodz*((D_2_20) + (D_2_10)*(-dZu1 + 3.0*dZu3));
	aa[offset2 +  6] = (D_2_21)*(drodz + dzodr) - 16.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset2 +  7] = drodz*2.0*(D_2_22) -aa[offset2 + 5];
	aa[offset2 +  8] = dzodr*2.0*(D_2_22) -aa[offset2 + 4];

	aa[offset2 +  9] = dzodr*((D_2_10)*(3.0*dRu2));
	aa[offset2 + 10] = drodz*((D_2_10)*(3.0*dZu2));
	aa[offset2 + 11] = 32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;
	aa[offset2 + 12] = -aa[offset2 + 10];
	aa[offset2 + 13] = -aa[offset2 +  9];

	aa[offset2 + 14] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phi2or2/h2;

	aa[offset2 + 15] = -32.0*M_PI*dr2*dzodr*a2*l*wplOmega*phior*rlm1/h2;

	aa[offset2 + 16] = dw_du(xi, m) * (-16.0*M_PI*dr2*dzodr*a2*l*phi2or2/h2);

	// Set column indices.
	ja[offset2     ] = BASE +           IDX(i - 1, j    );
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

	ja[offset2 + 16] = BASE + GNUM * dim;


	// Row start at offset. This is grid 3.
	ia[2 * dim + IDX(i, j)] = BASE + offset3;

	// Set values.
	aa[offset3     ] = dzodr*((D_2_10)*(1.0/ri + dRu3));
	aa[offset3 +  1] = drodz*((D_2_10)*dZu3);
	aa[offset3 +  2] = -r2h2oalpha2dOmega2;
	aa[offset3 +  3] = -aa[offset3 + 1];
	aa[offset3 +  4] = -aa[offset3    ];

	aa[offset3 +  5] = dzodr*((D_2_10)*(r2*h2*dRu2/alpha2));
	aa[offset3 +  6] = drodz*((D_2_10)*(r2*h2*dZu2/alpha2));
	aa[offset3 +  7] = -aa[offset3 + 6];
	aa[offset3 +  8] = -aa[offset3 + 5];

	aa[offset3 +  9] = dzodr*((D_2_20) + (D_2_10)*(2.0/ri + dRu1 + 2.0*dRu3));
	aa[offset3 + 10] = drodz*((D_2_20) + (D_2_10)*(dZu1 + 2.0*dZu3));
	aa[offset3 + 11] = (D_2_21)*(drodz + dzodr) + r2h2oalpha2dOmega2 - 16.0*M_PI*dr2*dzodr*a2*l*l*phi2or2/h2;
	aa[offset3 + 12] = drodz*2.0*(D_2_22) -aa[offset3 + 10];
	aa[offset3 + 13] = dzodr*2.0*(D_2_22) -aa[offset3 +  9];

	aa[offset3 + 14] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;

	aa[offset3 + 15] = 8.0*M_PI*dr2*dzodr*a2*(r2*m2 + 2.0*l*l/h2)*phior*rlm1;

	// Set column indices.
	ja[offset3     ] = BASE +           IDX(i - 1, j    );
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


	// Row start at offset. This is grid 4.
	ia[3 * dim + IDX(i, j)] = BASE + offset4;

	// Set values.
	aa[offset4     ] = dzodr*((D_2_10)*(-1.0/ri - dRu3));
	aa[offset4 +  1] = drodz*((D_2_10)*(-dZu3));
	aa[offset4 +  2] = 0.5*r2h2oalpha2dOmega2 - 8.0*M_PI*dr2*dzodr*a2*wplOmega2*phi2/alpha2;
	aa[offset4 +  3] = -aa[offset4 + 1];
	aa[offset4 +  4] = -aa[offset4    ];

	aa[offset4 +  5] = dzodr*((D_2_10)*(-0.5*r2*h2*dRu2/alpha2));
	aa[offset4 +  6] = drodz*((D_2_10)*(-0.5*r2*h2*dZu2/alpha2));
	aa[offset4 +  7] = 8.0*M_PI*dr2*dzodr*l*a2*wplOmega*phi2/alpha2;
	aa[offset4 +  8] = -aa[offset4 + 6];
	aa[offset4 +  9] = -aa[offset4 + 5];

	aa[offset4 + 10] = dzodr*((D_2_10)*(-dRu1));
	aa[offset4 + 11] = drodz*((D_2_10)*(-dZu1));
	aa[offset4 + 12] = -0.5*r2h2oalpha2dOmega2 + 8.0*M_PI*dr2*dzodr*l*l*a2*phi2or2/h2;
	aa[offset4 + 13] = -aa[offset4 + 11];
	aa[offset4 + 14] = -aa[offset4 + 10];

	aa[offset4 + 15] = dzodr*(D_2_20);// CONSTANT!
	aa[offset4 + 16] = drodz*(D_2_20);// CONSTANT!
	aa[offset4 + 17] = (D_2_21)*(drodz + dzodr) + 8.0*M_PI*dr2*dzodr*(-l*l/h2 + r2*wplOmega2/alpha2)*a2*phi2or2;
	aa[offset4 + 18] = drodz*(D_2_22);// CONSTANT!
	aa[offset4 + 19] = dzodr*(D_2_22);// CONSTANT!

	aa[offset4 + 20] = dzodr*(D_2_10)*(8.0*M_PI*rl*rl*(l*psi/ri + dRu5));
	aa[offset4 + 21] = drodz*(D_2_10)*(8.0*M_PI*rl*rl*dZu5);
	aa[offset4 + 22] = 8.0*M_PI*dr2*dzodr*rlm1*rlm1*(l*ri*dRu5 + (a2*wplOmega2*r2/alpha2 + l*l*(1.0 - a2/h2))*psi);
	aa[offset4 + 23] = -aa[offset4 + 21];
	aa[offset4 + 24] = -aa[offset4 + 20];

	aa[offset4 + 25] = dw_du(xi, m) * (8.0*M_PI*dr2*dzodr*a2*wplOmega*phi2/alpha2);

	// Set column indices.
	ja[offset4     ] = BASE +           IDX(i - 1, j    );
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

	ja[offset4 + 25] = BASE + GNUM * dim;


	// Row start at offset. This is grid 5.
	ia[4 * dim + IDX(i, j)] = BASE + offset5;

	// Set values.
	aa[offset5     ] = dzodr*((D_2_10)*(l*psi/ri + dRu5));
	aa[offset5 +  1] = drodz*((D_2_10)*dZu5);
	aa[offset5 +  2] = -2.0*dr2*dzodr*a2*wplOmega2*psi/alpha2;
	aa[offset5 +  3] = -aa[offset5 + 1];
	aa[offset5 +  4] = -aa[offset5    ];

	aa[offset5 +  5] = 2.0*dr2*dzodr*l*a2*wplOmega*psi/alpha2;

	aa[offset5 +  6] = dzodr*((D_2_10)*(l*psi/ri + dRu5));
	aa[offset5 +  7] = drodz*((D_2_10)*dZu5);
	aa[offset5 +  8] = dzodr*dr2*2.0*l*l*lambda*psi/h2;
	//aa[offset5 +  8] = 2.0*dzodr*l*l*(a2/h2)*psi/(ri*ri);
	aa[offset5 +  9] = -aa[offset5 + 7];
	aa[offset5 + 10] = -aa[offset5 + 6];

	aa[offset5 + 11] = 2.0*dzodr*(dr2*a2*(wplOmega2/alpha2 - m2))*psi;
	//aa[offset5 + 11] = 2.0*dzodr*(dr2*a2*(wplOmega2/alpha2 - m2) - l*l*(a2/h2)/(ri*ri))*psi;

	aa[offset5 + 12] = dzodr*((D_2_20) + (D_2_10)*((2.0*l + 1.0)/ri + dRu1 + dRu3));
	aa[offset5 + 13] = drodz*((D_2_20) + (D_2_10)*(dZu1 + dZu3));
	aa[offset5 + 14] = (D_2_21)*(drodz + dzodr) + dzodr * (l * (dRu1 / ri + dRu3 / ri) + (dr2 * a2 * (wplOmega2 / alpha2 - m2) - l * l * (dr2*lambda) / h2));
	aa[offset5 + 15] = drodz*2.0*(D_2_22) -aa[offset5 + 13];
	aa[offset5 + 16] = dzodr*2.0*(D_2_22) -aa[offset5 + 12];

	aa[offset5 + 17] = dzodr*dr2*(-l*l/h2)*psi;

	aa[offset5 + 18] = dw_du(xi, m) * (2.0*dr2*dzodr*a2*wplOmega*psi/alpha2);

	// Set column indices.
	ja[offset5     ] = BASE +           IDX(i - 1, j    );
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
	
	ja[offset5 + 17] = BASE + 5 * dim + IDX(i    , j    );

	ja[offset5 + 18] = BASE + GNUM * dim;


	// Row start at offset. This is grid 6.
	ia[5 * dim + IDX(i, j)] = BASE + offset6;
	// a2 calculation changes.
	a2 = h2 + r2 * lambda;

	// Set values.
	aa[offset6 +  0] = D_2_20*dzodr*(2*lambda + (2*h2*Q1)/((r2))) + D_2_10*dzodr*(-dRu6 + 4*dRu1*lambda + (Q1*((-2*h2)/(ri*ri*ri) + (4*dRu1*h2)/(ri*ri)) - (4*dRu3*h2)/(ri*ri))/((dr2)) - (2*lambda)/ri);
	aa[offset6 +  1] = (D_2_10*dZu6)/dzodr;
	aa[offset6 +  2] = (2*(dZu2*dZu2)*(h2*h2))/(alpha2*dzodr) + D_2_21*dzodr*(2*lambda + (2*h2*Q1)/((r2))) + dzodr*((4*(dRu2*dRu2)*(h2*h2))/(alpha2) + (2*((dr2))*(dRu2*dRu2)*h2*lambda*(ri*ri))/(alpha2));
	aa[offset6 +  3] = (D_2_12*dZu6)/dzodr;
	aa[offset6 +  4] = D_2_22*dzodr*(2*lambda + (2*h2*Q1)/((r2))) + D_2_12*dzodr*(-dRu6 + 4*dRu1*lambda + (Q1*((-2*h2)/(ri*ri*ri) + (4*dRu1*h2)/(ri*ri)) - (4*dRu3*h2)/(ri*ri))/((dr2)) - (2*lambda)/ri);

	aa[offset6 +  5] = D_2_10*dzodr*((-4*dRu2*(h2*h2))/(alpha2) - (2*((dr2))*dRu2*h2*lambda*(ri*ri))/(alpha2));
	aa[offset6 +  6] = (-2*D_2_10*dZu2*(h2*h2))/(alpha2*dzodr);
	aa[offset6 +  7] = (-2*D_2_12*dZu2*(h2*h2))/(alpha2*dzodr);
	aa[offset6 +  8] = D_2_12*dzodr*((-4*dRu2*(h2*h2))/(alpha2) - (2*((dr2))*dRu2*h2*lambda*(ri*ri))/(alpha2));

	aa[offset6 +  9] = D_2_20*dzodr*(2*lambda + (2*h2*Q2)/((r2))) + D_2_10*((dr2))*dzodr*(-(dRu6/((dr2))) + (4*dRu3*lambda)/((dr2)) - (4*dRu1*h2)/((dr2)*(r2)) + (Q2*((8*dRu3*h2)/((dr2)) - (2*h2)/((dr2)*ri)))/((r2)) - (4*dRu6*h2)/((dr2)*(h2 + (dr2)*lambda*(ri*ri))) + (2*h2*lambda*((1/h2) - 4/(h2 + (dr2)*lambda*(ri*ri))))/((dr2)*ri) - (8*dRu3*(h2*h2)*(1/(2.*h2) + (1/a2)))/((dr2)*(r2)));
	aa[offset6 + 10] = D_2_10*((dr2))*dzodr*(dZu6/((dr2)*(dzodr*dzodr)) - (4*dZu6*h2)/((dr2)*(dzodr*dzodr)*(h2 + (dr2)*lambda*(ri*ri))) + (8*dZu3*h2*lambda)/((dr2)*(dzodr*dzodr)*(h2 + (dr2)*lambda*(ri*ri))));
	aa[offset6 + 11] = D_2_21*dzodr*(2*lambda + (2*h2*Q2)/((r2))) + (dr2)*dzodr*((-2*(dRu2*dRu2)*(h2*h2))/(alpha2*((dr2))) - (4*((dRu2*dRu2)/((dr2)) + (dZu2*dZu2)/((dr2)*(dzodr*dzodr)))*(h2*h2))/(alpha2) - ((-2*dRu3*dRu6*h2)/((dr2)) + (2*dZu3*dZu6*h2)/((dr2)*(dzodr*dzodr)))/h2 + ((-4*dRu3*dRu6*h2)/((dr2)) + (4*dZu3*dZu6*h2)/((dr2)*(dzodr*dzodr)))/(2.*h2) - (2*((2*dRRu3*h2)/((dr2)) + (4*(dRu3*dRu3)*h2)/((dr2)))*lambda)/h2 + (((4*dRRu3*h2)/((dr2)) + (8*(dRu3*dRu3)*h2)/((dr2)))*lambda)/h2 - (8*dRu1*dRu3*h2)/((dr2)*(r2)) + (4*h2*Q1*((alpha*dRRu1)/((dr2)) + (alpha*(dRu1*dRu1))/((dr2)) - (alpha*dRu1)/((dr2)*ri)))/(alpha*((dr2))*(ri*ri)) + (Q2*((4*dRRu3*h2)/((dr2)) + (8*(dRu3*dRu3)*h2)/((dr2)) - (4*dRu3*h2)/((dr2)*ri)))/((r2)) + (4*h2*((2*dRu3*dRu6*h2)/((dr2)) + (2*dZu3*dZu6*h2)/((dr2)*(dzodr*dzodr))))/	((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri))) - (8*(dZu3*dZu3)*(h2*h2)*lambda)/((dr2)*(dzodr*dzodr)*((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri)))) + (8*h2*(lambda*lambda))/((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri))) + (8*dRu6*h2*lambda*ri)/((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri))) + (2*((dr2))*((dRu6*dRu6)/((dr2)) + (dZu6*dZu6)/((dr2)*(dzodr*dzodr)))*h2*(ri*ri))/	((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri))) - (2*((4*dRu3*dRu6*h2)/((dr2)) + (4*dZu3*dZu6*h2)/((dr2)*(dzodr*dzodr))))/(h2 + (dr2)*lambda*(ri*ri)) + (8*(dZu3*dZu3)*h2*lambda)/((dr2)*(dzodr*dzodr)*(h2 + (dr2)*lambda*(ri*ri))) - (2*(dRu2*dRu2)*h2*(h2 + (dr2)*lambda*(ri*ri)))/(alpha2*((dr2))) - (4*(dRu3*dRu3)*(h2*h2)*(-(1/h2) - (2*h2)/((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri)))))/	((dr2)*(r2)) + (2*dRu3*h2*lambda*	(-2/h2 + (8*h2)/((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri)))))/((dr2)*ri) + (4*dRu3*h2*lambda*((1/h2) - 4/(h2 + (dr2)*lambda*(ri*ri))))/((dr2)*ri) - (16*(dRu3*dRu3)*(h2*h2)*(1/(2.*h2) + (1/a2)))/((dr2)*(r2)) + 16*h2*M_PI*(rlm1*rlm1)*(-(h2*((m2))*(psi*psi)) + (2*dRu5*(2*l*psi + dRu5*ri))/((dr2)*ri) + (m2)*(psi*psi)*(h2 + (dr2)*lambda*(ri*ri))));
	aa[offset6 + 12] = D_2_12*((dr2))*dzodr*(dZu6/((dr2)*(dzodr*dzodr)) - (4*dZu6*h2)/((dr2)*(dzodr*dzodr)*(h2 + (dr2)*lambda*(ri*ri))) + (8*dZu3*h2*lambda)/((dr2)*(dzodr*dzodr)*(h2 + (dr2)*lambda*(ri*ri))));
	aa[offset6 + 13] = D_2_22*dzodr*(2*lambda + (2*h2*Q2)/((r2))) + D_2_12*((dr2))*dzodr*(-(dRu6/((dr2))) + (4*dRu3*lambda)/((dr2)) - (4*dRu1*h2)/((dr2)*(r2)) + (Q2*((8*dRu3*h2)/((dr2)) - (2*h2)/((dr2)*ri)))/((r2)) - (4*dRu6*h2)/((dr2)*(h2 + (dr2)*lambda*(ri*ri))) + (2*h2*lambda*((1/h2) - 4/(h2 + (dr2)*lambda*(ri*ri))))/((dr2)*ri) - (8*dRu3*(h2*h2)*(1/(2.*h2) + (1/a2)))/((dr2)*(r2)));

	aa[offset6 + 14] = D_2_10*dzodr*((8*M_PI*(4*dRu5*h2 + (4*h2*l*psi)/ri)*(rl*rl))/((r2)) + (8*M_PI*(rl*rl)*(4*l*lambda*psi*ri + 4*dRu5*lambda*(ri*ri)))/(ri*ri));
	aa[offset6 + 15] = dzodr*(16*((dr2))*h2*lambda*((m2))*M_PI*psi*(rl*rl) + (32*dRu5*h2*l*M_PI*(rl*rl))/((dr2)*(ri*ri*ri)) + (32*dRu5*l*lambda*M_PI*(rl*rl))/ri + 16*((dr2)*(dr2))*(lambda*lambda)*((m2))*M_PI*psi*(ri*ri)*(rl*rl));
	aa[offset6 + 16] = D_2_12*dzodr*((8*M_PI*(4*dRu5*h2 + (4*h2*l*psi)/ri)*(rl*rl))/((r2)) + (8*M_PI*(rl*rl)*(4*l*lambda*psi*ri + 4*dRu5*lambda*(ri*ri)))/(ri*ri));

	aa[offset6 + 17] = D_2_20*dzodr + D_2_10*((dr2))*dzodr*(-(dRu1/((dr2))) - dRu3/((dr2)) + 3/((dr2)*ri) - (4*dRu3*h2)/((dr2)*(h2 + (dr2)*lambda*(ri*ri))) - (4*lambda*ri)/(h2 + (dr2)*lambda*(ri*ri)) - (2*dRu6*(ri*ri))/(h2 + (dr2)*lambda*(ri*ri)));
	aa[offset6 + 18] = D_2_20/dzodr + D_2_10*((dr2))*dzodr*(dZu1/((dr2)*(dzodr*dzodr)) + dZu3/((dr2)*(dzodr*dzodr)) - (4*dZu3*h2)/((dr2)*(dzodr*dzodr)*(h2 + (dr2)*lambda*(ri*ri))) - (2*dZu6*(ri*ri))/(dzodr*dzodr*(h2 + (dr2)*lambda*(ri*ri))));
	aa[offset6 + 19] = D_2_21/dzodr + D_2_21*dzodr + (dr2)*dzodr*((2*((alpha*dRRu1)/((dr2)) + (alpha*(dRu1*dRu1))/((dr2))))/alpha - (2*(dRu3*dRu3))/((dr2)) + ((2*dRRu3*h2)/((dr2)) + (4*(dRu3*dRu3)*h2)/((dr2)))/h2 - (2*dRu1)/((dr2)*ri) - (dRu2*dRu2*h2*(ri*ri))/(alpha2) + (4*(dRu3*dRu3)*(h2*h2))/	((dr2)*((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri)))) + (8*dRu3*h2*lambda*ri)/((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri))) + (2*((dr2))*((2*dRu3*dRu6*h2)/((dr2)) + (2*dZu3*dZu6*h2)/((dr2)*(dzodr*dzodr)))*(ri*ri))/	((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri))) - (4*(dZu3*dZu3)*h2*lambda*(ri*ri))/(dzodr*dzodr*((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri)))) + (4*((dr2))*(lambda*lambda)*(ri*ri))/((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri))) + (4*((dr2))*dRu6*lambda*(ri*ri*ri))/((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri))) + ((dr2)*(dr2)*((dRu6*dRu6)/((dr2)) + (dZu6*dZu6)/((dr2)*(dzodr*dzodr)))*(ri*ri*ri*ri))/	((h2 + (dr2)*lambda*(ri*ri))*(h2 + (dr2)*lambda*(ri*ri))) + (4*(dZu3*dZu3)*h2)/((dr2)*(dzodr*dzodr)*(h2 + (dr2)*lambda*(ri*ri))) - (8*lambda)/(h2 + (dr2)*lambda*(ri*ri)) - (4*dRu6*ri)/(h2 + (dr2)*lambda*(ri*ri)) + 8*((dr2))*((m2))*M_PI*(psi*psi)*(ri*ri)*(rlm1*rlm1)*	(h2 + (dr2)*lambda*(ri*ri)) + (2*dRu3*h2*((1/h2) - 4/(h2 + (dr2)*lambda*(ri*ri))))/((dr2)*ri) + 8*((dr2))*M_PI*(ri*ri)*(rlm1*rlm1)*(-(h2*((m2))*(psi*psi)) + (2*dRu5*(2*l*psi + dRu5*ri))/((dr2)*ri) + (m2)*(psi*psi)*(h2 + (dr2)*lambda*(ri*ri))));
	aa[offset6 + 20] = D_2_22/dzodr + D_2_12*((dr2))*dzodr*(dZu1/((dr2)*(dzodr*dzodr)) + dZu3/((dr2)*(dzodr*dzodr)) - (4*dZu3*h2)/((dr2)*(dzodr*dzodr)*(h2 + (dr2)*lambda*(ri*ri))) - (2*dZu6*(ri*ri))/(dzodr*dzodr*(h2 + (dr2)*lambda*(ri*ri))));
	aa[offset6 + 21] = D_2_22*dzodr + D_2_12*((dr2))*dzodr*(-(dRu1/((dr2))) - dRu3/((dr2)) + 3/((dr2)*ri) - (4*dRu3*h2)/((dr2)*(h2 + (dr2)*lambda*(ri*ri))) - (4*lambda*ri)/(h2 + (dr2)*lambda*(ri*ri)) - (2*dRu6*(ri*ri))/(h2 + (dr2)*lambda*(ri*ri)));

	// Columns.
	ja[offset6     ] = BASE +           IDX(i - 1, j    );
	ja[offset6 +  1] = BASE +           IDX(i    , j - 1);
	ja[offset6 +  2] = BASE +           IDX(i    , j    );
	ja[offset6 +  3] = BASE +           IDX(i    , j + 1);
	ja[offset6 +  4] = BASE +           IDX(i + 1, j    );

	ja[offset6 +  5] = BASE +     dim + IDX(i - 1, j    );
	ja[offset6 +  6] = BASE +     dim + IDX(i    , j - 1);
	ja[offset6 +  7] = BASE +     dim + IDX(i    , j + 1);
	ja[offset6 +  8] = BASE +     dim + IDX(i + 1, j    );

	ja[offset6 +  9] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset6 + 10] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset6 + 11] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset6 + 12] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset6 + 13] = BASE + 2 * dim + IDX(i + 1, j    );

	ja[offset6 + 14] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset6 + 15] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset6 + 16] = BASE + 4 * dim + IDX(i + 1, j    );

	ja[offset6 + 17] = BASE + 5 * dim + IDX(i - 1, j    );
	ja[offset6 + 18] = BASE + 5 * dim + IDX(i    , j - 1);
	ja[offset6 + 19] = BASE + 5 * dim + IDX(i    , j    );
	ja[offset6 + 20] = BASE + 5 * dim + IDX(i    , j + 1);
	ja[offset6 + 21] = BASE + 5 * dim + IDX(i + 1, j    );


	return;
}

// Finite difference coefficients for 4th order.
#define D10 (+1.0 / 12.0)
#define D11 (-2.0 / 3.0)
#define D12 (+0.0)
#define D13 (+2.0 / 3.0)
#define D14 (-1.0 / 12.0)

#define D20 (-1.0 / 12.0)
#define D21 (+4.0 / 3.0)
#define D22 (-2.5)
#define D23 (+4.0 / 3.0)
#define D24 (-1.0 / 12.0)

#define S10 (+0.0)
#define S11 (-1.0 / 12.0)
#define S12 (+0.5)
#define S13 (-1.5)
#define S14 (+5.0 / 6.0)
#define S15 (+0.25)

#define S20 (+1.0 / 12.0)
#define S21 (-0.5)
#define S22 (+7.0 / 6.0)
#define S23 (-1.0 / 3.0)
#define S24 (-1.25)
#define S25 (+5.0 / 6.0)

// Jacobian for centered-centered 4th order stencil and variable omega.
void jacobian_4th_order_variable_omega_cc
(
	double *aa,		// CSR array for values.
	MKL_INT *ia, 		// CSR array for row beginnings. 
	MKL_INT *ja,		// CSR array for columns.
	const MKL_INT NrTotal, 	// Grid total dimension in r.
	const MKL_INT NzTotal, 	// Grid total dimension in z.
	const MKL_INT dim,	// Grid total 2D dimension: dim = NrTotal * NzTotal.
	const MKL_INT ghost,	// Number of ghost zones.
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
	const double u602, const double u612, const double u620, const double u621, const double u622, const double u623, const double u624, const double u632, const double u642,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5,	// Number of elements filled before filling function 5.
	const MKL_INT offset6	// Number of elements filled before filling function 5.
)
{
	// Grid variables.
	double u1 = u122;
	double u2 = u222;
	double u3 = u322;
	double u4 = u422;
	double u5 = u522;
	double u6 = u622;
    
	// Physical names for readability.
	double alpha = exp(u1);
	double Omega = u2;
	double h = exp(u3);
	double a = exp(u4);
	double psi = u5;
	double lambda = u6;
    
	// Coordinates.
	double ri = (double)i + 0.5 - ghost;
	double r = ri * dr;
	double r2 = r * r;
    
	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	double dr2 = dr * dr;
    
	// Scalar field mass and frequency.
	double w = omega_calc(xi, m);
	double m2 = m * m;
	// Omega variable index position.
	MKL_INT w_idx = GNUM * dim;
    
	// Short hands.
	// Scalar field.
	double rlm1 = (l == 1) ? 1.0 : pow(r, l - 1);
	double rl = rlm1 * r;
	double phior = rlm1 * psi;
	double phi = r * phior;
	double phi2or2 = phior * phior;
	double phi2 = phi * phi;
	// Shift combined with scalar field rotation and frequency.
	double wplOmega = w + l * Omega;
	double wplOmega2 = wplOmega * wplOmega;
	// Squared variables.
	double alpha2 = alpha * alpha;
	double h2 = h * h;
	double a2 = a * a;
	// Regularization a2.
	double a2_r = h2 + r2 * lambda;
    
	// Finite differences.
	// Axial derivatives.
	double dRu1 = D10 * u102 + D11 * u112 + D13 * u132 + D14 * u142;
	double dRu2 = D10 * u202 + D11 * u212 + D13 * u232 + D14 * u242;
	double dRu3 = D10 * u302 + D11 * u312 + D13 * u332 + D14 * u342;
	//double dRu4 = D10 * u402 + D11 * u412 + D13 * u432 + D14 * u442;
	double dRu5 = D10 * u502 + D11 * u512 + D13 * u532 + D14 * u542;
	double dRu6 = D10 * u602 + D11 * u612 + D13 * u632 + D14 * u642;
	// Z derivatives.
	double dZu1 = D10 * u120 + D11 * u121 + D13 * u123 + D14 * u124;
	double dZu2 = D10 * u220 + D11 * u221 + D13 * u223 + D14 * u224;
	double dZu3 = D10 * u320 + D11 * u321 + D13 * u323 + D14 * u324;
	//double dZu4 = D10 * u420 + D11 * u421 + D13 * u423 + D14 * u424;
	double dZu5 = D10 * u520 + D11 * u521 + D13 * u523 + D14 * u524;
	double dZu6 = D10 * u620 + D11 * u621 + D13 * u623 + D14 * u624;
	// Second derivatives.
	double dRRu1 = D20 * u102 + D21 * u112 + D22 * u122 + D23 * u132 + D24 * u142;
	double dRRu3 = D20 * u302 + D21 * u312 + D22 * u322 + D23 * u332 + D24 * u342;
    
	// Declare Jacobian submatrices.
	double jacobian_submatrix_1[5] = { 0.0 };
	double jacobian_submatrix_2[5] = { 0.0 };
	double jacobian_submatrix_3[5] = { 0.0 };
	double jacobian_submatrix_4[5] = { 0.0 };
	double jacobian_submatrix_5[5] = { 0.0 };
	double jacobian_submatrix_6[5] = { 0.0 };
	double jacobian_submatrix_w = 0.0;

	// CSR CODE FOR GRID NUMBER 1.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*r2*h2*dZu2*dZu2/alpha2 + dzodr*(r2*h2*dRu2*dRu2/alpha2 + 16.0*M_PI*dr2*a2*phi2*wplOmega2/alpha2);
	jacobian_submatrix_1[1] = dzodr*(2.0*dRu1 + dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = drodz*(2.0*dZu1 + dZu3);
	jacobian_submatrix_1[3] = dzodr;
	jacobian_submatrix_1[4] = drodz;

	// Submatrix 2.
	jacobian_submatrix_2[0] = -dzodr*dr2*16.0*M_PI*l*a2*phi2*wplOmega/alpha2;
	jacobian_submatrix_2[1] = -dzodr*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = -drodz*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = -dzodr*r2*h2*(dRu2*dRu2 + drodz*drodz*dZu2*dZu2)/alpha2;
	jacobian_submatrix_3[1] = dzodr*dRu1;
	jacobian_submatrix_3[2] = drodz*dZu1;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*a2*(m2 - 2.0*wplOmega2/alpha2)*phi*rl;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (-dzodr*dr2*16.0*M_PI*a2*phi2*wplOmega/alpha2);

	// This row 0 * dim + IDX(i, j) starts at offset1
	ia[0 * dim + IDX(i, j)] = BASE + offset1;

	// Values.
	aa[offset1 +  0] = +D10*jacobian_submatrix_1[1]+D20*jacobian_submatrix_1[3];
	aa[offset1 +  1] = +D11*jacobian_submatrix_1[1]+D21*jacobian_submatrix_1[3];
	aa[offset1 +  2] = +D10*jacobian_submatrix_1[2]+D20*jacobian_submatrix_1[4];
	aa[offset1 +  3] = +D11*jacobian_submatrix_1[2]+D21*jacobian_submatrix_1[4];
	aa[offset1 +  4] = +1.0*jacobian_submatrix_1[0]+D22*jacobian_submatrix_1[3]+D22*jacobian_submatrix_1[4];
	aa[offset1 +  5] = +D13*jacobian_submatrix_1[2]+D23*jacobian_submatrix_1[4];
	aa[offset1 +  6] = +D14*jacobian_submatrix_1[2]+D24*jacobian_submatrix_1[4];
	aa[offset1 +  7] = +D13*jacobian_submatrix_1[1]+D23*jacobian_submatrix_1[3];
	aa[offset1 +  8] = +D14*jacobian_submatrix_1[1]+D24*jacobian_submatrix_1[3];
	aa[offset1 +  9] = +D10*jacobian_submatrix_2[1];
	aa[offset1 + 10] = +D11*jacobian_submatrix_2[1];
	aa[offset1 + 11] = +D10*jacobian_submatrix_2[2];
	aa[offset1 + 12] = +D11*jacobian_submatrix_2[2];
	aa[offset1 + 13] = +1.0*jacobian_submatrix_2[0];
	aa[offset1 + 14] = +D13*jacobian_submatrix_2[2];
	aa[offset1 + 15] = +D14*jacobian_submatrix_2[2];
	aa[offset1 + 16] = +D13*jacobian_submatrix_2[1];
	aa[offset1 + 17] = +D14*jacobian_submatrix_2[1];
	aa[offset1 + 18] = +D10*jacobian_submatrix_3[1];
	aa[offset1 + 19] = +D11*jacobian_submatrix_3[1];
	aa[offset1 + 20] = +D10*jacobian_submatrix_3[2];
	aa[offset1 + 21] = +D11*jacobian_submatrix_3[2];
	aa[offset1 + 22] = +1.0*jacobian_submatrix_3[0];
	aa[offset1 + 23] = +D13*jacobian_submatrix_3[2];
	aa[offset1 + 24] = +D14*jacobian_submatrix_3[2];
	aa[offset1 + 25] = +D13*jacobian_submatrix_3[1];
	aa[offset1 + 26] = +D14*jacobian_submatrix_3[1];
	aa[offset1 + 27] = +1.0*jacobian_submatrix_4[0];
	aa[offset1 + 28] = +1.0*jacobian_submatrix_5[0];
	aa[offset1 + 29] = jacobian_submatrix_w;

	// Columns.
	ja[offset1 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset1 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset1 +  2] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset1 +  3] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset1 +  4] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset1 +  5] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset1 +  6] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset1 +  7] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset1 +  8] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset1 +  9] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset1 + 10] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset1 + 11] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset1 + 12] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset1 + 13] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset1 + 14] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset1 + 15] = BASE + 1 * dim + IDX(i    , j + 2);
	ja[offset1 + 16] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset1 + 17] = BASE + 1 * dim + IDX(i + 2, j    );
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
	ja[offset1 + 29] = BASE + w_idx;


	// CSR CODE FOR GRID NUMBER 2.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = 0;
	jacobian_submatrix_1[1] = -dzodr*dRu2;
	jacobian_submatrix_1[2] = -drodz*dZu2;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = -dzodr*dr2*16.0*M_PI*l*l*a2*phi2or2/h2;
	jacobian_submatrix_2[1] = dzodr*(-dRu1 + 3.0*dRu3 + 3.0/ri);
	jacobian_submatrix_2[2] = drodz*(-dZu1 + 3.0*dZu3);
	jacobian_submatrix_2[3] = dzodr;
	jacobian_submatrix_2[4] = drodz;

	// Submatrix 3.
	jacobian_submatrix_3[0] = dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phi2or2/h2;
	jacobian_submatrix_3[1] = dzodr*3.0*dRu2;
	jacobian_submatrix_3[2] = drodz*3.0*dZu2;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = -dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phi2or2/h2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = -dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phior*rlm1/h2;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (-dzodr*dr2*16.0*M_PI*l*a2*phi2or2/h2);

	// This row 1 * dim + IDX(i, j) starts at offset2
	ia[1 * dim + IDX(i, j)] = BASE + offset2;

	// Values.
	aa[offset2 +  0] = +D10*jacobian_submatrix_1[1];
	aa[offset2 +  1] = +D11*jacobian_submatrix_1[1];
	aa[offset2 +  2] = +D10*jacobian_submatrix_1[2];
	aa[offset2 +  3] = +D11*jacobian_submatrix_1[2];
	aa[offset2 +  4] = +D13*jacobian_submatrix_1[2];
	aa[offset2 +  5] = +D14*jacobian_submatrix_1[2];
	aa[offset2 +  6] = +D13*jacobian_submatrix_1[1];
	aa[offset2 +  7] = +D14*jacobian_submatrix_1[1];
	aa[offset2 +  8] = +D10*jacobian_submatrix_2[1]+D20*jacobian_submatrix_2[3];
	aa[offset2 +  9] = +D11*jacobian_submatrix_2[1]+D21*jacobian_submatrix_2[3];
	aa[offset2 + 10] = +D10*jacobian_submatrix_2[2]+D20*jacobian_submatrix_2[4];
	aa[offset2 + 11] = +D11*jacobian_submatrix_2[2]+D21*jacobian_submatrix_2[4];
	aa[offset2 + 12] = +1.0*jacobian_submatrix_2[0]+D22*jacobian_submatrix_2[3]+D22*jacobian_submatrix_2[4];
	aa[offset2 + 13] = +D13*jacobian_submatrix_2[2]+D23*jacobian_submatrix_2[4];
	aa[offset2 + 14] = +D14*jacobian_submatrix_2[2]+D24*jacobian_submatrix_2[4];
	aa[offset2 + 15] = +D13*jacobian_submatrix_2[1]+D23*jacobian_submatrix_2[3];
	aa[offset2 + 16] = +D14*jacobian_submatrix_2[1]+D24*jacobian_submatrix_2[3];
	aa[offset2 + 17] = +D10*jacobian_submatrix_3[1];
	aa[offset2 + 18] = +D11*jacobian_submatrix_3[1];
	aa[offset2 + 19] = +D10*jacobian_submatrix_3[2];
	aa[offset2 + 20] = +D11*jacobian_submatrix_3[2];
	aa[offset2 + 21] = +1.0*jacobian_submatrix_3[0];
	aa[offset2 + 22] = +D13*jacobian_submatrix_3[2];
	aa[offset2 + 23] = +D14*jacobian_submatrix_3[2];
	aa[offset2 + 24] = +D13*jacobian_submatrix_3[1];
	aa[offset2 + 25] = +D14*jacobian_submatrix_3[1];
	aa[offset2 + 26] = +1.0*jacobian_submatrix_4[0];
	aa[offset2 + 27] = +1.0*jacobian_submatrix_5[0];
	aa[offset2 + 28] = jacobian_submatrix_w;

	// Columns.
	ja[offset2 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset2 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset2 +  2] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset2 +  3] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset2 +  4] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset2 +  5] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset2 +  6] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset2 +  7] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset2 +  8] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset2 +  9] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset2 + 10] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset2 + 11] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset2 + 12] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset2 + 13] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset2 + 14] = BASE + 1 * dim + IDX(i    , j + 2);
	ja[offset2 + 15] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset2 + 16] = BASE + 1 * dim + IDX(i + 2, j    );
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
	ja[offset2 + 28] = BASE + w_idx;


	// CSR CODE FOR GRID NUMBER 3.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = -dzodr*r2*h2*dRu2*dRu2/alpha2 - drodz*r2*h2*dZu2*dZu2/alpha2;
	jacobian_submatrix_1[1] = dzodr*(dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = drodz*dZu3;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = 0;
	jacobian_submatrix_2[1] = dzodr*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = drodz*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = drodz*r2*h2*dZu2*dZu2/alpha2 + dzodr*(r2*h2*dRu2*dRu2/alpha2 - dr2*16.0*M_PI*l*l*a2*phi2or2/h2);
	jacobian_submatrix_3[1] = dzodr*(dRu1 + 2.0*dRu3 + 2.0/ri);
	jacobian_submatrix_3[2] = drodz*(dZu1 + 2.0*dZu3);
	jacobian_submatrix_3[3] = dzodr;
	jacobian_submatrix_3[4] = drodz;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*a2*(r2*m2 + 2.0*l*l/h2)*phior*rlm1;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = 0;

	// This row 2 * dim + IDX(i, j) starts at offset3
	ia[2 * dim + IDX(i, j)] = BASE + offset3;

	// Values.
	aa[offset3 +  0] = +D10*jacobian_submatrix_1[1];
	aa[offset3 +  1] = +D11*jacobian_submatrix_1[1];
	aa[offset3 +  2] = +D10*jacobian_submatrix_1[2];
	aa[offset3 +  3] = +D11*jacobian_submatrix_1[2];
	aa[offset3 +  4] = +1.0*jacobian_submatrix_1[0];
	aa[offset3 +  5] = +D13*jacobian_submatrix_1[2];
	aa[offset3 +  6] = +D14*jacobian_submatrix_1[2];
	aa[offset3 +  7] = +D13*jacobian_submatrix_1[1];
	aa[offset3 +  8] = +D14*jacobian_submatrix_1[1];
	aa[offset3 +  9] = +D10*jacobian_submatrix_2[1];
	aa[offset3 + 10] = +D11*jacobian_submatrix_2[1];
	aa[offset3 + 11] = +D10*jacobian_submatrix_2[2];
	aa[offset3 + 12] = +D11*jacobian_submatrix_2[2];
	aa[offset3 + 13] = +D13*jacobian_submatrix_2[2];
	aa[offset3 + 14] = +D14*jacobian_submatrix_2[2];
	aa[offset3 + 15] = +D13*jacobian_submatrix_2[1];
	aa[offset3 + 16] = +D14*jacobian_submatrix_2[1];
	aa[offset3 + 17] = +D10*jacobian_submatrix_3[1]+D20*jacobian_submatrix_3[3];
	aa[offset3 + 18] = +D11*jacobian_submatrix_3[1]+D21*jacobian_submatrix_3[3];
	aa[offset3 + 19] = +D10*jacobian_submatrix_3[2]+D20*jacobian_submatrix_3[4];
	aa[offset3 + 20] = +D11*jacobian_submatrix_3[2]+D21*jacobian_submatrix_3[4];
	aa[offset3 + 21] = +1.0*jacobian_submatrix_3[0]+D22*jacobian_submatrix_3[3]+D22*jacobian_submatrix_3[4];
	aa[offset3 + 22] = +D13*jacobian_submatrix_3[2]+D23*jacobian_submatrix_3[4];
	aa[offset3 + 23] = +D14*jacobian_submatrix_3[2]+D24*jacobian_submatrix_3[4];
	aa[offset3 + 24] = +D13*jacobian_submatrix_3[1]+D23*jacobian_submatrix_3[3];
	aa[offset3 + 25] = +D14*jacobian_submatrix_3[1]+D24*jacobian_submatrix_3[3];
	aa[offset3 + 26] = +1.0*jacobian_submatrix_4[0];
	aa[offset3 + 27] = +1.0*jacobian_submatrix_5[0];

	// Columns.
	ja[offset3 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset3 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset3 +  2] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset3 +  3] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset3 +  4] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset3 +  5] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset3 +  6] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset3 +  7] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset3 +  8] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset3 +  9] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset3 + 10] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset3 + 11] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset3 + 12] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset3 + 13] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset3 + 14] = BASE + 1 * dim + IDX(i    , j + 2);
	ja[offset3 + 15] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset3 + 16] = BASE + 1 * dim + IDX(i + 2, j    );
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


	// CSR CODE FOR GRID NUMBER 4.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*0.5*r2*h2*dZu2*dZu2/alpha2 + dzodr*(0.5*r2*h2*dRu2*dRu2/alpha2 - dr2*8.0*M_PI*a2*wplOmega2*phi2/alpha2);
	jacobian_submatrix_1[1] = -dzodr*(dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = -drodz*dZu3;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = dzodr*dr2*8.0*M_PI*l*a2*wplOmega*phi2/alpha2;
	jacobian_submatrix_2[1] = -dzodr*0.5*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = -drodz*0.5*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = -drodz*0.5*r2*h2*dZu2*dZu2/alpha2 + dzodr*(-0.5*r2*h2*dRu2*dRu2/alpha2 + dr2*8.0*l*l*M_PI*phi2or2*a2/h2);
	jacobian_submatrix_3[1] = -dzodr*dRu1;
	jacobian_submatrix_3[2] = -drodz*dZu1;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(-l*l*phi2or2/h2 + wplOmega2*phi2/alpha2);
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = dzodr;
	jacobian_submatrix_4[4] = drodz;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*(l*ri*dRu5 + l*l*psi - l*l*a2*psi/h2 + r2*a2*wplOmega2*psi/alpha2)*rlm1*rlm1;
	jacobian_submatrix_5[1] = dzodr*8.0*M_PI*rl*rl*(dRu5 + l*psi/ri);
	jacobian_submatrix_5[2] = drodz*8.0*M_PI*rl*rl*dZu5;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (dzodr*dr2*8.0*M_PI*phi2*a2*wplOmega/alpha2);

	// This row 3 * dim + IDX(i, j) starts at offset4
	ia[3 * dim + IDX(i, j)] = BASE + offset4;

	// Values.
	aa[offset4 +  0] = +D10*jacobian_submatrix_1[1];
	aa[offset4 +  1] = +D11*jacobian_submatrix_1[1];
	aa[offset4 +  2] = +D10*jacobian_submatrix_1[2];
	aa[offset4 +  3] = +D11*jacobian_submatrix_1[2];
	aa[offset4 +  4] = +1.0*jacobian_submatrix_1[0];
	aa[offset4 +  5] = +D13*jacobian_submatrix_1[2];
	aa[offset4 +  6] = +D14*jacobian_submatrix_1[2];
	aa[offset4 +  7] = +D13*jacobian_submatrix_1[1];
	aa[offset4 +  8] = +D14*jacobian_submatrix_1[1];
	aa[offset4 +  9] = +D10*jacobian_submatrix_2[1];
	aa[offset4 + 10] = +D11*jacobian_submatrix_2[1];
	aa[offset4 + 11] = +D10*jacobian_submatrix_2[2];
	aa[offset4 + 12] = +D11*jacobian_submatrix_2[2];
	aa[offset4 + 13] = +1.0*jacobian_submatrix_2[0];
	aa[offset4 + 14] = +D13*jacobian_submatrix_2[2];
	aa[offset4 + 15] = +D14*jacobian_submatrix_2[2];
	aa[offset4 + 16] = +D13*jacobian_submatrix_2[1];
	aa[offset4 + 17] = +D14*jacobian_submatrix_2[1];
	aa[offset4 + 18] = +D10*jacobian_submatrix_3[1];
	aa[offset4 + 19] = +D11*jacobian_submatrix_3[1];
	aa[offset4 + 20] = +D10*jacobian_submatrix_3[2];
	aa[offset4 + 21] = +D11*jacobian_submatrix_3[2];
	aa[offset4 + 22] = +1.0*jacobian_submatrix_3[0];
	aa[offset4 + 23] = +D13*jacobian_submatrix_3[2];
	aa[offset4 + 24] = +D14*jacobian_submatrix_3[2];
	aa[offset4 + 25] = +D13*jacobian_submatrix_3[1];
	aa[offset4 + 26] = +D14*jacobian_submatrix_3[1];
	aa[offset4 + 27] = +D20*jacobian_submatrix_4[3];
	aa[offset4 + 28] = +D21*jacobian_submatrix_4[3];
	aa[offset4 + 29] = +D20*jacobian_submatrix_4[4];
	aa[offset4 + 30] = +D21*jacobian_submatrix_4[4];
	aa[offset4 + 31] = +1.0*jacobian_submatrix_4[0]+D22*jacobian_submatrix_4[3]+D22*jacobian_submatrix_4[4];
	aa[offset4 + 32] = +D23*jacobian_submatrix_4[4];
	aa[offset4 + 33] = +D24*jacobian_submatrix_4[4];
	aa[offset4 + 34] = +D23*jacobian_submatrix_4[3];
	aa[offset4 + 35] = +D24*jacobian_submatrix_4[3];
	aa[offset4 + 36] = +D10*jacobian_submatrix_5[1];
	aa[offset4 + 37] = +D11*jacobian_submatrix_5[1];
	aa[offset4 + 38] = +D10*jacobian_submatrix_5[2];
	aa[offset4 + 39] = +D11*jacobian_submatrix_5[2];
	aa[offset4 + 40] = +1.0*jacobian_submatrix_5[0];
	aa[offset4 + 41] = +D13*jacobian_submatrix_5[2];
	aa[offset4 + 42] = +D14*jacobian_submatrix_5[2];
	aa[offset4 + 43] = +D13*jacobian_submatrix_5[1];
	aa[offset4 + 44] = +D14*jacobian_submatrix_5[1];
	aa[offset4 + 45] = jacobian_submatrix_w;

	// Columns.
	ja[offset4 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset4 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset4 +  2] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset4 +  3] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset4 +  4] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset4 +  5] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset4 +  6] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset4 +  7] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset4 +  8] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset4 +  9] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset4 + 10] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset4 + 11] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset4 + 12] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset4 + 13] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset4 + 14] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset4 + 15] = BASE + 1 * dim + IDX(i    , j + 2);
	ja[offset4 + 16] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset4 + 17] = BASE + 1 * dim + IDX(i + 2, j    );
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
	ja[offset4 + 45] = BASE + w_idx;


	// CSR CODE FOR GRID NUMBER 5.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = -dzodr*dr2*2.0*a2*wplOmega2*psi/alpha2;
	jacobian_submatrix_1[1] = dzodr*(dRu5 + l*psi/ri);
	jacobian_submatrix_1[2] = drodz*dZu5;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = dzodr*dr2*2.0*l*a2*wplOmega*psi/alpha2;
	jacobian_submatrix_2[1] = 0;
	jacobian_submatrix_2[2] = 0;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = dzodr*dr2*2.0*l*l*psi*lambda/h2;
	jacobian_submatrix_3[1] = dzodr*(dRu5 + l*psi/ri);
	jacobian_submatrix_3[2] = drodz*dZu5;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = -dzodr*dr2*2.0*a2*(m2 - wplOmega2/alpha2)*psi;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*(l*(dRu1/ri + dRu3/ri) - dr2*(a2*(m2 - wplOmega2/alpha2) + l*l*lambda/h2));
	jacobian_submatrix_5[1] = dzodr*(dRu1 + dRu3 + (2.0*l + 1.0)/ri);
	jacobian_submatrix_5[2] = drodz*(dZu1 + dZu3);
	jacobian_submatrix_5[3] = dzodr;
	jacobian_submatrix_5[4] = drodz;

	// Submatrix 6.
	jacobian_submatrix_6[0] = -dzodr*dr2*l*l*psi/h2;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (dzodr*dr2*2.0*a2*wplOmega*psi/alpha2);

	// This row 4 * dim + IDX(i, j) starts at offset5
	ia[4 * dim + IDX(i, j)] = BASE + offset5;

	// Values.
	aa[offset5 +  0] = +D10*jacobian_submatrix_1[1];
	aa[offset5 +  1] = +D11*jacobian_submatrix_1[1];
	aa[offset5 +  2] = +D10*jacobian_submatrix_1[2];
	aa[offset5 +  3] = +D11*jacobian_submatrix_1[2];
	aa[offset5 +  4] = +1.0*jacobian_submatrix_1[0];
	aa[offset5 +  5] = +D13*jacobian_submatrix_1[2];
	aa[offset5 +  6] = +D14*jacobian_submatrix_1[2];
	aa[offset5 +  7] = +D13*jacobian_submatrix_1[1];
	aa[offset5 +  8] = +D14*jacobian_submatrix_1[1];
	aa[offset5 +  9] = +1.0*jacobian_submatrix_2[0];
	aa[offset5 + 10] = +D10*jacobian_submatrix_3[1];
	aa[offset5 + 11] = +D11*jacobian_submatrix_3[1];
	aa[offset5 + 12] = +D10*jacobian_submatrix_3[2];
	aa[offset5 + 13] = +D11*jacobian_submatrix_3[2];
	aa[offset5 + 14] = +1.0*jacobian_submatrix_3[0];
	aa[offset5 + 15] = +D13*jacobian_submatrix_3[2];
	aa[offset5 + 16] = +D14*jacobian_submatrix_3[2];
	aa[offset5 + 17] = +D13*jacobian_submatrix_3[1];
	aa[offset5 + 18] = +D14*jacobian_submatrix_3[1];
	aa[offset5 + 19] = +1.0*jacobian_submatrix_4[0];
	aa[offset5 + 20] = +D10*jacobian_submatrix_5[1]+D20*jacobian_submatrix_5[3];
	aa[offset5 + 21] = +D11*jacobian_submatrix_5[1]+D21*jacobian_submatrix_5[3];
	aa[offset5 + 22] = +D10*jacobian_submatrix_5[2]+D20*jacobian_submatrix_5[4];
	aa[offset5 + 23] = +D11*jacobian_submatrix_5[2]+D21*jacobian_submatrix_5[4];
	aa[offset5 + 24] = +1.0*jacobian_submatrix_5[0]+D22*jacobian_submatrix_5[3]+D22*jacobian_submatrix_5[4];
	aa[offset5 + 25] = +D13*jacobian_submatrix_5[2]+D23*jacobian_submatrix_5[4];
	aa[offset5 + 26] = +D14*jacobian_submatrix_5[2]+D24*jacobian_submatrix_5[4];
	aa[offset5 + 27] = +D13*jacobian_submatrix_5[1]+D23*jacobian_submatrix_5[3];
	aa[offset5 + 28] = +D14*jacobian_submatrix_5[1]+D24*jacobian_submatrix_5[3];
	aa[offset5 + 29] = +1.0*jacobian_submatrix_6[0];
	aa[offset5 + 30] = jacobian_submatrix_w;

	// Columns.
	ja[offset5 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset5 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset5 +  2] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset5 +  3] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset5 +  4] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset5 +  5] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset5 +  6] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset5 +  7] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset5 +  8] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset5 +  9] = BASE + 1 * dim + IDX(i    , j    );
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
	ja[offset5 + 29] = BASE + 5 * dim + IDX(i    , j    );
	ja[offset5 + 30] = BASE + w_idx;


	// CSR CODE FOR GRID NUMBER 6.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*(2.0*dZu2*dZu2*h2*h2/alpha2) + dzodr*(2.0*h2 + r2*lambda)*(2.0*dRu2*dRu2*h2/alpha2);
	jacobian_submatrix_1[1] = dzodr*(-dRu6 + 4.0*dRu1*(lambda + Q1*h2/r2) - (2.0/ri)*(lambda + Q1*h2/r2) - 4.0*dRu3*h2/r2);
	jacobian_submatrix_1[2] = drodz*dZu6;
	jacobian_submatrix_1[3] = dzodr*2.0*(lambda + Q1*h2/r2);
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = 0;
	jacobian_submatrix_2[1] = dzodr*(-2.0*dRu2*h2/alpha2)*(2.0*h2 + r2*lambda);
	jacobian_submatrix_2[2] = drodz*(-2.0*dZu2*h2*h2/alpha2);
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = drodz*(2.0*h2)*(-2.0*dZu2*dZu2*h2/alpha2 + r2*(dZu6 - 2.0*dZu3*lambda)*(dZu6 - 2.0*dZu3*lambda)/(a2_r*a2_r))+ dzodr*((Q1*(dRRu1 + dRu1*(dRu1 - 1.0/ri)) + Q2*(dRRu3 + dRu3*(2.0*dRu3 - 1.0/ri)))*(4.0*h2/r2) - 8.0*dRu1*dRu3*(h2/r2) + 64.0*M_PI*l*h2*rlm1*rlm1*psi*(dRu5/ri) + 32.0*M_PI*h2*rlm1*rlm1*(dRu5*dRu5) + 8.0*h2*r2*lambda*(dRu6/ri)/(a2_r*a2_r) + 2.0*r2*h2*dRu6*dRu6/(a2_r*a2_r) + (dRu2*dRu2)*(-8.0*h2*h2/alpha2 - 2.0*r2*lambda*h2/alpha2) + dr2*h2*(16.0*M_PI*rlm1*rlm1*r2*lambda*m2*psi*psi + 8.0*(lambda/a2_r)*(lambda/a2_r)) + (dRu3/ri)*(-16.0*h2*r2*lambda*lambda/(a2_r*a2_r) - 8.0*h2*r2*lambda*(ri*dRu6)/(a2_r*a2_r)) + (dRu3*dRu3)*(h2/r2)*(-4.0 + 8.0*(h2/a2_r)*(h2/a2_r) - 16.0*h2/a2_r));
	jacobian_submatrix_3[1] = dzodr*(Q2*(8.0*dRu3 - 2.0/ri)*(lambda + h2/r2)*(h2/a2_r) + dRu6*(-5.0*h2 - r2*lambda)/a2_r - 4.0*dRu1*(h2/a2_r)*(lambda + h2/r2) +dRu3*(-12.0*h2*h2/r2 + 4.0*r2*lambda*lambda)/a2_r + lambda*(-6.0*h2/ri + 2.0*r2*lambda/ri)/a2_r);
	jacobian_submatrix_3[2] = drodz*(8.0*dZu3*lambda*h2 + dZu6*(-3.0*h2 + r2*lambda))/a2_r;
	jacobian_submatrix_3[3] = dzodr*2.0*(lambda + Q2*h2/r2);
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = 0;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*(16.0*a2_r*M_PI*rlm1*rlm1)*(dr2*r2*lambda*m2*psi + 2.0*l*dRu5/ri);
	jacobian_submatrix_5[1] = dzodr*(32.0*a2_r*M_PI*rlm1*rlm1)*(l*psi/ri + dRu5);
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = drodz*(r2*dZu6 + 2.0*h2*dZu3)*(r2*dZu6 + 2.0*h2*dZu3)/(a2_r*a2_r) + dzodr*(2.0*dRRu1 + 2.0*dRRu3 + 2.0*dRu1*(dRu1 - 1.0/ri) - r2*h2*dRu2*dRu2/alpha2 + 16.0*M_PI*rl*rl*dRu5*dRu5 + 32.0*M_PI*l*rl*rl*psi*(dRu5/ri) + (r2*dRu6/a2_r)*(r2*dRu6/a2_r) + (dRu3*dRu3)*(2.0 + 4.0*(h2/a2_r)*(h2/a2_r)) + dr2*(r2*lambda)*(8.0*M_PI*m2*phi2 + 4.0*lambda/(a2_r*a2_r)) + (dRu6/ri)*(4.0*r2)*(-h2/(a2_r*a2_r)) + (dRu3/ri)*(-6.0*(h2/a2_r)*(h2/a2_r) + 4.0*h2*(ri*r2*dRu6)/(a2_r*a2_r) - 2.0*(r2*lambda/a2_r)*(r2*lambda/a2_r) + 4.0*r2*lambda/a2_r) + dr2*(-8.0*lambda/a2_r + 8.0*M_PI*a2_r*m2*phi2));
	jacobian_submatrix_6[1] = dzodr*((3.0*h2 - r2*lambda)/ri - dRu1*(h2 + r2*lambda) - dRu3*(5.0*h2 + r2*lambda) - 2.0*r2*dRu6)/a2_r;
	jacobian_submatrix_6[2] = drodz*(-2.0*r2*dZu6 + dZu1*(h2 + r2*lambda) + dZu3*(-3.0*h2 + r2*lambda))/a2_r;
	jacobian_submatrix_6[3] = dzodr;
	jacobian_submatrix_6[4] = drodz;

	// Omega term.
	jacobian_submatrix_w = 0;

	// This row 5 * dim + IDX(i, j) starts at offset6
	ia[5 * dim + IDX(i, j)] = BASE + offset6;

	// Values.
	aa[offset6 +  0] = +D10*jacobian_submatrix_1[1]+D20*jacobian_submatrix_1[3];
	aa[offset6 +  1] = +D11*jacobian_submatrix_1[1]+D21*jacobian_submatrix_1[3];
	aa[offset6 +  2] = +D10*jacobian_submatrix_1[2];
	aa[offset6 +  3] = +D11*jacobian_submatrix_1[2];
	aa[offset6 +  4] = +1.0*jacobian_submatrix_1[0]+D22*jacobian_submatrix_1[3];
	aa[offset6 +  5] = +D13*jacobian_submatrix_1[2];
	aa[offset6 +  6] = +D14*jacobian_submatrix_1[2];
	aa[offset6 +  7] = +D13*jacobian_submatrix_1[1]+D23*jacobian_submatrix_1[3];
	aa[offset6 +  8] = +D14*jacobian_submatrix_1[1]+D24*jacobian_submatrix_1[3];
	aa[offset6 +  9] = +D10*jacobian_submatrix_2[1];
	aa[offset6 + 10] = +D11*jacobian_submatrix_2[1];
	aa[offset6 + 11] = +D10*jacobian_submatrix_2[2];
	aa[offset6 + 12] = +D11*jacobian_submatrix_2[2];
	aa[offset6 + 13] = +D13*jacobian_submatrix_2[2];
	aa[offset6 + 14] = +D14*jacobian_submatrix_2[2];
	aa[offset6 + 15] = +D13*jacobian_submatrix_2[1];
	aa[offset6 + 16] = +D14*jacobian_submatrix_2[1];
	aa[offset6 + 17] = +D10*jacobian_submatrix_3[1]+D20*jacobian_submatrix_3[3];
	aa[offset6 + 18] = +D11*jacobian_submatrix_3[1]+D21*jacobian_submatrix_3[3];
	aa[offset6 + 19] = +D10*jacobian_submatrix_3[2];
	aa[offset6 + 20] = +D11*jacobian_submatrix_3[2];
	aa[offset6 + 21] = +1.0*jacobian_submatrix_3[0]+D22*jacobian_submatrix_3[3];
	aa[offset6 + 22] = +D13*jacobian_submatrix_3[2];
	aa[offset6 + 23] = +D14*jacobian_submatrix_3[2];
	aa[offset6 + 24] = +D13*jacobian_submatrix_3[1]+D23*jacobian_submatrix_3[3];
	aa[offset6 + 25] = +D14*jacobian_submatrix_3[1]+D24*jacobian_submatrix_3[3];
	aa[offset6 + 26] = +D10*jacobian_submatrix_5[1];
	aa[offset6 + 27] = +D11*jacobian_submatrix_5[1];
	aa[offset6 + 28] = +1.0*jacobian_submatrix_5[0];
	aa[offset6 + 29] = +D13*jacobian_submatrix_5[1];
	aa[offset6 + 30] = +D14*jacobian_submatrix_5[1];
	aa[offset6 + 31] = +D10*jacobian_submatrix_6[1]+D20*jacobian_submatrix_6[3];
	aa[offset6 + 32] = +D11*jacobian_submatrix_6[1]+D21*jacobian_submatrix_6[3];
	aa[offset6 + 33] = +D10*jacobian_submatrix_6[2]+D20*jacobian_submatrix_6[4];
	aa[offset6 + 34] = +D11*jacobian_submatrix_6[2]+D21*jacobian_submatrix_6[4];
	aa[offset6 + 35] = +1.0*jacobian_submatrix_6[0]+D22*jacobian_submatrix_6[3]+D22*jacobian_submatrix_6[4];
	aa[offset6 + 36] = +D13*jacobian_submatrix_6[2]+D23*jacobian_submatrix_6[4];
	aa[offset6 + 37] = +D14*jacobian_submatrix_6[2]+D24*jacobian_submatrix_6[4];
	aa[offset6 + 38] = +D13*jacobian_submatrix_6[1]+D23*jacobian_submatrix_6[3];
	aa[offset6 + 39] = +D14*jacobian_submatrix_6[1]+D24*jacobian_submatrix_6[3];

	// Columns.
	ja[offset6 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset6 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset6 +  2] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset6 +  3] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset6 +  4] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset6 +  5] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset6 +  6] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset6 +  7] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset6 +  8] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset6 +  9] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset6 + 10] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset6 + 11] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset6 + 12] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset6 + 13] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset6 + 14] = BASE + 1 * dim + IDX(i    , j + 2);
	ja[offset6 + 15] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset6 + 16] = BASE + 1 * dim + IDX(i + 2, j    );
	ja[offset6 + 17] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset6 + 18] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset6 + 19] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset6 + 20] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset6 + 21] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset6 + 22] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset6 + 23] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset6 + 24] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset6 + 25] = BASE + 2 * dim + IDX(i + 2, j    );
	ja[offset6 + 26] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset6 + 27] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset6 + 28] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset6 + 29] = BASE + 4 * dim + IDX(i + 1, j    );
	ja[offset6 + 30] = BASE + 4 * dim + IDX(i + 2, j    );
	ja[offset6 + 31] = BASE + 5 * dim + IDX(i - 2, j    );
	ja[offset6 + 32] = BASE + 5 * dim + IDX(i - 1, j    );
	ja[offset6 + 33] = BASE + 5 * dim + IDX(i    , j - 2);
	ja[offset6 + 34] = BASE + 5 * dim + IDX(i    , j - 1);
	ja[offset6 + 35] = BASE + 5 * dim + IDX(i    , j    );
	ja[offset6 + 36] = BASE + 5 * dim + IDX(i    , j + 1);
	ja[offset6 + 37] = BASE + 5 * dim + IDX(i    , j + 2);
	ja[offset6 + 38] = BASE + 5 * dim + IDX(i + 1, j    );
	ja[offset6 + 39] = BASE + 5 * dim + IDX(i + 2, j    );


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
	const MKL_INT ghost,	// Number of ghost zones.
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
	const double u604, const double u614, const double u620, const double u621, const double u622, const double u623, const double u624, const double u625, const double u634, const double u644,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5,	// Number of elements filled before filling function 5
	const MKL_INT offset6	// Number of elements filled before filling function 5
)
{
	// Grid variables.
	double u1 = u124;
	double u2 = u224;
	double u3 = u324;
	double u4 = u424;
	double u5 = u524;
	double u6 = u624;
    
	// Physical names for readability.
	double alpha = exp(u1);
	double Omega = u2;
	double h = exp(u3);
	double a = exp(u4);
	double psi = u5;
	double lambda = u6;
    
	// Coordinates.
	double ri = (double)i + 0.5 - ghost;
	double r = ri * dr;
	double r2 = r * r;
    
	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	double dr2 = dr * dr;
    
	// Scalar field mass and frequency.
	double w = omega_calc(xi, m);
	double m2 = m * m;
	// Omega variable index position.
	MKL_INT w_idx = GNUM * dim;
    
	// Short hands.
	// Scalar field.
	double rlm1 = (l == 1) ? 1.0 : pow(r, l - 1);
	double rl = rlm1 * r;
	double phior = rlm1 * psi;
	double phi = r * phior;
	double phi2or2 = phior * phior;
	double phi2 = phi * phi;
	// Shift combined with scalar field rotation and frequency.
	double wplOmega = w + l * Omega;
	double wplOmega2 = wplOmega * wplOmega;
	// Squared variables.
	double alpha2 = alpha * alpha;
	double h2 = h * h;
	double a2 = a * a;
	// Regularization a2.
	double a2_r = h2 + r2 * lambda;
    
	// Finite differences.
	// Axial derivatives.
	double dRu1 = D10 * u104 + D11 * u114 + D13 * u134 + D14 * u144;
	double dRu2 = D10 * u204 + D11 * u214 + D13 * u234 + D14 * u244;
	double dRu3 = D10 * u304 + D11 * u314 + D13 * u334 + D14 * u344;
	//double dRu4 = D10 * u404 + D11 * u414 + D13 * u434 + D14 * u444;
	double dRu5 = D10 * u504 + D11 * u514 + D13 * u534 + D14 * u544;
	double dRu6 = D10 * u604 + D11 * u614 + D13 * u634 + D14 * u644;
	// Z derivatives.
	double dZu1 = S11 * u121 + S12 * u122 + S13 * u123 + S14 * u124 + S15 * u125;
	double dZu2 = S11 * u221 + S12 * u222 + S13 * u223 + S14 * u224 + S15 * u225;
	double dZu3 = S11 * u321 + S12 * u322 + S13 * u323 + S14 * u324 + S15 * u325;
	//double dZu4 = S11 * u421 + S12 * u422 + S13 * u423 + S14 * u424 + S15 * u425;
	double dZu5 = S11 * u521 + S12 * u522 + S13 * u523 + S14 * u524 + S15 * u525;
	double dZu6 = S11 * u621 + S12 * u622 + S13 * u623 + S14 * u624 + S15 * u625;
	// Second derivatives.
	double dRRu1 = D20 * u104 + D21 * u114 + D22 * u124 + D23 * u134 + D24 * u144;
	double dRRu3 = D20 * u304 + D21 * u314 + D22 * u324 + D23 * u334 + D24 * u344;
    
	// Declare Jacobian submatrices.
	double jacobian_submatrix_1[5] = { 0.0 };
	double jacobian_submatrix_2[5] = { 0.0 };
	double jacobian_submatrix_3[5] = { 0.0 };
	double jacobian_submatrix_4[5] = { 0.0 };
	double jacobian_submatrix_5[5] = { 0.0 };
	double jacobian_submatrix_6[5] = { 0.0 };
	double jacobian_submatrix_w = 0.0;

	// CSR CODE FOR GRID NUMBER 1.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*r2*h2*dZu2*dZu2/alpha2 + dzodr*(r2*h2*dRu2*dRu2/alpha2 + 16.0*M_PI*dr2*a2*phi2*wplOmega2/alpha2);
	jacobian_submatrix_1[1] = dzodr*(2.0*dRu1 + dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = drodz*(2.0*dZu1 + dZu3);
	jacobian_submatrix_1[3] = dzodr;
	jacobian_submatrix_1[4] = drodz;

	// Submatrix 2.
	jacobian_submatrix_2[0] = -dzodr*dr2*16.0*M_PI*l*a2*phi2*wplOmega/alpha2;
	jacobian_submatrix_2[1] = -dzodr*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = -drodz*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = -dzodr*r2*h2*(dRu2*dRu2 + drodz*drodz*dZu2*dZu2)/alpha2;
	jacobian_submatrix_3[1] = dzodr*dRu1;
	jacobian_submatrix_3[2] = drodz*dZu1;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*a2*(m2 - 2.0*wplOmega2/alpha2)*phi*rl;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (-dzodr*dr2*16.0*M_PI*a2*phi2*wplOmega/alpha2);

	// This row 0 * dim + IDX(i, j) starts at offset1
	ia[0 * dim + IDX(i, j)] = BASE + offset1;

	// Values.
	aa[offset1 +  0] = +D10*jacobian_submatrix_1[1]+D20*jacobian_submatrix_1[3];
	aa[offset1 +  1] = +D11*jacobian_submatrix_1[1]+D21*jacobian_submatrix_1[3];
	aa[offset1 +  2] = +S20*jacobian_submatrix_1[4];
	aa[offset1 +  3] = +S11*jacobian_submatrix_1[2]+S21*jacobian_submatrix_1[4];
	aa[offset1 +  4] = +S12*jacobian_submatrix_1[2]+S22*jacobian_submatrix_1[4];
	aa[offset1 +  5] = +S13*jacobian_submatrix_1[2]+S23*jacobian_submatrix_1[4];
	aa[offset1 +  6] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[2]+D22*jacobian_submatrix_1[3]+S24*jacobian_submatrix_1[4];
	aa[offset1 +  7] = +S15*jacobian_submatrix_1[2]+S25*jacobian_submatrix_1[4];
	aa[offset1 +  8] = +D13*jacobian_submatrix_1[1]+D23*jacobian_submatrix_1[3];
	aa[offset1 +  9] = +D14*jacobian_submatrix_1[1]+D24*jacobian_submatrix_1[3];
	aa[offset1 + 10] = +D10*jacobian_submatrix_2[1];
	aa[offset1 + 11] = +D11*jacobian_submatrix_2[1];
	aa[offset1 + 12] = +S11*jacobian_submatrix_2[2];
	aa[offset1 + 13] = +S12*jacobian_submatrix_2[2];
	aa[offset1 + 14] = +S13*jacobian_submatrix_2[2];
	aa[offset1 + 15] = +1.0*jacobian_submatrix_2[0]+S14*jacobian_submatrix_2[2];
	aa[offset1 + 16] = +S15*jacobian_submatrix_2[2];
	aa[offset1 + 17] = +D13*jacobian_submatrix_2[1];
	aa[offset1 + 18] = +D14*jacobian_submatrix_2[1];
	aa[offset1 + 19] = +D10*jacobian_submatrix_3[1];
	aa[offset1 + 20] = +D11*jacobian_submatrix_3[1];
	aa[offset1 + 21] = +S11*jacobian_submatrix_3[2];
	aa[offset1 + 22] = +S12*jacobian_submatrix_3[2];
	aa[offset1 + 23] = +S13*jacobian_submatrix_3[2];
	aa[offset1 + 24] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[2];
	aa[offset1 + 25] = +S15*jacobian_submatrix_3[2];
	aa[offset1 + 26] = +D13*jacobian_submatrix_3[1];
	aa[offset1 + 27] = +D14*jacobian_submatrix_3[1];
	aa[offset1 + 28] = +1.0*jacobian_submatrix_4[0];
	aa[offset1 + 29] = +1.0*jacobian_submatrix_5[0];
	aa[offset1 + 30] = jacobian_submatrix_w;

	// Columns.
	ja[offset1 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset1 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset1 +  2] = BASE + 0 * dim + IDX(i    , j - 4);
	ja[offset1 +  3] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset1 +  4] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset1 +  5] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset1 +  6] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset1 +  7] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset1 +  8] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset1 +  9] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset1 + 10] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset1 + 11] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset1 + 12] = BASE + 1 * dim + IDX(i    , j - 3);
	ja[offset1 + 13] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset1 + 14] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset1 + 15] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset1 + 16] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset1 + 17] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset1 + 18] = BASE + 1 * dim + IDX(i + 2, j    );
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
	ja[offset1 + 30] = BASE + w_idx;


	// CSR CODE FOR GRID NUMBER 2.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = 0;
	jacobian_submatrix_1[1] = -dzodr*dRu2;
	jacobian_submatrix_1[2] = -drodz*dZu2;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = -dzodr*dr2*16.0*M_PI*l*l*a2*phi2or2/h2;
	jacobian_submatrix_2[1] = dzodr*(-dRu1 + 3.0*dRu3 + 3.0/ri);
	jacobian_submatrix_2[2] = drodz*(-dZu1 + 3.0*dZu3);
	jacobian_submatrix_2[3] = dzodr;
	jacobian_submatrix_2[4] = drodz;

	// Submatrix 3.
	jacobian_submatrix_3[0] = dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phi2or2/h2;
	jacobian_submatrix_3[1] = dzodr*3.0*dRu2;
	jacobian_submatrix_3[2] = drodz*3.0*dZu2;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = -dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phi2or2/h2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = -dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phior*rlm1/h2;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (-dzodr*dr2*16.0*M_PI*l*a2*phi2or2/h2);

	// This row 1 * dim + IDX(i, j) starts at offset2
	ia[1 * dim + IDX(i, j)] = BASE + offset2;

	// Values.
	aa[offset2 +  0] = +D10*jacobian_submatrix_1[1];
	aa[offset2 +  1] = +D11*jacobian_submatrix_1[1];
	aa[offset2 +  2] = +S11*jacobian_submatrix_1[2];
	aa[offset2 +  3] = +S12*jacobian_submatrix_1[2];
	aa[offset2 +  4] = +S13*jacobian_submatrix_1[2];
	aa[offset2 +  5] = +S14*jacobian_submatrix_1[2];
	aa[offset2 +  6] = +S15*jacobian_submatrix_1[2];
	aa[offset2 +  7] = +D13*jacobian_submatrix_1[1];
	aa[offset2 +  8] = +D14*jacobian_submatrix_1[1];
	aa[offset2 +  9] = +D10*jacobian_submatrix_2[1]+D20*jacobian_submatrix_2[3];
	aa[offset2 + 10] = +D11*jacobian_submatrix_2[1]+D21*jacobian_submatrix_2[3];
	aa[offset2 + 11] = +S20*jacobian_submatrix_2[4];
	aa[offset2 + 12] = +S11*jacobian_submatrix_2[2]+S21*jacobian_submatrix_2[4];
	aa[offset2 + 13] = +S12*jacobian_submatrix_2[2]+S22*jacobian_submatrix_2[4];
	aa[offset2 + 14] = +S13*jacobian_submatrix_2[2]+S23*jacobian_submatrix_2[4];
	aa[offset2 + 15] = +1.0*jacobian_submatrix_2[0]+S14*jacobian_submatrix_2[2]+D22*jacobian_submatrix_2[3]+S24*jacobian_submatrix_2[4];
	aa[offset2 + 16] = +S15*jacobian_submatrix_2[2]+S25*jacobian_submatrix_2[4];
	aa[offset2 + 17] = +D13*jacobian_submatrix_2[1]+D23*jacobian_submatrix_2[3];
	aa[offset2 + 18] = +D14*jacobian_submatrix_2[1]+D24*jacobian_submatrix_2[3];
	aa[offset2 + 19] = +D10*jacobian_submatrix_3[1];
	aa[offset2 + 20] = +D11*jacobian_submatrix_3[1];
	aa[offset2 + 21] = +S11*jacobian_submatrix_3[2];
	aa[offset2 + 22] = +S12*jacobian_submatrix_3[2];
	aa[offset2 + 23] = +S13*jacobian_submatrix_3[2];
	aa[offset2 + 24] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[2];
	aa[offset2 + 25] = +S15*jacobian_submatrix_3[2];
	aa[offset2 + 26] = +D13*jacobian_submatrix_3[1];
	aa[offset2 + 27] = +D14*jacobian_submatrix_3[1];
	aa[offset2 + 28] = +1.0*jacobian_submatrix_4[0];
	aa[offset2 + 29] = +1.0*jacobian_submatrix_5[0];
	aa[offset2 + 30] = jacobian_submatrix_w;

	// Columns.
	ja[offset2 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset2 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset2 +  2] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset2 +  3] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset2 +  4] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset2 +  5] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset2 +  6] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset2 +  7] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset2 +  8] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset2 +  9] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset2 + 10] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset2 + 11] = BASE + 1 * dim + IDX(i    , j - 4);
	ja[offset2 + 12] = BASE + 1 * dim + IDX(i    , j - 3);
	ja[offset2 + 13] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset2 + 14] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset2 + 15] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset2 + 16] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset2 + 17] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset2 + 18] = BASE + 1 * dim + IDX(i + 2, j    );
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
	ja[offset2 + 30] = BASE + w_idx;


	// CSR CODE FOR GRID NUMBER 3.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = -dzodr*r2*h2*dRu2*dRu2/alpha2 - drodz*r2*h2*dZu2*dZu2/alpha2;
	jacobian_submatrix_1[1] = dzodr*(dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = drodz*dZu3;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = 0;
	jacobian_submatrix_2[1] = dzodr*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = drodz*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = drodz*r2*h2*dZu2*dZu2/alpha2 + dzodr*(r2*h2*dRu2*dRu2/alpha2 - dr2*16.0*M_PI*l*l*a2*phi2or2/h2);
	jacobian_submatrix_3[1] = dzodr*(dRu1 + 2.0*dRu3 + 2.0/ri);
	jacobian_submatrix_3[2] = drodz*(dZu1 + 2.0*dZu3);
	jacobian_submatrix_3[3] = dzodr;
	jacobian_submatrix_3[4] = drodz;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*a2*(r2*m2 + 2.0*l*l/h2)*phior*rlm1;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = 0;

	// This row 2 * dim + IDX(i, j) starts at offset3
	ia[2 * dim + IDX(i, j)] = BASE + offset3;

	// Values.
	aa[offset3 +  0] = +D10*jacobian_submatrix_1[1];
	aa[offset3 +  1] = +D11*jacobian_submatrix_1[1];
	aa[offset3 +  2] = +S11*jacobian_submatrix_1[2];
	aa[offset3 +  3] = +S12*jacobian_submatrix_1[2];
	aa[offset3 +  4] = +S13*jacobian_submatrix_1[2];
	aa[offset3 +  5] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[2];
	aa[offset3 +  6] = +S15*jacobian_submatrix_1[2];
	aa[offset3 +  7] = +D13*jacobian_submatrix_1[1];
	aa[offset3 +  8] = +D14*jacobian_submatrix_1[1];
	aa[offset3 +  9] = +D10*jacobian_submatrix_2[1];
	aa[offset3 + 10] = +D11*jacobian_submatrix_2[1];
	aa[offset3 + 11] = +S11*jacobian_submatrix_2[2];
	aa[offset3 + 12] = +S12*jacobian_submatrix_2[2];
	aa[offset3 + 13] = +S13*jacobian_submatrix_2[2];
	aa[offset3 + 14] = +S14*jacobian_submatrix_2[2];
	aa[offset3 + 15] = +S15*jacobian_submatrix_2[2];
	aa[offset3 + 16] = +D13*jacobian_submatrix_2[1];
	aa[offset3 + 17] = +D14*jacobian_submatrix_2[1];
	aa[offset3 + 18] = +D10*jacobian_submatrix_3[1]+D20*jacobian_submatrix_3[3];
	aa[offset3 + 19] = +D11*jacobian_submatrix_3[1]+D21*jacobian_submatrix_3[3];
	aa[offset3 + 20] = +S20*jacobian_submatrix_3[4];
	aa[offset3 + 21] = +S11*jacobian_submatrix_3[2]+S21*jacobian_submatrix_3[4];
	aa[offset3 + 22] = +S12*jacobian_submatrix_3[2]+S22*jacobian_submatrix_3[4];
	aa[offset3 + 23] = +S13*jacobian_submatrix_3[2]+S23*jacobian_submatrix_3[4];
	aa[offset3 + 24] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[2]+D22*jacobian_submatrix_3[3]+S24*jacobian_submatrix_3[4];
	aa[offset3 + 25] = +S15*jacobian_submatrix_3[2]+S25*jacobian_submatrix_3[4];
	aa[offset3 + 26] = +D13*jacobian_submatrix_3[1]+D23*jacobian_submatrix_3[3];
	aa[offset3 + 27] = +D14*jacobian_submatrix_3[1]+D24*jacobian_submatrix_3[3];
	aa[offset3 + 28] = +1.0*jacobian_submatrix_4[0];
	aa[offset3 + 29] = +1.0*jacobian_submatrix_5[0];

	// Columns.
	ja[offset3 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset3 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset3 +  2] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset3 +  3] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset3 +  4] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset3 +  5] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset3 +  6] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset3 +  7] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset3 +  8] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset3 +  9] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset3 + 10] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset3 + 11] = BASE + 1 * dim + IDX(i    , j - 3);
	ja[offset3 + 12] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset3 + 13] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset3 + 14] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset3 + 15] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset3 + 16] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset3 + 17] = BASE + 1 * dim + IDX(i + 2, j    );
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


	// CSR CODE FOR GRID NUMBER 4.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*0.5*r2*h2*dZu2*dZu2/alpha2 + dzodr*(0.5*r2*h2*dRu2*dRu2/alpha2 - dr2*8.0*M_PI*a2*wplOmega2*phi2/alpha2);
	jacobian_submatrix_1[1] = -dzodr*(dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = -drodz*dZu3;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = dzodr*dr2*8.0*M_PI*l*a2*wplOmega*phi2/alpha2;
	jacobian_submatrix_2[1] = -dzodr*0.5*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = -drodz*0.5*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = -drodz*0.5*r2*h2*dZu2*dZu2/alpha2 + dzodr*(-0.5*r2*h2*dRu2*dRu2/alpha2 + dr2*8.0*l*l*M_PI*phi2or2*a2/h2);
	jacobian_submatrix_3[1] = -dzodr*dRu1;
	jacobian_submatrix_3[2] = -drodz*dZu1;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(-l*l*phi2or2/h2 + wplOmega2*phi2/alpha2);
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = dzodr;
	jacobian_submatrix_4[4] = drodz;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*(l*ri*dRu5 + l*l*psi - l*l*a2*psi/h2 + r2*a2*wplOmega2*psi/alpha2)*rlm1*rlm1;
	jacobian_submatrix_5[1] = dzodr*8.0*M_PI*rl*rl*(dRu5 + l*psi/ri);
	jacobian_submatrix_5[2] = drodz*8.0*M_PI*rl*rl*dZu5;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (dzodr*dr2*8.0*M_PI*phi2*a2*wplOmega/alpha2);

	// This row 3 * dim + IDX(i, j) starts at offset4
	ia[3 * dim + IDX(i, j)] = BASE + offset4;

	// Values.
	aa[offset4 +  0] = +D10*jacobian_submatrix_1[1];
	aa[offset4 +  1] = +D11*jacobian_submatrix_1[1];
	aa[offset4 +  2] = +S11*jacobian_submatrix_1[2];
	aa[offset4 +  3] = +S12*jacobian_submatrix_1[2];
	aa[offset4 +  4] = +S13*jacobian_submatrix_1[2];
	aa[offset4 +  5] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[2];
	aa[offset4 +  6] = +S15*jacobian_submatrix_1[2];
	aa[offset4 +  7] = +D13*jacobian_submatrix_1[1];
	aa[offset4 +  8] = +D14*jacobian_submatrix_1[1];
	aa[offset4 +  9] = +D10*jacobian_submatrix_2[1];
	aa[offset4 + 10] = +D11*jacobian_submatrix_2[1];
	aa[offset4 + 11] = +S11*jacobian_submatrix_2[2];
	aa[offset4 + 12] = +S12*jacobian_submatrix_2[2];
	aa[offset4 + 13] = +S13*jacobian_submatrix_2[2];
	aa[offset4 + 14] = +1.0*jacobian_submatrix_2[0]+S14*jacobian_submatrix_2[2];
	aa[offset4 + 15] = +S15*jacobian_submatrix_2[2];
	aa[offset4 + 16] = +D13*jacobian_submatrix_2[1];
	aa[offset4 + 17] = +D14*jacobian_submatrix_2[1];
	aa[offset4 + 18] = +D10*jacobian_submatrix_3[1];
	aa[offset4 + 19] = +D11*jacobian_submatrix_3[1];
	aa[offset4 + 20] = +S11*jacobian_submatrix_3[2];
	aa[offset4 + 21] = +S12*jacobian_submatrix_3[2];
	aa[offset4 + 22] = +S13*jacobian_submatrix_3[2];
	aa[offset4 + 23] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[2];
	aa[offset4 + 24] = +S15*jacobian_submatrix_3[2];
	aa[offset4 + 25] = +D13*jacobian_submatrix_3[1];
	aa[offset4 + 26] = +D14*jacobian_submatrix_3[1];
	aa[offset4 + 27] = +D20*jacobian_submatrix_4[3];
	aa[offset4 + 28] = +D21*jacobian_submatrix_4[3];
	aa[offset4 + 29] = +S20*jacobian_submatrix_4[4];
	aa[offset4 + 30] = +S21*jacobian_submatrix_4[4];
	aa[offset4 + 31] = +S22*jacobian_submatrix_4[4];
	aa[offset4 + 32] = +S23*jacobian_submatrix_4[4];
	aa[offset4 + 33] = +1.0*jacobian_submatrix_4[0]+D22*jacobian_submatrix_4[3]+S24*jacobian_submatrix_4[4];
	aa[offset4 + 34] = +S25*jacobian_submatrix_4[4];
	aa[offset4 + 35] = +D23*jacobian_submatrix_4[3];
	aa[offset4 + 36] = +D24*jacobian_submatrix_4[3];
	aa[offset4 + 37] = +D10*jacobian_submatrix_5[1];
	aa[offset4 + 38] = +D11*jacobian_submatrix_5[1];
	aa[offset4 + 39] = +S11*jacobian_submatrix_5[2];
	aa[offset4 + 40] = +S12*jacobian_submatrix_5[2];
	aa[offset4 + 41] = +S13*jacobian_submatrix_5[2];
	aa[offset4 + 42] = +1.0*jacobian_submatrix_5[0]+S14*jacobian_submatrix_5[2];
	aa[offset4 + 43] = +S15*jacobian_submatrix_5[2];
	aa[offset4 + 44] = +D13*jacobian_submatrix_5[1];
	aa[offset4 + 45] = +D14*jacobian_submatrix_5[1];
	aa[offset4 + 46] = jacobian_submatrix_w;

	// Columns.
	ja[offset4 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset4 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset4 +  2] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset4 +  3] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset4 +  4] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset4 +  5] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset4 +  6] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset4 +  7] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset4 +  8] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset4 +  9] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset4 + 10] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset4 + 11] = BASE + 1 * dim + IDX(i    , j - 3);
	ja[offset4 + 12] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset4 + 13] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset4 + 14] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset4 + 15] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset4 + 16] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset4 + 17] = BASE + 1 * dim + IDX(i + 2, j    );
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
	ja[offset4 + 46] = BASE + w_idx;


	// CSR CODE FOR GRID NUMBER 5.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = -dzodr*dr2*2.0*a2*wplOmega2*psi/alpha2;
	jacobian_submatrix_1[1] = dzodr*(dRu5 + l*psi/ri);
	jacobian_submatrix_1[2] = drodz*dZu5;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = dzodr*dr2*2.0*l*a2*wplOmega*psi/alpha2;
	jacobian_submatrix_2[1] = 0;
	jacobian_submatrix_2[2] = 0;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = dzodr*dr2*2.0*l*l*psi*lambda/h2;
	jacobian_submatrix_3[1] = dzodr*(dRu5 + l*psi/ri);
	jacobian_submatrix_3[2] = drodz*dZu5;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = -dzodr*dr2*2.0*a2*(m2 - wplOmega2/alpha2)*psi;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*(l*(dRu1/ri + dRu3/ri) - dr2*(a2*(m2 - wplOmega2/alpha2) + l*l*lambda/h2));
	jacobian_submatrix_5[1] = dzodr*(dRu1 + dRu3 + (2.0*l + 1.0)/ri);
	jacobian_submatrix_5[2] = drodz*(dZu1 + dZu3);
	jacobian_submatrix_5[3] = dzodr;
	jacobian_submatrix_5[4] = drodz;

	// Submatrix 6.
	jacobian_submatrix_6[0] = -dzodr*dr2*l*l*psi/h2;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (dzodr*dr2*2.0*a2*wplOmega*psi/alpha2);

	// This row 4 * dim + IDX(i, j) starts at offset5
	ia[4 * dim + IDX(i, j)] = BASE + offset5;

	// Values.
	aa[offset5 +  0] = +D10*jacobian_submatrix_1[1];
	aa[offset5 +  1] = +D11*jacobian_submatrix_1[1];
	aa[offset5 +  2] = +S11*jacobian_submatrix_1[2];
	aa[offset5 +  3] = +S12*jacobian_submatrix_1[2];
	aa[offset5 +  4] = +S13*jacobian_submatrix_1[2];
	aa[offset5 +  5] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[2];
	aa[offset5 +  6] = +S15*jacobian_submatrix_1[2];
	aa[offset5 +  7] = +D13*jacobian_submatrix_1[1];
	aa[offset5 +  8] = +D14*jacobian_submatrix_1[1];
	aa[offset5 +  9] = +1.0*jacobian_submatrix_2[0];
	aa[offset5 + 10] = +D10*jacobian_submatrix_3[1];
	aa[offset5 + 11] = +D11*jacobian_submatrix_3[1];
	aa[offset5 + 12] = +S11*jacobian_submatrix_3[2];
	aa[offset5 + 13] = +S12*jacobian_submatrix_3[2];
	aa[offset5 + 14] = +S13*jacobian_submatrix_3[2];
	aa[offset5 + 15] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[2];
	aa[offset5 + 16] = +S15*jacobian_submatrix_3[2];
	aa[offset5 + 17] = +D13*jacobian_submatrix_3[1];
	aa[offset5 + 18] = +D14*jacobian_submatrix_3[1];
	aa[offset5 + 19] = +1.0*jacobian_submatrix_4[0];
	aa[offset5 + 20] = +D10*jacobian_submatrix_5[1]+D20*jacobian_submatrix_5[3];
	aa[offset5 + 21] = +D11*jacobian_submatrix_5[1]+D21*jacobian_submatrix_5[3];
	aa[offset5 + 22] = +S20*jacobian_submatrix_5[4];
	aa[offset5 + 23] = +S11*jacobian_submatrix_5[2]+S21*jacobian_submatrix_5[4];
	aa[offset5 + 24] = +S12*jacobian_submatrix_5[2]+S22*jacobian_submatrix_5[4];
	aa[offset5 + 25] = +S13*jacobian_submatrix_5[2]+S23*jacobian_submatrix_5[4];
	aa[offset5 + 26] = +1.0*jacobian_submatrix_5[0]+S14*jacobian_submatrix_5[2]+D22*jacobian_submatrix_5[3]+S24*jacobian_submatrix_5[4];
	aa[offset5 + 27] = +S15*jacobian_submatrix_5[2]+S25*jacobian_submatrix_5[4];
	aa[offset5 + 28] = +D13*jacobian_submatrix_5[1]+D23*jacobian_submatrix_5[3];
	aa[offset5 + 29] = +D14*jacobian_submatrix_5[1]+D24*jacobian_submatrix_5[3];
	aa[offset5 + 30] = +1.0*jacobian_submatrix_6[0];
	aa[offset5 + 31] = jacobian_submatrix_w;

	// Columns.
	ja[offset5 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset5 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset5 +  2] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset5 +  3] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset5 +  4] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset5 +  5] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset5 +  6] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset5 +  7] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset5 +  8] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset5 +  9] = BASE + 1 * dim + IDX(i    , j    );
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
	ja[offset5 + 30] = BASE + 5 * dim + IDX(i    , j    );
	ja[offset5 + 31] = BASE + w_idx;


	// CSR CODE FOR GRID NUMBER 6.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*(2.0*dZu2*dZu2*h2*h2/alpha2) + dzodr*(2.0*h2 + r2*lambda)*(2.0*dRu2*dRu2*h2/alpha2);
	jacobian_submatrix_1[1] = dzodr*(-dRu6 + 4.0*dRu1*(lambda + Q1*h2/r2) - (2.0/ri)*(lambda + Q1*h2/r2) - 4.0*dRu3*h2/r2);
	jacobian_submatrix_1[2] = drodz*dZu6;
	jacobian_submatrix_1[3] = dzodr*2.0*(lambda + Q1*h2/r2);
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = 0;
	jacobian_submatrix_2[1] = dzodr*(-2.0*dRu2*h2/alpha2)*(2.0*h2 + r2*lambda);
	jacobian_submatrix_2[2] = drodz*(-2.0*dZu2*h2*h2/alpha2);
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = drodz*(2.0*h2)*(-2.0*dZu2*dZu2*h2/alpha2 + r2*(dZu6 - 2.0*dZu3*lambda)*(dZu6 - 2.0*dZu3*lambda)/(a2_r*a2_r))+ dzodr*((Q1*(dRRu1 + dRu1*(dRu1 - 1.0/ri)) + Q2*(dRRu3 + dRu3*(2.0*dRu3 - 1.0/ri)))*(4.0*h2/r2) - 8.0*dRu1*dRu3*(h2/r2) + 64.0*M_PI*l*h2*rlm1*rlm1*psi*(dRu5/ri) + 32.0*M_PI*h2*rlm1*rlm1*(dRu5*dRu5) + 8.0*h2*r2*lambda*(dRu6/ri)/(a2_r*a2_r) + 2.0*r2*h2*dRu6*dRu6/(a2_r*a2_r) + (dRu2*dRu2)*(-8.0*h2*h2/alpha2 - 2.0*r2*lambda*h2/alpha2) + dr2*h2*(16.0*M_PI*rlm1*rlm1*r2*lambda*m2*psi*psi + 8.0*(lambda/a2_r)*(lambda/a2_r)) + (dRu3/ri)*(-16.0*h2*r2*lambda*lambda/(a2_r*a2_r) - 8.0*h2*r2*lambda*(ri*dRu6)/(a2_r*a2_r)) + (dRu3*dRu3)*(h2/r2)*(-4.0 + 8.0*(h2/a2_r)*(h2/a2_r) - 16.0*h2/a2_r));
	jacobian_submatrix_3[1] = dzodr*(Q2*(8.0*dRu3 - 2.0/ri)*(lambda + h2/r2)*(h2/a2_r) + dRu6*(-5.0*h2 - r2*lambda)/a2_r - 4.0*dRu1*(h2/a2_r)*(lambda + h2/r2) +dRu3*(-12.0*h2*h2/r2 + 4.0*r2*lambda*lambda)/a2_r + lambda*(-6.0*h2/ri + 2.0*r2*lambda/ri)/a2_r);
	jacobian_submatrix_3[2] = drodz*(8.0*dZu3*lambda*h2 + dZu6*(-3.0*h2 + r2*lambda))/a2_r;
	jacobian_submatrix_3[3] = dzodr*2.0*(lambda + Q2*h2/r2);
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = 0;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*(16.0*a2_r*M_PI*rlm1*rlm1)*(dr2*r2*lambda*m2*psi + 2.0*l*dRu5/ri);
	jacobian_submatrix_5[1] = dzodr*(32.0*a2_r*M_PI*rlm1*rlm1)*(l*psi/ri + dRu5);
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = drodz*(r2*dZu6 + 2.0*h2*dZu3)*(r2*dZu6 + 2.0*h2*dZu3)/(a2_r*a2_r) + dzodr*(2.0*dRRu1 + 2.0*dRRu3 + 2.0*dRu1*(dRu1 - 1.0/ri) - r2*h2*dRu2*dRu2/alpha2 + 16.0*M_PI*rl*rl*dRu5*dRu5 + 32.0*M_PI*l*rl*rl*psi*(dRu5/ri) + (r2*dRu6/a2_r)*(r2*dRu6/a2_r) + (dRu3*dRu3)*(2.0 + 4.0*(h2/a2_r)*(h2/a2_r)) + dr2*(r2*lambda)*(8.0*M_PI*m2*phi2 + 4.0*lambda/(a2_r*a2_r)) + (dRu6/ri)*(4.0*r2)*(-h2/(a2_r*a2_r)) + (dRu3/ri)*(-6.0*(h2/a2_r)*(h2/a2_r) + 4.0*h2*(ri*r2*dRu6)/(a2_r*a2_r) - 2.0*(r2*lambda/a2_r)*(r2*lambda/a2_r) + 4.0*r2*lambda/a2_r) + dr2*(-8.0*lambda/a2_r + 8.0*M_PI*a2_r*m2*phi2));
	jacobian_submatrix_6[1] = dzodr*((3.0*h2 - r2*lambda)/ri - dRu1*(h2 + r2*lambda) - dRu3*(5.0*h2 + r2*lambda) - 2.0*r2*dRu6)/a2_r;
	jacobian_submatrix_6[2] = drodz*(-2.0*r2*dZu6 + dZu1*(h2 + r2*lambda) + dZu3*(-3.0*h2 + r2*lambda))/a2_r;
	jacobian_submatrix_6[3] = dzodr;
	jacobian_submatrix_6[4] = drodz;

	// Omega term.
	jacobian_submatrix_w = 0;

	// This row 5 * dim + IDX(i, j) starts at offset6
	ia[5 * dim + IDX(i, j)] = BASE + offset6;

	// Values.
	aa[offset6 +  0] = +D10*jacobian_submatrix_1[1]+D20*jacobian_submatrix_1[3];
	aa[offset6 +  1] = +D11*jacobian_submatrix_1[1]+D21*jacobian_submatrix_1[3];
	aa[offset6 +  2] = +S11*jacobian_submatrix_1[2];
	aa[offset6 +  3] = +S12*jacobian_submatrix_1[2];
	aa[offset6 +  4] = +S13*jacobian_submatrix_1[2];
	aa[offset6 +  5] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[2]+D22*jacobian_submatrix_1[3];
	aa[offset6 +  6] = +S15*jacobian_submatrix_1[2];
	aa[offset6 +  7] = +D13*jacobian_submatrix_1[1]+D23*jacobian_submatrix_1[3];
	aa[offset6 +  8] = +D14*jacobian_submatrix_1[1]+D24*jacobian_submatrix_1[3];
	aa[offset6 +  9] = +D10*jacobian_submatrix_2[1];
	aa[offset6 + 10] = +D11*jacobian_submatrix_2[1];
	aa[offset6 + 11] = +S11*jacobian_submatrix_2[2];
	aa[offset6 + 12] = +S12*jacobian_submatrix_2[2];
	aa[offset6 + 13] = +S13*jacobian_submatrix_2[2];
	aa[offset6 + 14] = +S14*jacobian_submatrix_2[2];
	aa[offset6 + 15] = +S15*jacobian_submatrix_2[2];
	aa[offset6 + 16] = +D13*jacobian_submatrix_2[1];
	aa[offset6 + 17] = +D14*jacobian_submatrix_2[1];
	aa[offset6 + 18] = +D10*jacobian_submatrix_3[1]+D20*jacobian_submatrix_3[3];
	aa[offset6 + 19] = +D11*jacobian_submatrix_3[1]+D21*jacobian_submatrix_3[3];
	aa[offset6 + 20] = +S11*jacobian_submatrix_3[2];
	aa[offset6 + 21] = +S12*jacobian_submatrix_3[2];
	aa[offset6 + 22] = +S13*jacobian_submatrix_3[2];
	aa[offset6 + 23] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[2]+D22*jacobian_submatrix_3[3];
	aa[offset6 + 24] = +S15*jacobian_submatrix_3[2];
	aa[offset6 + 25] = +D13*jacobian_submatrix_3[1]+D23*jacobian_submatrix_3[3];
	aa[offset6 + 26] = +D14*jacobian_submatrix_3[1]+D24*jacobian_submatrix_3[3];
	aa[offset6 + 27] = +D10*jacobian_submatrix_5[1];
	aa[offset6 + 28] = +D11*jacobian_submatrix_5[1];
	aa[offset6 + 29] = +1.0*jacobian_submatrix_5[0];
	aa[offset6 + 30] = +D13*jacobian_submatrix_5[1];
	aa[offset6 + 31] = +D14*jacobian_submatrix_5[1];
	aa[offset6 + 32] = +D10*jacobian_submatrix_6[1]+D20*jacobian_submatrix_6[3];
	aa[offset6 + 33] = +D11*jacobian_submatrix_6[1]+D21*jacobian_submatrix_6[3];
	aa[offset6 + 34] = +S20*jacobian_submatrix_6[4];
	aa[offset6 + 35] = +S11*jacobian_submatrix_6[2]+S21*jacobian_submatrix_6[4];
	aa[offset6 + 36] = +S12*jacobian_submatrix_6[2]+S22*jacobian_submatrix_6[4];
	aa[offset6 + 37] = +S13*jacobian_submatrix_6[2]+S23*jacobian_submatrix_6[4];
	aa[offset6 + 38] = +1.0*jacobian_submatrix_6[0]+S14*jacobian_submatrix_6[2]+D22*jacobian_submatrix_6[3]+S24*jacobian_submatrix_6[4];
	aa[offset6 + 39] = +S15*jacobian_submatrix_6[2]+S25*jacobian_submatrix_6[4];
	aa[offset6 + 40] = +D13*jacobian_submatrix_6[1]+D23*jacobian_submatrix_6[3];
	aa[offset6 + 41] = +D14*jacobian_submatrix_6[1]+D24*jacobian_submatrix_6[3];

	// Columns.
	ja[offset6 +  0] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset6 +  1] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset6 +  2] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset6 +  3] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset6 +  4] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset6 +  5] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset6 +  6] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset6 +  7] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset6 +  8] = BASE + 0 * dim + IDX(i + 2, j    );
	ja[offset6 +  9] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset6 + 10] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset6 + 11] = BASE + 1 * dim + IDX(i    , j - 3);
	ja[offset6 + 12] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset6 + 13] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset6 + 14] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset6 + 15] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset6 + 16] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset6 + 17] = BASE + 1 * dim + IDX(i + 2, j    );
	ja[offset6 + 18] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset6 + 19] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset6 + 20] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset6 + 21] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset6 + 22] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset6 + 23] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset6 + 24] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset6 + 25] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset6 + 26] = BASE + 2 * dim + IDX(i + 2, j    );
	ja[offset6 + 27] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset6 + 28] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset6 + 29] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset6 + 30] = BASE + 4 * dim + IDX(i + 1, j    );
	ja[offset6 + 31] = BASE + 4 * dim + IDX(i + 2, j    );
	ja[offset6 + 32] = BASE + 5 * dim + IDX(i - 2, j    );
	ja[offset6 + 33] = BASE + 5 * dim + IDX(i - 1, j    );
	ja[offset6 + 34] = BASE + 5 * dim + IDX(i    , j - 4);
	ja[offset6 + 35] = BASE + 5 * dim + IDX(i    , j - 3);
	ja[offset6 + 36] = BASE + 5 * dim + IDX(i    , j - 2);
	ja[offset6 + 37] = BASE + 5 * dim + IDX(i    , j - 1);
	ja[offset6 + 38] = BASE + 5 * dim + IDX(i    , j    );
	ja[offset6 + 39] = BASE + 5 * dim + IDX(i    , j + 1);
	ja[offset6 + 40] = BASE + 5 * dim + IDX(i + 1, j    );
	ja[offset6 + 41] = BASE + 5 * dim + IDX(i + 2, j    );


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
	const MKL_INT ghost,	// Number of ghost zones.
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
	const double u602, const double u612, const double u622, const double u632, const double u640, const double u641, const double u642, const double u643, const double u644, const double u652,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5, 	// Number of elements filled before filling function 5
	const MKL_INT offset6 	// Number of elements filled before filling function 5
)
{
	// Grid variables.
	double u1 = u142;
	double u2 = u242;
	double u3 = u342;
	double u4 = u442;
	double u5 = u542;
	double u6 = u642;
    
	// Physical names for readability.
	double alpha = exp(u1);
	double Omega = u2;
	double h = exp(u3);
	double a = exp(u4);
	double psi = u5;
	double lambda = u6;
    
	// Coordinates.
	double ri = (double)i + 0.5 - ghost;
	double r = ri * dr;
	double r2 = r * r;
    
	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	double dr2 = dr * dr;
    
	// Scalar field mass and frequency.
	double w = omega_calc(xi, m);
	double m2 = m * m;
	// Omega variable index position.
	MKL_INT w_idx = GNUM * dim;
    
	// Short hands.
	// Scalar field.
	double rlm1 = (l == 1) ? 1.0 : pow(r, l - 1);
	double rl = rlm1 * r;
	double phior = rlm1 * psi;
	double phi = r * phior;
	double phi2or2 = phior * phior;
	double phi2 = phi * phi;
	// Shift combined with scalar field rotation and frequency.
	double wplOmega = w + l * Omega;
	double wplOmega2 = wplOmega * wplOmega;
	// Squared variables.
	double alpha2 = alpha * alpha;
	double h2 = h * h;
	double a2 = a * a;
	// Regularization a2.
	double a2_r = h2 + r2 * lambda;
    
	// Finite differences.
	// Axial derivatives.
	double dRu1 = S11 * u112 + S12 * u122 + S13 * u132 + S14 * u142 + S15 * u152;
	double dRu2 = S11 * u212 + S12 * u222 + S13 * u232 + S14 * u242 + S15 * u252;
	double dRu3 = S11 * u312 + S12 * u322 + S13 * u332 + S14 * u342 + S15 * u352;
	//double dRu4 = S11 * u412 + S12 * u422 + S13 * u432 + S14 * u442 + S15 * u452;
	double dRu5 = S11 * u512 + S12 * u522 + S13 * u532 + S14 * u542 + S15 * u552;
	double dRu6 = S11 * u612 + S12 * u622 + S13 * u632 + S14 * u642 + S15 * u652;
	// Z derivatives.
	double dZu1 = D10 * u140 + D11 * u141 + D13 * u143 + D14 * u144;
	double dZu2 = D10 * u240 + D11 * u241 + D13 * u243 + D14 * u244;
	double dZu3 = D10 * u340 + D11 * u341 + D13 * u343 + D14 * u344;
	//double dZu4 = D10 * u440 + D11 * u441 + D13 * u443 + D14 * u444;
	double dZu5 = D10 * u540 + D11 * u541 + D13 * u543 + D14 * u544;
	double dZu6 = D10 * u640 + D11 * u641 + D13 * u643 + D14 * u644;
	// Second derivatives.
	double dRRu1 = S20 * u102 + S21 * u112 + S22 * u122 + S23 * u132 + S24 * u142 + S25 * u152;
	double dRRu3 = S20 * u302 + S21 * u312 + S22 * u322 + S23 * u332 + S24 * u342 + S25 * u352;
    
	// Declare Jacobian submatrices.
	double jacobian_submatrix_1[5] = { 0.0 };
	double jacobian_submatrix_2[5] = { 0.0 };
	double jacobian_submatrix_3[5] = { 0.0 };
	double jacobian_submatrix_4[5] = { 0.0 };
	double jacobian_submatrix_5[5] = { 0.0 };
	double jacobian_submatrix_6[5] = { 0.0 };
	double jacobian_submatrix_w = 0.0;

	// scR CODE FOR GRID NUMBER 1.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*r2*h2*dZu2*dZu2/alpha2 + dzodr*(r2*h2*dRu2*dRu2/alpha2 + 16.0*M_PI*dr2*a2*phi2*wplOmega2/alpha2);
	jacobian_submatrix_1[1] = dzodr*(2.0*dRu1 + dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = drodz*(2.0*dZu1 + dZu3);
	jacobian_submatrix_1[3] = dzodr;
	jacobian_submatrix_1[4] = drodz;

	// Submatrix 2.
	jacobian_submatrix_2[0] = -dzodr*dr2*16.0*M_PI*l*a2*phi2*wplOmega/alpha2;
	jacobian_submatrix_2[1] = -dzodr*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = -drodz*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = -dzodr*r2*h2*(dRu2*dRu2 + drodz*drodz*dZu2*dZu2)/alpha2;
	jacobian_submatrix_3[1] = dzodr*dRu1;
	jacobian_submatrix_3[2] = drodz*dZu1;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*a2*(m2 - 2.0*wplOmega2/alpha2)*phi*rl;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (-dzodr*dr2*16.0*M_PI*a2*phi2*wplOmega/alpha2);

	// This row 0 * dim + IDX(i, j) starts at offset1
	ia[0 * dim + IDX(i, j)] = BASE + offset1;

	// Values.
	aa[offset1 +  0] = +S20*jacobian_submatrix_1[3];
	aa[offset1 +  1] = +S11*jacobian_submatrix_1[1]+S21*jacobian_submatrix_1[3];
	aa[offset1 +  2] = +S12*jacobian_submatrix_1[1]+S22*jacobian_submatrix_1[3];
	aa[offset1 +  3] = +S13*jacobian_submatrix_1[1]+S23*jacobian_submatrix_1[3];
	aa[offset1 +  4] = +D10*jacobian_submatrix_1[2]+D20*jacobian_submatrix_1[4];
	aa[offset1 +  5] = +D11*jacobian_submatrix_1[2]+D21*jacobian_submatrix_1[4];
	aa[offset1 +  6] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[1]+S24*jacobian_submatrix_1[3]+D22*jacobian_submatrix_1[4];
	aa[offset1 +  7] = +D13*jacobian_submatrix_1[2]+D23*jacobian_submatrix_1[4];
	aa[offset1 +  8] = +D14*jacobian_submatrix_1[2]+D24*jacobian_submatrix_1[4];
	aa[offset1 +  9] = +S15*jacobian_submatrix_1[1]+S25*jacobian_submatrix_1[3];
	aa[offset1 + 10] = +S11*jacobian_submatrix_2[1];
	aa[offset1 + 11] = +S12*jacobian_submatrix_2[1];
	aa[offset1 + 12] = +S13*jacobian_submatrix_2[1];
	aa[offset1 + 13] = +D10*jacobian_submatrix_2[2];
	aa[offset1 + 14] = +D11*jacobian_submatrix_2[2];
	aa[offset1 + 15] = +1.0*jacobian_submatrix_2[0]+S14*jacobian_submatrix_2[1];
	aa[offset1 + 16] = +D13*jacobian_submatrix_2[2];
	aa[offset1 + 17] = +D14*jacobian_submatrix_2[2];
	aa[offset1 + 18] = +S15*jacobian_submatrix_2[1];
	aa[offset1 + 19] = +S11*jacobian_submatrix_3[1];
	aa[offset1 + 20] = +S12*jacobian_submatrix_3[1];
	aa[offset1 + 21] = +S13*jacobian_submatrix_3[1];
	aa[offset1 + 22] = +D10*jacobian_submatrix_3[2];
	aa[offset1 + 23] = +D11*jacobian_submatrix_3[2];
	aa[offset1 + 24] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1];
	aa[offset1 + 25] = +D13*jacobian_submatrix_3[2];
	aa[offset1 + 26] = +D14*jacobian_submatrix_3[2];
	aa[offset1 + 27] = +S15*jacobian_submatrix_3[1];
	aa[offset1 + 28] = +1.0*jacobian_submatrix_4[0];
	aa[offset1 + 29] = +1.0*jacobian_submatrix_5[0];
	aa[offset1 + 30] = jacobian_submatrix_w;

	// Columns.
	ja[offset1 +  0] = BASE + 0 * dim + IDX(i - 4, j    );
	ja[offset1 +  1] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset1 +  2] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset1 +  3] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset1 +  4] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset1 +  5] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset1 +  6] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset1 +  7] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset1 +  8] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset1 +  9] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset1 + 10] = BASE + 1 * dim + IDX(i - 3, j    );
	ja[offset1 + 11] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset1 + 12] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset1 + 13] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset1 + 14] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset1 + 15] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset1 + 16] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset1 + 17] = BASE + 1 * dim + IDX(i    , j + 2);
	ja[offset1 + 18] = BASE + 1 * dim + IDX(i + 1, j    );
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
	ja[offset1 + 30] = BASE + w_idx;


	// scR CODE FOR GRID NUMBER 2.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = 0;
	jacobian_submatrix_1[1] = -dzodr*dRu2;
	jacobian_submatrix_1[2] = -drodz*dZu2;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = -dzodr*dr2*16.0*M_PI*l*l*a2*phi2or2/h2;
	jacobian_submatrix_2[1] = dzodr*(-dRu1 + 3.0*dRu3 + 3.0/ri);
	jacobian_submatrix_2[2] = drodz*(-dZu1 + 3.0*dZu3);
	jacobian_submatrix_2[3] = dzodr;
	jacobian_submatrix_2[4] = drodz;

	// Submatrix 3.
	jacobian_submatrix_3[0] = dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phi2or2/h2;
	jacobian_submatrix_3[1] = dzodr*3.0*dRu2;
	jacobian_submatrix_3[2] = drodz*3.0*dZu2;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = -dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phi2or2/h2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = -dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phior*rlm1/h2;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (-dzodr*dr2*16.0*M_PI*l*a2*phi2or2/h2);

	// This row 1 * dim + IDX(i, j) starts at offset2
	ia[1 * dim + IDX(i, j)] = BASE + offset2;

	// Values.
	aa[offset2 +  0] = +S11*jacobian_submatrix_1[1];
	aa[offset2 +  1] = +S12*jacobian_submatrix_1[1];
	aa[offset2 +  2] = +S13*jacobian_submatrix_1[1];
	aa[offset2 +  3] = +D10*jacobian_submatrix_1[2];
	aa[offset2 +  4] = +D11*jacobian_submatrix_1[2];
	aa[offset2 +  5] = +S14*jacobian_submatrix_1[1];
	aa[offset2 +  6] = +D13*jacobian_submatrix_1[2];
	aa[offset2 +  7] = +D14*jacobian_submatrix_1[2];
	aa[offset2 +  8] = +S15*jacobian_submatrix_1[1];
	aa[offset2 +  9] = +S20*jacobian_submatrix_2[3];
	aa[offset2 + 10] = +S11*jacobian_submatrix_2[1]+S21*jacobian_submatrix_2[3];
	aa[offset2 + 11] = +S12*jacobian_submatrix_2[1]+S22*jacobian_submatrix_2[3];
	aa[offset2 + 12] = +S13*jacobian_submatrix_2[1]+S23*jacobian_submatrix_2[3];
	aa[offset2 + 13] = +D10*jacobian_submatrix_2[2]+D20*jacobian_submatrix_2[4];
	aa[offset2 + 14] = +D11*jacobian_submatrix_2[2]+D21*jacobian_submatrix_2[4];
	aa[offset2 + 15] = +1.0*jacobian_submatrix_2[0]+S14*jacobian_submatrix_2[1]+S24*jacobian_submatrix_2[3]+D22*jacobian_submatrix_2[4];
	aa[offset2 + 16] = +D13*jacobian_submatrix_2[2]+D23*jacobian_submatrix_2[4];
	aa[offset2 + 17] = +D14*jacobian_submatrix_2[2]+D24*jacobian_submatrix_2[4];
	aa[offset2 + 18] = +S15*jacobian_submatrix_2[1]+S25*jacobian_submatrix_2[3];
	aa[offset2 + 19] = +S11*jacobian_submatrix_3[1];
	aa[offset2 + 20] = +S12*jacobian_submatrix_3[1];
	aa[offset2 + 21] = +S13*jacobian_submatrix_3[1];
	aa[offset2 + 22] = +D10*jacobian_submatrix_3[2];
	aa[offset2 + 23] = +D11*jacobian_submatrix_3[2];
	aa[offset2 + 24] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1];
	aa[offset2 + 25] = +D13*jacobian_submatrix_3[2];
	aa[offset2 + 26] = +D14*jacobian_submatrix_3[2];
	aa[offset2 + 27] = +S15*jacobian_submatrix_3[1];
	aa[offset2 + 28] = +1.0*jacobian_submatrix_4[0];
	aa[offset2 + 29] = +1.0*jacobian_submatrix_5[0];
	aa[offset2 + 30] = jacobian_submatrix_w;

	// Columns.
	ja[offset2 +  0] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset2 +  1] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset2 +  2] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset2 +  3] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset2 +  4] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset2 +  5] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset2 +  6] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset2 +  7] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset2 +  8] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset2 +  9] = BASE + 1 * dim + IDX(i - 4, j    );
	ja[offset2 + 10] = BASE + 1 * dim + IDX(i - 3, j    );
	ja[offset2 + 11] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset2 + 12] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset2 + 13] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset2 + 14] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset2 + 15] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset2 + 16] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset2 + 17] = BASE + 1 * dim + IDX(i    , j + 2);
	ja[offset2 + 18] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset2 + 19] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset2 + 20] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset2 + 21] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset2 + 22] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset2 + 23] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset2 + 24] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset2 + 25] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset2 + 26] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset2 + 27] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset2 + 28] = BASE + 3 * dim + IDX(i    , j    );
	ja[offset2 + 29] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset2 + 30] = BASE + w_idx;


	// scR CODE FOR GRID NUMBER 3.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = -dzodr*r2*h2*dRu2*dRu2/alpha2 - drodz*r2*h2*dZu2*dZu2/alpha2;
	jacobian_submatrix_1[1] = dzodr*(dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = drodz*dZu3;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = 0;
	jacobian_submatrix_2[1] = dzodr*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = drodz*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = drodz*r2*h2*dZu2*dZu2/alpha2 + dzodr*(r2*h2*dRu2*dRu2/alpha2 - dr2*16.0*M_PI*l*l*a2*phi2or2/h2);
	jacobian_submatrix_3[1] = dzodr*(dRu1 + 2.0*dRu3 + 2.0/ri);
	jacobian_submatrix_3[2] = drodz*(dZu1 + 2.0*dZu3);
	jacobian_submatrix_3[3] = dzodr;
	jacobian_submatrix_3[4] = drodz;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*a2*(r2*m2 + 2.0*l*l/h2)*phior*rlm1;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = 0;

	// This row 2 * dim + IDX(i, j) starts at offset3
	ia[2 * dim + IDX(i, j)] = BASE + offset3;

	// Values.
	aa[offset3 +  0] = +S11*jacobian_submatrix_1[1];
	aa[offset3 +  1] = +S12*jacobian_submatrix_1[1];
	aa[offset3 +  2] = +S13*jacobian_submatrix_1[1];
	aa[offset3 +  3] = +D10*jacobian_submatrix_1[2];
	aa[offset3 +  4] = +D11*jacobian_submatrix_1[2];
	aa[offset3 +  5] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[1];
	aa[offset3 +  6] = +D13*jacobian_submatrix_1[2];
	aa[offset3 +  7] = +D14*jacobian_submatrix_1[2];
	aa[offset3 +  8] = +S15*jacobian_submatrix_1[1];
	aa[offset3 +  9] = +S11*jacobian_submatrix_2[1];
	aa[offset3 + 10] = +S12*jacobian_submatrix_2[1];
	aa[offset3 + 11] = +S13*jacobian_submatrix_2[1];
	aa[offset3 + 12] = +D10*jacobian_submatrix_2[2];
	aa[offset3 + 13] = +D11*jacobian_submatrix_2[2];
	aa[offset3 + 14] = +S14*jacobian_submatrix_2[1];
	aa[offset3 + 15] = +D13*jacobian_submatrix_2[2];
	aa[offset3 + 16] = +D14*jacobian_submatrix_2[2];
	aa[offset3 + 17] = +S15*jacobian_submatrix_2[1];
	aa[offset3 + 18] = +S20*jacobian_submatrix_3[3];
	aa[offset3 + 19] = +S11*jacobian_submatrix_3[1]+S21*jacobian_submatrix_3[3];
	aa[offset3 + 20] = +S12*jacobian_submatrix_3[1]+S22*jacobian_submatrix_3[3];
	aa[offset3 + 21] = +S13*jacobian_submatrix_3[1]+S23*jacobian_submatrix_3[3];
	aa[offset3 + 22] = +D10*jacobian_submatrix_3[2]+D20*jacobian_submatrix_3[4];
	aa[offset3 + 23] = +D11*jacobian_submatrix_3[2]+D21*jacobian_submatrix_3[4];
	aa[offset3 + 24] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1]+S24*jacobian_submatrix_3[3]+D22*jacobian_submatrix_3[4];
	aa[offset3 + 25] = +D13*jacobian_submatrix_3[2]+D23*jacobian_submatrix_3[4];
	aa[offset3 + 26] = +D14*jacobian_submatrix_3[2]+D24*jacobian_submatrix_3[4];
	aa[offset3 + 27] = +S15*jacobian_submatrix_3[1]+S25*jacobian_submatrix_3[3];
	aa[offset3 + 28] = +1.0*jacobian_submatrix_4[0];
	aa[offset3 + 29] = +1.0*jacobian_submatrix_5[0];

	// Columns.
	ja[offset3 +  0] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset3 +  1] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset3 +  2] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset3 +  3] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset3 +  4] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset3 +  5] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset3 +  6] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset3 +  7] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset3 +  8] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset3 +  9] = BASE + 1 * dim + IDX(i - 3, j    );
	ja[offset3 + 10] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset3 + 11] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset3 + 12] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset3 + 13] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset3 + 14] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset3 + 15] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset3 + 16] = BASE + 1 * dim + IDX(i    , j + 2);
	ja[offset3 + 17] = BASE + 1 * dim + IDX(i + 1, j    );
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


	// scR CODE FOR GRID NUMBER 4.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*0.5*r2*h2*dZu2*dZu2/alpha2 + dzodr*(0.5*r2*h2*dRu2*dRu2/alpha2 - dr2*8.0*M_PI*a2*wplOmega2*phi2/alpha2);
	jacobian_submatrix_1[1] = -dzodr*(dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = -drodz*dZu3;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = dzodr*dr2*8.0*M_PI*l*a2*wplOmega*phi2/alpha2;
	jacobian_submatrix_2[1] = -dzodr*0.5*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = -drodz*0.5*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = -drodz*0.5*r2*h2*dZu2*dZu2/alpha2 + dzodr*(-0.5*r2*h2*dRu2*dRu2/alpha2 + dr2*8.0*l*l*M_PI*phi2or2*a2/h2);
	jacobian_submatrix_3[1] = -dzodr*dRu1;
	jacobian_submatrix_3[2] = -drodz*dZu1;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(-l*l*phi2or2/h2 + wplOmega2*phi2/alpha2);
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = dzodr;
	jacobian_submatrix_4[4] = drodz;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*(l*ri*dRu5 + l*l*psi - l*l*a2*psi/h2 + r2*a2*wplOmega2*psi/alpha2)*rlm1*rlm1;
	jacobian_submatrix_5[1] = dzodr*8.0*M_PI*rl*rl*(dRu5 + l*psi/ri);
	jacobian_submatrix_5[2] = drodz*8.0*M_PI*rl*rl*dZu5;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (dzodr*dr2*8.0*M_PI*phi2*a2*wplOmega/alpha2);

	// This row 3 * dim + IDX(i, j) starts at offset4
	ia[3 * dim + IDX(i, j)] = BASE + offset4;

	// Values.
	aa[offset4 +  0] = +S11*jacobian_submatrix_1[1];
	aa[offset4 +  1] = +S12*jacobian_submatrix_1[1];
	aa[offset4 +  2] = +S13*jacobian_submatrix_1[1];
	aa[offset4 +  3] = +D10*jacobian_submatrix_1[2];
	aa[offset4 +  4] = +D11*jacobian_submatrix_1[2];
	aa[offset4 +  5] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[1];
	aa[offset4 +  6] = +D13*jacobian_submatrix_1[2];
	aa[offset4 +  7] = +D14*jacobian_submatrix_1[2];
	aa[offset4 +  8] = +S15*jacobian_submatrix_1[1];
	aa[offset4 +  9] = +S11*jacobian_submatrix_2[1];
	aa[offset4 + 10] = +S12*jacobian_submatrix_2[1];
	aa[offset4 + 11] = +S13*jacobian_submatrix_2[1];
	aa[offset4 + 12] = +D10*jacobian_submatrix_2[2];
	aa[offset4 + 13] = +D11*jacobian_submatrix_2[2];
	aa[offset4 + 14] = +1.0*jacobian_submatrix_2[0]+S14*jacobian_submatrix_2[1];
	aa[offset4 + 15] = +D13*jacobian_submatrix_2[2];
	aa[offset4 + 16] = +D14*jacobian_submatrix_2[2];
	aa[offset4 + 17] = +S15*jacobian_submatrix_2[1];
	aa[offset4 + 18] = +S11*jacobian_submatrix_3[1];
	aa[offset4 + 19] = +S12*jacobian_submatrix_3[1];
	aa[offset4 + 20] = +S13*jacobian_submatrix_3[1];
	aa[offset4 + 21] = +D10*jacobian_submatrix_3[2];
	aa[offset4 + 22] = +D11*jacobian_submatrix_3[2];
	aa[offset4 + 23] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1];
	aa[offset4 + 24] = +D13*jacobian_submatrix_3[2];
	aa[offset4 + 25] = +D14*jacobian_submatrix_3[2];
	aa[offset4 + 26] = +S15*jacobian_submatrix_3[1];
	aa[offset4 + 27] = +S20*jacobian_submatrix_4[3];
	aa[offset4 + 28] = +S21*jacobian_submatrix_4[3];
	aa[offset4 + 29] = +S22*jacobian_submatrix_4[3];
	aa[offset4 + 30] = +S23*jacobian_submatrix_4[3];
	aa[offset4 + 31] = +D20*jacobian_submatrix_4[4];
	aa[offset4 + 32] = +D21*jacobian_submatrix_4[4];
	aa[offset4 + 33] = +1.0*jacobian_submatrix_4[0]+S24*jacobian_submatrix_4[3]+D22*jacobian_submatrix_4[4];
	aa[offset4 + 34] = +D23*jacobian_submatrix_4[4];
	aa[offset4 + 35] = +D24*jacobian_submatrix_4[4];
	aa[offset4 + 36] = +S25*jacobian_submatrix_4[3];
	aa[offset4 + 37] = +S11*jacobian_submatrix_5[1];
	aa[offset4 + 38] = +S12*jacobian_submatrix_5[1];
	aa[offset4 + 39] = +S13*jacobian_submatrix_5[1];
	aa[offset4 + 40] = +D10*jacobian_submatrix_5[2];
	aa[offset4 + 41] = +D11*jacobian_submatrix_5[2];
	aa[offset4 + 42] = +1.0*jacobian_submatrix_5[0]+S14*jacobian_submatrix_5[1];
	aa[offset4 + 43] = +D13*jacobian_submatrix_5[2];
	aa[offset4 + 44] = +D14*jacobian_submatrix_5[2];
	aa[offset4 + 45] = +S15*jacobian_submatrix_5[1];
	aa[offset4 + 46] = jacobian_submatrix_w;

	// Columns.
	ja[offset4 +  0] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset4 +  1] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset4 +  2] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset4 +  3] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset4 +  4] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset4 +  5] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset4 +  6] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset4 +  7] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset4 +  8] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset4 +  9] = BASE + 1 * dim + IDX(i - 3, j    );
	ja[offset4 + 10] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset4 + 11] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset4 + 12] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset4 + 13] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset4 + 14] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset4 + 15] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset4 + 16] = BASE + 1 * dim + IDX(i    , j + 2);
	ja[offset4 + 17] = BASE + 1 * dim + IDX(i + 1, j    );
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
	ja[offset4 + 46] = BASE + w_idx;


	// scR CODE FOR GRID NUMBER 5.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = -dzodr*dr2*2.0*a2*wplOmega2*psi/alpha2;
	jacobian_submatrix_1[1] = dzodr*(dRu5 + l*psi/ri);
	jacobian_submatrix_1[2] = drodz*dZu5;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = dzodr*dr2*2.0*l*a2*wplOmega*psi/alpha2;
	jacobian_submatrix_2[1] = 0;
	jacobian_submatrix_2[2] = 0;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = dzodr*dr2*2.0*l*l*psi*lambda/h2;
	jacobian_submatrix_3[1] = dzodr*(dRu5 + l*psi/ri);
	jacobian_submatrix_3[2] = drodz*dZu5;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = -dzodr*dr2*2.0*a2*(m2 - wplOmega2/alpha2)*psi;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*(l*(dRu1/ri + dRu3/ri) - dr2*(a2*(m2 - wplOmega2/alpha2) + l*l*lambda/h2));
	jacobian_submatrix_5[1] = dzodr*(dRu1 + dRu3 + (2.0*l + 1.0)/ri);
	jacobian_submatrix_5[2] = drodz*(dZu1 + dZu3);
	jacobian_submatrix_5[3] = dzodr;
	jacobian_submatrix_5[4] = drodz;

	// Submatrix 6.
	jacobian_submatrix_6[0] = -dzodr*dr2*l*l*psi/h2;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (dzodr*dr2*2.0*a2*wplOmega*psi/alpha2);

	// This row 4 * dim + IDX(i, j) starts at offset5
	ia[4 * dim + IDX(i, j)] = BASE + offset5;

	// Values.
	aa[offset5 +  0] = +S11*jacobian_submatrix_1[1];
	aa[offset5 +  1] = +S12*jacobian_submatrix_1[1];
	aa[offset5 +  2] = +S13*jacobian_submatrix_1[1];
	aa[offset5 +  3] = +D10*jacobian_submatrix_1[2];
	aa[offset5 +  4] = +D11*jacobian_submatrix_1[2];
	aa[offset5 +  5] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[1];
	aa[offset5 +  6] = +D13*jacobian_submatrix_1[2];
	aa[offset5 +  7] = +D14*jacobian_submatrix_1[2];
	aa[offset5 +  8] = +S15*jacobian_submatrix_1[1];
	aa[offset5 +  9] = +1.0*jacobian_submatrix_2[0];
	aa[offset5 + 10] = +S11*jacobian_submatrix_3[1];
	aa[offset5 + 11] = +S12*jacobian_submatrix_3[1];
	aa[offset5 + 12] = +S13*jacobian_submatrix_3[1];
	aa[offset5 + 13] = +D10*jacobian_submatrix_3[2];
	aa[offset5 + 14] = +D11*jacobian_submatrix_3[2];
	aa[offset5 + 15] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1];
	aa[offset5 + 16] = +D13*jacobian_submatrix_3[2];
	aa[offset5 + 17] = +D14*jacobian_submatrix_3[2];
	aa[offset5 + 18] = +S15*jacobian_submatrix_3[1];
	aa[offset5 + 19] = +1.0*jacobian_submatrix_4[0];
	aa[offset5 + 20] = +S20*jacobian_submatrix_5[3];
	aa[offset5 + 21] = +S11*jacobian_submatrix_5[1]+S21*jacobian_submatrix_5[3];
	aa[offset5 + 22] = +S12*jacobian_submatrix_5[1]+S22*jacobian_submatrix_5[3];
	aa[offset5 + 23] = +S13*jacobian_submatrix_5[1]+S23*jacobian_submatrix_5[3];
	aa[offset5 + 24] = +D10*jacobian_submatrix_5[2]+D20*jacobian_submatrix_5[4];
	aa[offset5 + 25] = +D11*jacobian_submatrix_5[2]+D21*jacobian_submatrix_5[4];
	aa[offset5 + 26] = +1.0*jacobian_submatrix_5[0]+S14*jacobian_submatrix_5[1]+S24*jacobian_submatrix_5[3]+D22*jacobian_submatrix_5[4];
	aa[offset5 + 27] = +D13*jacobian_submatrix_5[2]+D23*jacobian_submatrix_5[4];
	aa[offset5 + 28] = +D14*jacobian_submatrix_5[2]+D24*jacobian_submatrix_5[4];
	aa[offset5 + 29] = +S15*jacobian_submatrix_5[1]+S25*jacobian_submatrix_5[3];
	aa[offset5 + 30] = +1.0*jacobian_submatrix_6[0];
	aa[offset5 + 31] = jacobian_submatrix_w;

	// Columns.
	ja[offset5 +  0] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset5 +  1] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset5 +  2] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset5 +  3] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset5 +  4] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset5 +  5] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset5 +  6] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset5 +  7] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset5 +  8] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset5 +  9] = BASE + 1 * dim + IDX(i    , j    );
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
	ja[offset5 + 30] = BASE + 5 * dim + IDX(i    , j    );
	ja[offset5 + 31] = BASE + w_idx;


	// scR CODE FOR GRID NUMBER 6.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*(2.0*dZu2*dZu2*h2*h2/alpha2) + dzodr*(2.0*h2 + r2*lambda)*(2.0*dRu2*dRu2*h2/alpha2);
	jacobian_submatrix_1[1] = dzodr*(-dRu6 + 4.0*dRu1*(lambda + Q1*h2/r2) - (2.0/ri)*(lambda + Q1*h2/r2) - 4.0*dRu3*h2/r2);
	jacobian_submatrix_1[2] = drodz*dZu6;
	jacobian_submatrix_1[3] = dzodr*2.0*(lambda + Q1*h2/r2);
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = 0;
	jacobian_submatrix_2[1] = dzodr*(-2.0*dRu2*h2/alpha2)*(2.0*h2 + r2*lambda);
	jacobian_submatrix_2[2] = drodz*(-2.0*dZu2*h2*h2/alpha2);
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = drodz*(2.0*h2)*(-2.0*dZu2*dZu2*h2/alpha2 + r2*(dZu6 - 2.0*dZu3*lambda)*(dZu6 - 2.0*dZu3*lambda)/(a2_r*a2_r))+ dzodr*((Q1*(dRRu1 + dRu1*(dRu1 - 1.0/ri)) + Q2*(dRRu3 + dRu3*(2.0*dRu3 - 1.0/ri)))*(4.0*h2/r2) - 8.0*dRu1*dRu3*(h2/r2) + 64.0*M_PI*l*h2*rlm1*rlm1*psi*(dRu5/ri) + 32.0*M_PI*h2*rlm1*rlm1*(dRu5*dRu5) + 8.0*h2*r2*lambda*(dRu6/ri)/(a2_r*a2_r) + 2.0*r2*h2*dRu6*dRu6/(a2_r*a2_r) + (dRu2*dRu2)*(-8.0*h2*h2/alpha2 - 2.0*r2*lambda*h2/alpha2) + dr2*h2*(16.0*M_PI*rlm1*rlm1*r2*lambda*m2*psi*psi + 8.0*(lambda/a2_r)*(lambda/a2_r)) + (dRu3/ri)*(-16.0*h2*r2*lambda*lambda/(a2_r*a2_r) - 8.0*h2*r2*lambda*(ri*dRu6)/(a2_r*a2_r)) + (dRu3*dRu3)*(h2/r2)*(-4.0 + 8.0*(h2/a2_r)*(h2/a2_r) - 16.0*h2/a2_r));
	jacobian_submatrix_3[1] = dzodr*(Q2*(8.0*dRu3 - 2.0/ri)*(lambda + h2/r2)*(h2/a2_r) + dRu6*(-5.0*h2 - r2*lambda)/a2_r - 4.0*dRu1*(h2/a2_r)*(lambda + h2/r2) +dRu3*(-12.0*h2*h2/r2 + 4.0*r2*lambda*lambda)/a2_r + lambda*(-6.0*h2/ri + 2.0*r2*lambda/ri)/a2_r);
	jacobian_submatrix_3[2] = drodz*(8.0*dZu3*lambda*h2 + dZu6*(-3.0*h2 + r2*lambda))/a2_r;
	jacobian_submatrix_3[3] = dzodr*2.0*(lambda + Q2*h2/r2);
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = 0;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*(16.0*a2_r*M_PI*rlm1*rlm1)*(dr2*r2*lambda*m2*psi + 2.0*l*dRu5/ri);
	jacobian_submatrix_5[1] = dzodr*(32.0*a2_r*M_PI*rlm1*rlm1)*(l*psi/ri + dRu5);
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = drodz*(r2*dZu6 + 2.0*h2*dZu3)*(r2*dZu6 + 2.0*h2*dZu3)/(a2_r*a2_r) + dzodr*(2.0*dRRu1 + 2.0*dRRu3 + 2.0*dRu1*(dRu1 - 1.0/ri) - r2*h2*dRu2*dRu2/alpha2 + 16.0*M_PI*rl*rl*dRu5*dRu5 + 32.0*M_PI*l*rl*rl*psi*(dRu5/ri) + (r2*dRu6/a2_r)*(r2*dRu6/a2_r) + (dRu3*dRu3)*(2.0 + 4.0*(h2/a2_r)*(h2/a2_r)) + dr2*(r2*lambda)*(8.0*M_PI*m2*phi2 + 4.0*lambda/(a2_r*a2_r)) + (dRu6/ri)*(4.0*r2)*(-h2/(a2_r*a2_r)) + (dRu3/ri)*(-6.0*(h2/a2_r)*(h2/a2_r) + 4.0*h2*(ri*r2*dRu6)/(a2_r*a2_r) - 2.0*(r2*lambda/a2_r)*(r2*lambda/a2_r) + 4.0*r2*lambda/a2_r) + dr2*(-8.0*lambda/a2_r + 8.0*M_PI*a2_r*m2*phi2));
	jacobian_submatrix_6[1] = dzodr*((3.0*h2 - r2*lambda)/ri - dRu1*(h2 + r2*lambda) - dRu3*(5.0*h2 + r2*lambda) - 2.0*r2*dRu6)/a2_r;
	jacobian_submatrix_6[2] = drodz*(-2.0*r2*dZu6 + dZu1*(h2 + r2*lambda) + dZu3*(-3.0*h2 + r2*lambda))/a2_r;
	jacobian_submatrix_6[3] = dzodr;
	jacobian_submatrix_6[4] = drodz;

	// Omega term.
	jacobian_submatrix_w = 0;

	// This row 5 * dim + IDX(i, j) starts at offset6
	ia[5 * dim + IDX(i, j)] = BASE + offset6;

	// Values.
	aa[offset6 +  0] = +S20*jacobian_submatrix_1[3];
	aa[offset6 +  1] = +S11*jacobian_submatrix_1[1]+S21*jacobian_submatrix_1[3];
	aa[offset6 +  2] = +S12*jacobian_submatrix_1[1]+S22*jacobian_submatrix_1[3];
	aa[offset6 +  3] = +S13*jacobian_submatrix_1[1]+S23*jacobian_submatrix_1[3];
	aa[offset6 +  4] = +D10*jacobian_submatrix_1[2];
	aa[offset6 +  5] = +D11*jacobian_submatrix_1[2];
	aa[offset6 +  6] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[1]+S24*jacobian_submatrix_1[3];
	aa[offset6 +  7] = +D13*jacobian_submatrix_1[2];
	aa[offset6 +  8] = +D14*jacobian_submatrix_1[2];
	aa[offset6 +  9] = +S15*jacobian_submatrix_1[1]+S25*jacobian_submatrix_1[3];
	aa[offset6 + 10] = +S11*jacobian_submatrix_2[1];
	aa[offset6 + 11] = +S12*jacobian_submatrix_2[1];
	aa[offset6 + 12] = +S13*jacobian_submatrix_2[1];
	aa[offset6 + 13] = +D10*jacobian_submatrix_2[2];
	aa[offset6 + 14] = +D11*jacobian_submatrix_2[2];
	aa[offset6 + 15] = +S14*jacobian_submatrix_2[1];
	aa[offset6 + 16] = +D13*jacobian_submatrix_2[2];
	aa[offset6 + 17] = +D14*jacobian_submatrix_2[2];
	aa[offset6 + 18] = +S15*jacobian_submatrix_2[1];
	aa[offset6 + 19] = +S20*jacobian_submatrix_3[3];
	aa[offset6 + 20] = +S11*jacobian_submatrix_3[1]+S21*jacobian_submatrix_3[3];
	aa[offset6 + 21] = +S12*jacobian_submatrix_3[1]+S22*jacobian_submatrix_3[3];
	aa[offset6 + 22] = +S13*jacobian_submatrix_3[1]+S23*jacobian_submatrix_3[3];
	aa[offset6 + 23] = +D10*jacobian_submatrix_3[2];
	aa[offset6 + 24] = +D11*jacobian_submatrix_3[2];
	aa[offset6 + 25] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1]+S24*jacobian_submatrix_3[3];
	aa[offset6 + 26] = +D13*jacobian_submatrix_3[2];
	aa[offset6 + 27] = +D14*jacobian_submatrix_3[2];
	aa[offset6 + 28] = +S15*jacobian_submatrix_3[1]+S25*jacobian_submatrix_3[3];
	aa[offset6 + 29] = +S11*jacobian_submatrix_5[1];
	aa[offset6 + 30] = +S12*jacobian_submatrix_5[1];
	aa[offset6 + 31] = +S13*jacobian_submatrix_5[1];
	aa[offset6 + 32] = +1.0*jacobian_submatrix_5[0]+S14*jacobian_submatrix_5[1];
	aa[offset6 + 33] = +S15*jacobian_submatrix_5[1];
	aa[offset6 + 34] = +S20*jacobian_submatrix_6[3];
	aa[offset6 + 35] = +S11*jacobian_submatrix_6[1]+S21*jacobian_submatrix_6[3];
	aa[offset6 + 36] = +S12*jacobian_submatrix_6[1]+S22*jacobian_submatrix_6[3];
	aa[offset6 + 37] = +S13*jacobian_submatrix_6[1]+S23*jacobian_submatrix_6[3];
	aa[offset6 + 38] = +D10*jacobian_submatrix_6[2]+D20*jacobian_submatrix_6[4];
	aa[offset6 + 39] = +D11*jacobian_submatrix_6[2]+D21*jacobian_submatrix_6[4];
	aa[offset6 + 40] = +1.0*jacobian_submatrix_6[0]+S14*jacobian_submatrix_6[1]+S24*jacobian_submatrix_6[3]+D22*jacobian_submatrix_6[4];
	aa[offset6 + 41] = +D13*jacobian_submatrix_6[2]+D23*jacobian_submatrix_6[4];
	aa[offset6 + 42] = +D14*jacobian_submatrix_6[2]+D24*jacobian_submatrix_6[4];
	aa[offset6 + 43] = +S15*jacobian_submatrix_6[1]+S25*jacobian_submatrix_6[3];

	// Columns.
	ja[offset6 +  0] = BASE + 0 * dim + IDX(i - 4, j    );
	ja[offset6 +  1] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset6 +  2] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset6 +  3] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset6 +  4] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset6 +  5] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset6 +  6] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset6 +  7] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset6 +  8] = BASE + 0 * dim + IDX(i    , j + 2);
	ja[offset6 +  9] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset6 + 10] = BASE + 1 * dim + IDX(i - 3, j    );
	ja[offset6 + 11] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset6 + 12] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset6 + 13] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset6 + 14] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset6 + 15] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset6 + 16] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset6 + 17] = BASE + 1 * dim + IDX(i    , j + 2);
	ja[offset6 + 18] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset6 + 19] = BASE + 2 * dim + IDX(i - 4, j    );
	ja[offset6 + 20] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset6 + 21] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset6 + 22] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset6 + 23] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset6 + 24] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset6 + 25] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset6 + 26] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset6 + 27] = BASE + 2 * dim + IDX(i    , j + 2);
	ja[offset6 + 28] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset6 + 29] = BASE + 4 * dim + IDX(i - 3, j    );
	ja[offset6 + 30] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset6 + 31] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset6 + 32] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset6 + 33] = BASE + 4 * dim + IDX(i + 1, j    );
	ja[offset6 + 34] = BASE + 5 * dim + IDX(i - 4, j    );
	ja[offset6 + 35] = BASE + 5 * dim + IDX(i - 3, j    );
	ja[offset6 + 36] = BASE + 5 * dim + IDX(i - 2, j    );
	ja[offset6 + 37] = BASE + 5 * dim + IDX(i - 1, j    );
	ja[offset6 + 38] = BASE + 5 * dim + IDX(i    , j - 2);
	ja[offset6 + 39] = BASE + 5 * dim + IDX(i    , j - 1);
	ja[offset6 + 40] = BASE + 5 * dim + IDX(i    , j    );
	ja[offset6 + 41] = BASE + 5 * dim + IDX(i    , j + 1);
	ja[offset6 + 42] = BASE + 5 * dim + IDX(i    , j + 2);
	ja[offset6 + 43] = BASE + 5 * dim + IDX(i + 1, j    );


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
	const MKL_INT ghost,	// Number of ghost zones.
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
	const double u604, const double u614, const double u624, const double u634, const double u640, const double u641, const double u642, const double u643, const double u644, const double u645, const double u654,
	const MKL_INT offset1,	// Number of elements filled before filling function 1.
	const MKL_INT offset2, 	// Number of elements filled before filling function 2.
	const MKL_INT offset3, 	// Number of elements filled before filling function 3.
	const MKL_INT offset4, 	// Number of elements filled before filling function 4.
	const MKL_INT offset5, 	// Number of elements filled before filling function 5
	const MKL_INT offset6 	// Number of elements filled before filling function 5
)
{
	// Grid variables.
	double u1 = u144;
	double u2 = u244;
	double u3 = u344;
	double u4 = u444;
	double u5 = u544;
	double u6 = u644;
    
	// Physical names for readability.
	double alpha = exp(u1);
	double Omega = u2;
	double h = exp(u3);
	double a = exp(u4);
	double psi = u5;
	double lambda = u6;
    
	// Coordinates.
	double ri = (double)i + 0.5 - ghost;
	double r = ri * dr;
	double r2 = r * r;
    
	// Step ratios.
	double dzodr = dz / dr;
	double drodz = dr / dz;
	double dr2 = dr * dr;
    
	// Scalar field mass and frequency.
	double w = omega_calc(xi, m);
	double m2 = m * m;
	// Omega variable index position.
	MKL_INT w_idx = GNUM * dim;
    
	// Short hands.
	// Scalar field.
	double rlm1 = (l == 1) ? 1.0 : pow(r, l - 1);
	double rl = rlm1 * r;
	double phior = rlm1 * psi;
	double phi = r * phior;
	double phi2or2 = phior * phior;
	double phi2 = phi * phi;
	// Shift combined with scalar field rotation and frequency.
	double wplOmega = w + l * Omega;
	double wplOmega2 = wplOmega * wplOmega;
	// Squared variables.
	double alpha2 = alpha * alpha;
	double h2 = h * h;
	double a2 = a * a;
	// Regularization a2.
	double a2_r = h2 + r2 * lambda;
    
	// Finite differences.
	// Axial derivatives.
	double dRu1 = S11 * u114 + S12 * u124 + S13 * u134 + S14 * u144 + S15 * u154;
	double dRu2 = S11 * u214 + S12 * u224 + S13 * u234 + S14 * u244 + S15 * u254;
	double dRu3 = S11 * u314 + S12 * u324 + S13 * u334 + S14 * u344 + S15 * u354;
	//double dRu4 = S11 * u414 + S12 * u424 + S13 * u434 + S14 * u444 + S15 * u454;
	double dRu5 = S11 * u514 + S12 * u524 + S13 * u534 + S14 * u544 + S15 * u554;
	double dRu6 = S11 * u614 + S12 * u624 + S13 * u634 + S14 * u644 + S15 * u654;
	// Z derivatives.
	double dZu1 = S11 * u141 + S12 * u142 + S13 * u143 + S14 * u144 + S15 * u145;
	double dZu2 = S11 * u241 + S12 * u242 + S13 * u243 + S14 * u244 + S15 * u245;
	double dZu3 = S11 * u341 + S12 * u342 + S13 * u343 + S14 * u344 + S15 * u345;
	//double dZu4 = S11 * u441 + S12 * u442 + S13 * u443 + S14 * u444 + S15 * u445;
	double dZu5 = S11 * u541 + S12 * u542 + S13 * u543 + S14 * u544 + S15 * u545;
	double dZu6 = S11 * u641 + S12 * u642 + S13 * u643 + S14 * u644 + S15 * u645;
	// Second derivatives.
	double dRRu1 = S20 * u104 + S21 * u114 + S22 * u124 + S23 * u134 + S24 * u144 + S25 * u154;
	double dRRu3 = S20 * u304 + S21 * u314 + S22 * u324 + S23 * u334 + S24 * u344 + S25 * u354;
    
	// Declare Jacobian submatrices.
	double jacobian_submatrix_1[5] = { 0.0 };
	double jacobian_submatrix_2[5] = { 0.0 };
	double jacobian_submatrix_3[5] = { 0.0 };
	double jacobian_submatrix_4[5] = { 0.0 };
	double jacobian_submatrix_5[5] = { 0.0 };
	double jacobian_submatrix_6[5] = { 0.0 };
	double jacobian_submatrix_w = 0.0;

	// ssR CODE FOR GRID NUMBER 1.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*r2*h2*dZu2*dZu2/alpha2 + dzodr*(r2*h2*dRu2*dRu2/alpha2 + 16.0*M_PI*dr2*a2*phi2*wplOmega2/alpha2);
	jacobian_submatrix_1[1] = dzodr*(2.0*dRu1 + dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = drodz*(2.0*dZu1 + dZu3);
	jacobian_submatrix_1[3] = dzodr;
	jacobian_submatrix_1[4] = drodz;

	// Submatrix 2.
	jacobian_submatrix_2[0] = -dzodr*dr2*16.0*M_PI*l*a2*phi2*wplOmega/alpha2;
	jacobian_submatrix_2[1] = -dzodr*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = -drodz*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = -dzodr*r2*h2*(dRu2*dRu2 + drodz*drodz*dZu2*dZu2)/alpha2;
	jacobian_submatrix_3[1] = dzodr*dRu1;
	jacobian_submatrix_3[2] = drodz*dZu1;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(m2 - 2.0*wplOmega2/alpha2)*phi2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*a2*(m2 - 2.0*wplOmega2/alpha2)*phi*rl;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (-dzodr*dr2*16.0*M_PI*a2*phi2*wplOmega/alpha2);

	// This row 0 * dim + IDX(i, j) starts at offset1
	ia[0 * dim + IDX(i, j)] = BASE + offset1;

	// Values.
	aa[offset1 +  0] = +S20*jacobian_submatrix_1[3];
	aa[offset1 +  1] = +S11*jacobian_submatrix_1[1]+S21*jacobian_submatrix_1[3];
	aa[offset1 +  2] = +S12*jacobian_submatrix_1[1]+S22*jacobian_submatrix_1[3];
	aa[offset1 +  3] = +S13*jacobian_submatrix_1[1]+S23*jacobian_submatrix_1[3];
	aa[offset1 +  4] = +S20*jacobian_submatrix_1[4];
	aa[offset1 +  5] = +S11*jacobian_submatrix_1[2]+S21*jacobian_submatrix_1[4];
	aa[offset1 +  6] = +S12*jacobian_submatrix_1[2]+S22*jacobian_submatrix_1[4];
	aa[offset1 +  7] = +S13*jacobian_submatrix_1[2]+S23*jacobian_submatrix_1[4];
	aa[offset1 +  8] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[1]+S14*jacobian_submatrix_1[2]+S24*jacobian_submatrix_1[3]+S24*jacobian_submatrix_1[4];
	aa[offset1 +  9] = +S15*jacobian_submatrix_1[2]+S25*jacobian_submatrix_1[4];
	aa[offset1 + 10] = +S15*jacobian_submatrix_1[1]+S25*jacobian_submatrix_1[3];
	aa[offset1 + 11] = +S11*jacobian_submatrix_2[1];
	aa[offset1 + 12] = +S12*jacobian_submatrix_2[1];
	aa[offset1 + 13] = +S13*jacobian_submatrix_2[1];
	aa[offset1 + 14] = +S11*jacobian_submatrix_2[2];
	aa[offset1 + 15] = +S12*jacobian_submatrix_2[2];
	aa[offset1 + 16] = +S13*jacobian_submatrix_2[2];
	aa[offset1 + 17] = +1.0*jacobian_submatrix_2[0]+S14*jacobian_submatrix_2[1]+S14*jacobian_submatrix_2[2];
	aa[offset1 + 18] = +S15*jacobian_submatrix_2[2];
	aa[offset1 + 19] = +S15*jacobian_submatrix_2[1];
	aa[offset1 + 20] = +S11*jacobian_submatrix_3[1];
	aa[offset1 + 21] = +S12*jacobian_submatrix_3[1];
	aa[offset1 + 22] = +S13*jacobian_submatrix_3[1];
	aa[offset1 + 23] = +S11*jacobian_submatrix_3[2];
	aa[offset1 + 24] = +S12*jacobian_submatrix_3[2];
	aa[offset1 + 25] = +S13*jacobian_submatrix_3[2];
	aa[offset1 + 26] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1]+S14*jacobian_submatrix_3[2];
	aa[offset1 + 27] = +S15*jacobian_submatrix_3[2];
	aa[offset1 + 28] = +S15*jacobian_submatrix_3[1];
	aa[offset1 + 29] = +1.0*jacobian_submatrix_4[0];
	aa[offset1 + 30] = +1.0*jacobian_submatrix_5[0];
	aa[offset1 + 31] = jacobian_submatrix_w;

	// Columns.
	ja[offset1 +  0] = BASE + 0 * dim + IDX(i - 4, j    );
	ja[offset1 +  1] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset1 +  2] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset1 +  3] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset1 +  4] = BASE + 0 * dim + IDX(i    , j - 4);
	ja[offset1 +  5] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset1 +  6] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset1 +  7] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset1 +  8] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset1 +  9] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset1 + 10] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset1 + 11] = BASE + 1 * dim + IDX(i - 3, j    );
	ja[offset1 + 12] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset1 + 13] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset1 + 14] = BASE + 1 * dim + IDX(i    , j - 3);
	ja[offset1 + 15] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset1 + 16] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset1 + 17] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset1 + 18] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset1 + 19] = BASE + 1 * dim + IDX(i + 1, j    );
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
	ja[offset1 + 31] = BASE + w_idx;


	// ssR CODE FOR GRID NUMBER 2.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = 0;
	jacobian_submatrix_1[1] = -dzodr*dRu2;
	jacobian_submatrix_1[2] = -drodz*dZu2;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = -dzodr*dr2*16.0*M_PI*l*l*a2*phi2or2/h2;
	jacobian_submatrix_2[1] = dzodr*(-dRu1 + 3.0*dRu3 + 3.0/ri);
	jacobian_submatrix_2[2] = drodz*(-dZu1 + 3.0*dZu3);
	jacobian_submatrix_2[3] = dzodr;
	jacobian_submatrix_2[4] = drodz;

	// Submatrix 3.
	jacobian_submatrix_3[0] = dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phi2or2/h2;
	jacobian_submatrix_3[1] = dzodr*3.0*dRu2;
	jacobian_submatrix_3[2] = drodz*3.0*dZu2;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = -dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phi2or2/h2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = -dzodr*dr2*32.0*M_PI*l*a2*wplOmega*phior*rlm1/h2;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (-dzodr*dr2*16.0*M_PI*l*a2*phi2or2/h2);

	// This row 1 * dim + IDX(i, j) starts at offset2
	ia[1 * dim + IDX(i, j)] = BASE + offset2;

	// Values.
	aa[offset2 +  0] = +S11*jacobian_submatrix_1[1];
	aa[offset2 +  1] = +S12*jacobian_submatrix_1[1];
	aa[offset2 +  2] = +S13*jacobian_submatrix_1[1];
	aa[offset2 +  3] = +S11*jacobian_submatrix_1[2];
	aa[offset2 +  4] = +S12*jacobian_submatrix_1[2];
	aa[offset2 +  5] = +S13*jacobian_submatrix_1[2];
	aa[offset2 +  6] = +S14*jacobian_submatrix_1[1]+S14*jacobian_submatrix_1[2];
	aa[offset2 +  7] = +S15*jacobian_submatrix_1[2];
	aa[offset2 +  8] = +S15*jacobian_submatrix_1[1];
	aa[offset2 +  9] = +S20*jacobian_submatrix_2[3];
	aa[offset2 + 10] = +S11*jacobian_submatrix_2[1]+S21*jacobian_submatrix_2[3];
	aa[offset2 + 11] = +S12*jacobian_submatrix_2[1]+S22*jacobian_submatrix_2[3];
	aa[offset2 + 12] = +S13*jacobian_submatrix_2[1]+S23*jacobian_submatrix_2[3];
	aa[offset2 + 13] = +S20*jacobian_submatrix_2[4];
	aa[offset2 + 14] = +S11*jacobian_submatrix_2[2]+S21*jacobian_submatrix_2[4];
	aa[offset2 + 15] = +S12*jacobian_submatrix_2[2]+S22*jacobian_submatrix_2[4];
	aa[offset2 + 16] = +S13*jacobian_submatrix_2[2]+S23*jacobian_submatrix_2[4];
	aa[offset2 + 17] = +1.0*jacobian_submatrix_2[0]+S14*jacobian_submatrix_2[1]+S14*jacobian_submatrix_2[2]+S24*jacobian_submatrix_2[3]+S24*jacobian_submatrix_2[4];
	aa[offset2 + 18] = +S15*jacobian_submatrix_2[2]+S25*jacobian_submatrix_2[4];
	aa[offset2 + 19] = +S15*jacobian_submatrix_2[1]+S25*jacobian_submatrix_2[3];
	aa[offset2 + 20] = +S11*jacobian_submatrix_3[1];
	aa[offset2 + 21] = +S12*jacobian_submatrix_3[1];
	aa[offset2 + 22] = +S13*jacobian_submatrix_3[1];
	aa[offset2 + 23] = +S11*jacobian_submatrix_3[2];
	aa[offset2 + 24] = +S12*jacobian_submatrix_3[2];
	aa[offset2 + 25] = +S13*jacobian_submatrix_3[2];
	aa[offset2 + 26] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1]+S14*jacobian_submatrix_3[2];
	aa[offset2 + 27] = +S15*jacobian_submatrix_3[2];
	aa[offset2 + 28] = +S15*jacobian_submatrix_3[1];
	aa[offset2 + 29] = +1.0*jacobian_submatrix_4[0];
	aa[offset2 + 30] = +1.0*jacobian_submatrix_5[0];
	aa[offset2 + 31] = jacobian_submatrix_w;

	// Columns.
	ja[offset2 +  0] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset2 +  1] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset2 +  2] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset2 +  3] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset2 +  4] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset2 +  5] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset2 +  6] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset2 +  7] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset2 +  8] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset2 +  9] = BASE + 1 * dim + IDX(i - 4, j    );
	ja[offset2 + 10] = BASE + 1 * dim + IDX(i - 3, j    );
	ja[offset2 + 11] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset2 + 12] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset2 + 13] = BASE + 1 * dim + IDX(i    , j - 4);
	ja[offset2 + 14] = BASE + 1 * dim + IDX(i    , j - 3);
	ja[offset2 + 15] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset2 + 16] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset2 + 17] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset2 + 18] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset2 + 19] = BASE + 1 * dim + IDX(i + 1, j    );
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
	ja[offset2 + 31] = BASE + w_idx;


	// ssR CODE FOR GRID NUMBER 3.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = -dzodr*r2*h2*dRu2*dRu2/alpha2 - drodz*r2*h2*dZu2*dZu2/alpha2;
	jacobian_submatrix_1[1] = dzodr*(dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = drodz*dZu3;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = 0;
	jacobian_submatrix_2[1] = dzodr*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = drodz*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = drodz*r2*h2*dZu2*dZu2/alpha2 + dzodr*(r2*h2*dRu2*dRu2/alpha2 - dr2*16.0*M_PI*l*l*a2*phi2or2/h2);
	jacobian_submatrix_3[1] = dzodr*(dRu1 + 2.0*dRu3 + 2.0/ri);
	jacobian_submatrix_3[2] = drodz*(dZu1 + 2.0*dZu3);
	jacobian_submatrix_3[3] = dzodr;
	jacobian_submatrix_3[4] = drodz;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(r2*m2 + 2.0*l*l/h2)*phi2or2;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*a2*(r2*m2 + 2.0*l*l/h2)*phior*rlm1;
	jacobian_submatrix_5[1] = 0;
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = 0;

	// This row 2 * dim + IDX(i, j) starts at offset3
	ia[2 * dim + IDX(i, j)] = BASE + offset3;

	// Values.
	aa[offset3 +  0] = +S11*jacobian_submatrix_1[1];
	aa[offset3 +  1] = +S12*jacobian_submatrix_1[1];
	aa[offset3 +  2] = +S13*jacobian_submatrix_1[1];
	aa[offset3 +  3] = +S11*jacobian_submatrix_1[2];
	aa[offset3 +  4] = +S12*jacobian_submatrix_1[2];
	aa[offset3 +  5] = +S13*jacobian_submatrix_1[2];
	aa[offset3 +  6] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[1]+S14*jacobian_submatrix_1[2];
	aa[offset3 +  7] = +S15*jacobian_submatrix_1[2];
	aa[offset3 +  8] = +S15*jacobian_submatrix_1[1];
	aa[offset3 +  9] = +S11*jacobian_submatrix_2[1];
	aa[offset3 + 10] = +S12*jacobian_submatrix_2[1];
	aa[offset3 + 11] = +S13*jacobian_submatrix_2[1];
	aa[offset3 + 12] = +S11*jacobian_submatrix_2[2];
	aa[offset3 + 13] = +S12*jacobian_submatrix_2[2];
	aa[offset3 + 14] = +S13*jacobian_submatrix_2[2];
	aa[offset3 + 15] = +S14*jacobian_submatrix_2[1]+S14*jacobian_submatrix_2[2];
	aa[offset3 + 16] = +S15*jacobian_submatrix_2[2];
	aa[offset3 + 17] = +S15*jacobian_submatrix_2[1];
	aa[offset3 + 18] = +S20*jacobian_submatrix_3[3];
	aa[offset3 + 19] = +S11*jacobian_submatrix_3[1]+S21*jacobian_submatrix_3[3];
	aa[offset3 + 20] = +S12*jacobian_submatrix_3[1]+S22*jacobian_submatrix_3[3];
	aa[offset3 + 21] = +S13*jacobian_submatrix_3[1]+S23*jacobian_submatrix_3[3];
	aa[offset3 + 22] = +S20*jacobian_submatrix_3[4];
	aa[offset3 + 23] = +S11*jacobian_submatrix_3[2]+S21*jacobian_submatrix_3[4];
	aa[offset3 + 24] = +S12*jacobian_submatrix_3[2]+S22*jacobian_submatrix_3[4];
	aa[offset3 + 25] = +S13*jacobian_submatrix_3[2]+S23*jacobian_submatrix_3[4];
	aa[offset3 + 26] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1]+S14*jacobian_submatrix_3[2]+S24*jacobian_submatrix_3[3]+S24*jacobian_submatrix_3[4];
	aa[offset3 + 27] = +S15*jacobian_submatrix_3[2]+S25*jacobian_submatrix_3[4];
	aa[offset3 + 28] = +S15*jacobian_submatrix_3[1]+S25*jacobian_submatrix_3[3];
	aa[offset3 + 29] = +1.0*jacobian_submatrix_4[0];
	aa[offset3 + 30] = +1.0*jacobian_submatrix_5[0];

	// Columns.
	ja[offset3 +  0] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset3 +  1] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset3 +  2] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset3 +  3] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset3 +  4] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset3 +  5] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset3 +  6] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset3 +  7] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset3 +  8] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset3 +  9] = BASE + 1 * dim + IDX(i - 3, j    );
	ja[offset3 + 10] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset3 + 11] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset3 + 12] = BASE + 1 * dim + IDX(i    , j - 3);
	ja[offset3 + 13] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset3 + 14] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset3 + 15] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset3 + 16] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset3 + 17] = BASE + 1 * dim + IDX(i + 1, j    );
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


	// ssR CODE FOR GRID NUMBER 4.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*0.5*r2*h2*dZu2*dZu2/alpha2 + dzodr*(0.5*r2*h2*dRu2*dRu2/alpha2 - dr2*8.0*M_PI*a2*wplOmega2*phi2/alpha2);
	jacobian_submatrix_1[1] = -dzodr*(dRu3 + 1.0/ri);
	jacobian_submatrix_1[2] = -drodz*dZu3;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = dzodr*dr2*8.0*M_PI*l*a2*wplOmega*phi2/alpha2;
	jacobian_submatrix_2[1] = -dzodr*0.5*r2*h2*dRu2/alpha2;
	jacobian_submatrix_2[2] = -drodz*0.5*r2*h2*dZu2/alpha2;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = -drodz*0.5*r2*h2*dZu2*dZu2/alpha2 + dzodr*(-0.5*r2*h2*dRu2*dRu2/alpha2 + dr2*8.0*l*l*M_PI*phi2or2*a2/h2);
	jacobian_submatrix_3[1] = -dzodr*dRu1;
	jacobian_submatrix_3[2] = -drodz*dZu1;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = dzodr*dr2*8.0*M_PI*a2*(-l*l*phi2or2/h2 + wplOmega2*phi2/alpha2);
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = dzodr;
	jacobian_submatrix_4[4] = drodz;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*dr2*8.0*M_PI*(l*ri*dRu5 + l*l*psi - l*l*a2*psi/h2 + r2*a2*wplOmega2*psi/alpha2)*rlm1*rlm1;
	jacobian_submatrix_5[1] = dzodr*8.0*M_PI*rl*rl*(dRu5 + l*psi/ri);
	jacobian_submatrix_5[2] = drodz*8.0*M_PI*rl*rl*dZu5;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = 0;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (dzodr*dr2*8.0*M_PI*phi2*a2*wplOmega/alpha2);

	// This row 3 * dim + IDX(i, j) starts at offset4
	ia[3 * dim + IDX(i, j)] = BASE + offset4;

	// Values.
	aa[offset4 +  0] = +S11*jacobian_submatrix_1[1];
	aa[offset4 +  1] = +S12*jacobian_submatrix_1[1];
	aa[offset4 +  2] = +S13*jacobian_submatrix_1[1];
	aa[offset4 +  3] = +S11*jacobian_submatrix_1[2];
	aa[offset4 +  4] = +S12*jacobian_submatrix_1[2];
	aa[offset4 +  5] = +S13*jacobian_submatrix_1[2];
	aa[offset4 +  6] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[1]+S14*jacobian_submatrix_1[2];
	aa[offset4 +  7] = +S15*jacobian_submatrix_1[2];
	aa[offset4 +  8] = +S15*jacobian_submatrix_1[1];
	aa[offset4 +  9] = +S11*jacobian_submatrix_2[1];
	aa[offset4 + 10] = +S12*jacobian_submatrix_2[1];
	aa[offset4 + 11] = +S13*jacobian_submatrix_2[1];
	aa[offset4 + 12] = +S11*jacobian_submatrix_2[2];
	aa[offset4 + 13] = +S12*jacobian_submatrix_2[2];
	aa[offset4 + 14] = +S13*jacobian_submatrix_2[2];
	aa[offset4 + 15] = +1.0*jacobian_submatrix_2[0]+S14*jacobian_submatrix_2[1]+S14*jacobian_submatrix_2[2];
	aa[offset4 + 16] = +S15*jacobian_submatrix_2[2];
	aa[offset4 + 17] = +S15*jacobian_submatrix_2[1];
	aa[offset4 + 18] = +S11*jacobian_submatrix_3[1];
	aa[offset4 + 19] = +S12*jacobian_submatrix_3[1];
	aa[offset4 + 20] = +S13*jacobian_submatrix_3[1];
	aa[offset4 + 21] = +S11*jacobian_submatrix_3[2];
	aa[offset4 + 22] = +S12*jacobian_submatrix_3[2];
	aa[offset4 + 23] = +S13*jacobian_submatrix_3[2];
	aa[offset4 + 24] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1]+S14*jacobian_submatrix_3[2];
	aa[offset4 + 25] = +S15*jacobian_submatrix_3[2];
	aa[offset4 + 26] = +S15*jacobian_submatrix_3[1];
	aa[offset4 + 27] = +S20*jacobian_submatrix_4[3];
	aa[offset4 + 28] = +S21*jacobian_submatrix_4[3];
	aa[offset4 + 29] = +S22*jacobian_submatrix_4[3];
	aa[offset4 + 30] = +S23*jacobian_submatrix_4[3];
	aa[offset4 + 31] = +S20*jacobian_submatrix_4[4];
	aa[offset4 + 32] = +S21*jacobian_submatrix_4[4];
	aa[offset4 + 33] = +S22*jacobian_submatrix_4[4];
	aa[offset4 + 34] = +S23*jacobian_submatrix_4[4];
	aa[offset4 + 35] = +1.0*jacobian_submatrix_4[0]+S24*jacobian_submatrix_4[3]+S24*jacobian_submatrix_4[4];
	aa[offset4 + 36] = +S25*jacobian_submatrix_4[4];
	aa[offset4 + 37] = +S25*jacobian_submatrix_4[3];
	aa[offset4 + 38] = +S11*jacobian_submatrix_5[1];
	aa[offset4 + 39] = +S12*jacobian_submatrix_5[1];
	aa[offset4 + 40] = +S13*jacobian_submatrix_5[1];
	aa[offset4 + 41] = +S11*jacobian_submatrix_5[2];
	aa[offset4 + 42] = +S12*jacobian_submatrix_5[2];
	aa[offset4 + 43] = +S13*jacobian_submatrix_5[2];
	aa[offset4 + 44] = +1.0*jacobian_submatrix_5[0]+S14*jacobian_submatrix_5[1]+S14*jacobian_submatrix_5[2];
	aa[offset4 + 45] = +S15*jacobian_submatrix_5[2];
	aa[offset4 + 46] = +S15*jacobian_submatrix_5[1];
	aa[offset4 + 47] = jacobian_submatrix_w;

	// Columns.
	ja[offset4 +  0] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset4 +  1] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset4 +  2] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset4 +  3] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset4 +  4] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset4 +  5] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset4 +  6] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset4 +  7] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset4 +  8] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset4 +  9] = BASE + 1 * dim + IDX(i - 3, j    );
	ja[offset4 + 10] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset4 + 11] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset4 + 12] = BASE + 1 * dim + IDX(i    , j - 3);
	ja[offset4 + 13] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset4 + 14] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset4 + 15] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset4 + 16] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset4 + 17] = BASE + 1 * dim + IDX(i + 1, j    );
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
	ja[offset4 + 47] = BASE + w_idx;


	// ssR CODE FOR GRID NUMBER 5.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = -dzodr*dr2*2.0*a2*wplOmega2*psi/alpha2;
	jacobian_submatrix_1[1] = dzodr*(dRu5 + l*psi/ri);
	jacobian_submatrix_1[2] = drodz*dZu5;
	jacobian_submatrix_1[3] = 0;
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = dzodr*dr2*2.0*l*a2*wplOmega*psi/alpha2;
	jacobian_submatrix_2[1] = 0;
	jacobian_submatrix_2[2] = 0;
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = dzodr*dr2*2.0*l*l*psi*lambda/h2;
	jacobian_submatrix_3[1] = dzodr*(dRu5 + l*psi/ri);
	jacobian_submatrix_3[2] = drodz*dZu5;
	jacobian_submatrix_3[3] = 0;
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = -dzodr*dr2*2.0*a2*(m2 - wplOmega2/alpha2)*psi;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*(l*(dRu1/ri + dRu3/ri) - dr2*(a2*(m2 - wplOmega2/alpha2) + l*l*lambda/h2));
	jacobian_submatrix_5[1] = dzodr*(dRu1 + dRu3 + (2.0*l + 1.0)/ri);
	jacobian_submatrix_5[2] = drodz*(dZu1 + dZu3);
	jacobian_submatrix_5[3] = dzodr;
	jacobian_submatrix_5[4] = drodz;

	// Submatrix 6.
	jacobian_submatrix_6[0] = -dzodr*dr2*l*l*psi/h2;
	jacobian_submatrix_6[1] = 0;
	jacobian_submatrix_6[2] = 0;
	jacobian_submatrix_6[3] = 0;
	jacobian_submatrix_6[4] = 0;

	// Omega term.
	jacobian_submatrix_w = dw_du(xi, m) * (dzodr*dr2*2.0*a2*wplOmega*psi/alpha2);

	// This row 4 * dim + IDX(i, j) starts at offset5
	ia[4 * dim + IDX(i, j)] = BASE + offset5;

	// Values.
	aa[offset5 +  0] = +S11*jacobian_submatrix_1[1];
	aa[offset5 +  1] = +S12*jacobian_submatrix_1[1];
	aa[offset5 +  2] = +S13*jacobian_submatrix_1[1];
	aa[offset5 +  3] = +S11*jacobian_submatrix_1[2];
	aa[offset5 +  4] = +S12*jacobian_submatrix_1[2];
	aa[offset5 +  5] = +S13*jacobian_submatrix_1[2];
	aa[offset5 +  6] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[1]+S14*jacobian_submatrix_1[2];
	aa[offset5 +  7] = +S15*jacobian_submatrix_1[2];
	aa[offset5 +  8] = +S15*jacobian_submatrix_1[1];
	aa[offset5 +  9] = +1.0*jacobian_submatrix_2[0];
	aa[offset5 + 10] = +S11*jacobian_submatrix_3[1];
	aa[offset5 + 11] = +S12*jacobian_submatrix_3[1];
	aa[offset5 + 12] = +S13*jacobian_submatrix_3[1];
	aa[offset5 + 13] = +S11*jacobian_submatrix_3[2];
	aa[offset5 + 14] = +S12*jacobian_submatrix_3[2];
	aa[offset5 + 15] = +S13*jacobian_submatrix_3[2];
	aa[offset5 + 16] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1]+S14*jacobian_submatrix_3[2];
	aa[offset5 + 17] = +S15*jacobian_submatrix_3[2];
	aa[offset5 + 18] = +S15*jacobian_submatrix_3[1];
	aa[offset5 + 19] = +1.0*jacobian_submatrix_4[0];
	aa[offset5 + 20] = +S20*jacobian_submatrix_5[3];
	aa[offset5 + 21] = +S11*jacobian_submatrix_5[1]+S21*jacobian_submatrix_5[3];
	aa[offset5 + 22] = +S12*jacobian_submatrix_5[1]+S22*jacobian_submatrix_5[3];
	aa[offset5 + 23] = +S13*jacobian_submatrix_5[1]+S23*jacobian_submatrix_5[3];
	aa[offset5 + 24] = +S20*jacobian_submatrix_5[4];
	aa[offset5 + 25] = +S11*jacobian_submatrix_5[2]+S21*jacobian_submatrix_5[4];
	aa[offset5 + 26] = +S12*jacobian_submatrix_5[2]+S22*jacobian_submatrix_5[4];
	aa[offset5 + 27] = +S13*jacobian_submatrix_5[2]+S23*jacobian_submatrix_5[4];
	aa[offset5 + 28] = +1.0*jacobian_submatrix_5[0]+S14*jacobian_submatrix_5[1]+S14*jacobian_submatrix_5[2]+S24*jacobian_submatrix_5[3]+S24*jacobian_submatrix_5[4];
	aa[offset5 + 29] = +S15*jacobian_submatrix_5[2]+S25*jacobian_submatrix_5[4];
	aa[offset5 + 30] = +S15*jacobian_submatrix_5[1]+S25*jacobian_submatrix_5[3];
	aa[offset5 + 31] = +1.0*jacobian_submatrix_6[0];
	aa[offset5 + 32] = jacobian_submatrix_w;

	// Columns.
	ja[offset5 +  0] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset5 +  1] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset5 +  2] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset5 +  3] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset5 +  4] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset5 +  5] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset5 +  6] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset5 +  7] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset5 +  8] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset5 +  9] = BASE + 1 * dim + IDX(i    , j    );
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
	ja[offset5 + 31] = BASE + 5 * dim + IDX(i    , j    );
	ja[offset5 + 32] = BASE + w_idx;


	// ssR CODE FOR GRID NUMBER 6.

	// First write down Jacobian submatrices.
	// Submatrix 1.
	jacobian_submatrix_1[0] = drodz*(2.0*dZu2*dZu2*h2*h2/alpha2) + dzodr*(2.0*h2 + r2*lambda)*(2.0*dRu2*dRu2*h2/alpha2);
	jacobian_submatrix_1[1] = dzodr*(-dRu6 + 4.0*dRu1*(lambda + Q1*h2/r2) - (2.0/ri)*(lambda + Q1*h2/r2) - 4.0*dRu3*h2/r2);
	jacobian_submatrix_1[2] = drodz*dZu6;
	jacobian_submatrix_1[3] = dzodr*2.0*(lambda + Q1*h2/r2);
	jacobian_submatrix_1[4] = 0;

	// Submatrix 2.
	jacobian_submatrix_2[0] = 0;
	jacobian_submatrix_2[1] = dzodr*(-2.0*dRu2*h2/alpha2)*(2.0*h2 + r2*lambda);
	jacobian_submatrix_2[2] = drodz*(-2.0*dZu2*h2*h2/alpha2);
	jacobian_submatrix_2[3] = 0;
	jacobian_submatrix_2[4] = 0;

	// Submatrix 3.
	jacobian_submatrix_3[0] = drodz*(2.0*h2)*(-2.0*dZu2*dZu2*h2/alpha2 + r2*(dZu6 - 2.0*dZu3*lambda)*(dZu6 - 2.0*dZu3*lambda)/(a2_r*a2_r))+ dzodr*((Q1*(dRRu1 + dRu1*(dRu1 - 1.0/ri)) + Q2*(dRRu3 + dRu3*(2.0*dRu3 - 1.0/ri)))*(4.0*h2/r2) - 8.0*dRu1*dRu3*(h2/r2) + 64.0*M_PI*l*h2*rlm1*rlm1*psi*(dRu5/ri) + 32.0*M_PI*h2*rlm1*rlm1*(dRu5*dRu5) + 8.0*h2*r2*lambda*(dRu6/ri)/(a2_r*a2_r) + 2.0*r2*h2*dRu6*dRu6/(a2_r*a2_r) + (dRu2*dRu2)*(-8.0*h2*h2/alpha2 - 2.0*r2*lambda*h2/alpha2) + dr2*h2*(16.0*M_PI*rlm1*rlm1*r2*lambda*m2*psi*psi + 8.0*(lambda/a2_r)*(lambda/a2_r)) + (dRu3/ri)*(-16.0*h2*r2*lambda*lambda/(a2_r*a2_r) - 8.0*h2*r2*lambda*(ri*dRu6)/(a2_r*a2_r)) + (dRu3*dRu3)*(h2/r2)*(-4.0 + 8.0*(h2/a2_r)*(h2/a2_r) - 16.0*h2/a2_r));
	jacobian_submatrix_3[1] = dzodr*(Q2*(8.0*dRu3 - 2.0/ri)*(lambda + h2/r2)*(h2/a2_r) + dRu6*(-5.0*h2 - r2*lambda)/a2_r - 4.0*dRu1*(h2/a2_r)*(lambda + h2/r2) +dRu3*(-12.0*h2*h2/r2 + 4.0*r2*lambda*lambda)/a2_r + lambda*(-6.0*h2/ri + 2.0*r2*lambda/ri)/a2_r);
	jacobian_submatrix_3[2] = drodz*(8.0*dZu3*lambda*h2 + dZu6*(-3.0*h2 + r2*lambda))/a2_r;
	jacobian_submatrix_3[3] = dzodr*2.0*(lambda + Q2*h2/r2);
	jacobian_submatrix_3[4] = 0;

	// Submatrix 4.
	jacobian_submatrix_4[0] = 0;
	jacobian_submatrix_4[1] = 0;
	jacobian_submatrix_4[2] = 0;
	jacobian_submatrix_4[3] = 0;
	jacobian_submatrix_4[4] = 0;

	// Submatrix 5.
	jacobian_submatrix_5[0] = dzodr*(16.0*a2_r*M_PI*rlm1*rlm1)*(dr2*r2*lambda*m2*psi + 2.0*l*dRu5/ri);
	jacobian_submatrix_5[1] = dzodr*(32.0*a2_r*M_PI*rlm1*rlm1)*(l*psi/ri + dRu5);
	jacobian_submatrix_5[2] = 0;
	jacobian_submatrix_5[3] = 0;
	jacobian_submatrix_5[4] = 0;

	// Submatrix 6.
	jacobian_submatrix_6[0] = drodz*(r2*dZu6 + 2.0*h2*dZu3)*(r2*dZu6 + 2.0*h2*dZu3)/(a2_r*a2_r) + dzodr*(2.0*dRRu1 + 2.0*dRRu3 + 2.0*dRu1*(dRu1 - 1.0/ri) - r2*h2*dRu2*dRu2/alpha2 + 16.0*M_PI*rl*rl*dRu5*dRu5 + 32.0*M_PI*l*rl*rl*psi*(dRu5/ri) + (r2*dRu6/a2_r)*(r2*dRu6/a2_r) + (dRu3*dRu3)*(2.0 + 4.0*(h2/a2_r)*(h2/a2_r)) + dr2*(r2*lambda)*(8.0*M_PI*m2*phi2 + 4.0*lambda/(a2_r*a2_r)) + (dRu6/ri)*(4.0*r2)*(-h2/(a2_r*a2_r)) + (dRu3/ri)*(-6.0*(h2/a2_r)*(h2/a2_r) + 4.0*h2*(ri*r2*dRu6)/(a2_r*a2_r) - 2.0*(r2*lambda/a2_r)*(r2*lambda/a2_r) + 4.0*r2*lambda/a2_r) + dr2*(-8.0*lambda/a2_r + 8.0*M_PI*a2_r*m2*phi2));
	jacobian_submatrix_6[1] = dzodr*((3.0*h2 - r2*lambda)/ri - dRu1*(h2 + r2*lambda) - dRu3*(5.0*h2 + r2*lambda) - 2.0*r2*dRu6)/a2_r;
	jacobian_submatrix_6[2] = drodz*(-2.0*r2*dZu6 + dZu1*(h2 + r2*lambda) + dZu3*(-3.0*h2 + r2*lambda))/a2_r;
	jacobian_submatrix_6[3] = dzodr;
	jacobian_submatrix_6[4] = drodz;

	// Omega term.
	jacobian_submatrix_w = 0;

	// This row 5 * dim + IDX(i, j) starts at offset6
	ia[5 * dim + IDX(i, j)] = BASE + offset6;

	// Values.
	aa[offset6 +  0] = +S20*jacobian_submatrix_1[3];
	aa[offset6 +  1] = +S11*jacobian_submatrix_1[1]+S21*jacobian_submatrix_1[3];
	aa[offset6 +  2] = +S12*jacobian_submatrix_1[1]+S22*jacobian_submatrix_1[3];
	aa[offset6 +  3] = +S13*jacobian_submatrix_1[1]+S23*jacobian_submatrix_1[3];
	aa[offset6 +  4] = +S11*jacobian_submatrix_1[2];
	aa[offset6 +  5] = +S12*jacobian_submatrix_1[2];
	aa[offset6 +  6] = +S13*jacobian_submatrix_1[2];
	aa[offset6 +  7] = +1.0*jacobian_submatrix_1[0]+S14*jacobian_submatrix_1[1]+S14*jacobian_submatrix_1[2]+S24*jacobian_submatrix_1[3];
	aa[offset6 +  8] = +S15*jacobian_submatrix_1[2];
	aa[offset6 +  9] = +S15*jacobian_submatrix_1[1]+S25*jacobian_submatrix_1[3];
	aa[offset6 + 10] = +S11*jacobian_submatrix_2[1];
	aa[offset6 + 11] = +S12*jacobian_submatrix_2[1];
	aa[offset6 + 12] = +S13*jacobian_submatrix_2[1];
	aa[offset6 + 13] = +S11*jacobian_submatrix_2[2];
	aa[offset6 + 14] = +S12*jacobian_submatrix_2[2];
	aa[offset6 + 15] = +S13*jacobian_submatrix_2[2];
	aa[offset6 + 16] = +S14*jacobian_submatrix_2[1]+S14*jacobian_submatrix_2[2];
	aa[offset6 + 17] = +S15*jacobian_submatrix_2[2];
	aa[offset6 + 18] = +S15*jacobian_submatrix_2[1];
	aa[offset6 + 19] = +S20*jacobian_submatrix_3[3];
	aa[offset6 + 20] = +S11*jacobian_submatrix_3[1]+S21*jacobian_submatrix_3[3];
	aa[offset6 + 21] = +S12*jacobian_submatrix_3[1]+S22*jacobian_submatrix_3[3];
	aa[offset6 + 22] = +S13*jacobian_submatrix_3[1]+S23*jacobian_submatrix_3[3];
	aa[offset6 + 23] = +S11*jacobian_submatrix_3[2];
	aa[offset6 + 24] = +S12*jacobian_submatrix_3[2];
	aa[offset6 + 25] = +S13*jacobian_submatrix_3[2];
	aa[offset6 + 26] = +1.0*jacobian_submatrix_3[0]+S14*jacobian_submatrix_3[1]+S14*jacobian_submatrix_3[2]+S24*jacobian_submatrix_3[3];
	aa[offset6 + 27] = +S15*jacobian_submatrix_3[2];
	aa[offset6 + 28] = +S15*jacobian_submatrix_3[1]+S25*jacobian_submatrix_3[3];
	aa[offset6 + 29] = +S11*jacobian_submatrix_5[1];
	aa[offset6 + 30] = +S12*jacobian_submatrix_5[1];
	aa[offset6 + 31] = +S13*jacobian_submatrix_5[1];
	aa[offset6 + 32] = +1.0*jacobian_submatrix_5[0]+S14*jacobian_submatrix_5[1];
	aa[offset6 + 33] = +S15*jacobian_submatrix_5[1];
	aa[offset6 + 34] = +S20*jacobian_submatrix_6[3];
	aa[offset6 + 35] = +S11*jacobian_submatrix_6[1]+S21*jacobian_submatrix_6[3];
	aa[offset6 + 36] = +S12*jacobian_submatrix_6[1]+S22*jacobian_submatrix_6[3];
	aa[offset6 + 37] = +S13*jacobian_submatrix_6[1]+S23*jacobian_submatrix_6[3];
	aa[offset6 + 38] = +S20*jacobian_submatrix_6[4];
	aa[offset6 + 39] = +S11*jacobian_submatrix_6[2]+S21*jacobian_submatrix_6[4];
	aa[offset6 + 40] = +S12*jacobian_submatrix_6[2]+S22*jacobian_submatrix_6[4];
	aa[offset6 + 41] = +S13*jacobian_submatrix_6[2]+S23*jacobian_submatrix_6[4];
	aa[offset6 + 42] = +1.0*jacobian_submatrix_6[0]+S14*jacobian_submatrix_6[1]+S14*jacobian_submatrix_6[2]+S24*jacobian_submatrix_6[3]+S24*jacobian_submatrix_6[4];
	aa[offset6 + 43] = +S15*jacobian_submatrix_6[2]+S25*jacobian_submatrix_6[4];
	aa[offset6 + 44] = +S15*jacobian_submatrix_6[1]+S25*jacobian_submatrix_6[3];

	// Columns.
	ja[offset6 +  0] = BASE + 0 * dim + IDX(i - 4, j    );
	ja[offset6 +  1] = BASE + 0 * dim + IDX(i - 3, j    );
	ja[offset6 +  2] = BASE + 0 * dim + IDX(i - 2, j    );
	ja[offset6 +  3] = BASE + 0 * dim + IDX(i - 1, j    );
	ja[offset6 +  4] = BASE + 0 * dim + IDX(i    , j - 3);
	ja[offset6 +  5] = BASE + 0 * dim + IDX(i    , j - 2);
	ja[offset6 +  6] = BASE + 0 * dim + IDX(i    , j - 1);
	ja[offset6 +  7] = BASE + 0 * dim + IDX(i    , j    );
	ja[offset6 +  8] = BASE + 0 * dim + IDX(i    , j + 1);
	ja[offset6 +  9] = BASE + 0 * dim + IDX(i + 1, j    );
	ja[offset6 + 10] = BASE + 1 * dim + IDX(i - 3, j    );
	ja[offset6 + 11] = BASE + 1 * dim + IDX(i - 2, j    );
	ja[offset6 + 12] = BASE + 1 * dim + IDX(i - 1, j    );
	ja[offset6 + 13] = BASE + 1 * dim + IDX(i    , j - 3);
	ja[offset6 + 14] = BASE + 1 * dim + IDX(i    , j - 2);
	ja[offset6 + 15] = BASE + 1 * dim + IDX(i    , j - 1);
	ja[offset6 + 16] = BASE + 1 * dim + IDX(i    , j    );
	ja[offset6 + 17] = BASE + 1 * dim + IDX(i    , j + 1);
	ja[offset6 + 18] = BASE + 1 * dim + IDX(i + 1, j    );
	ja[offset6 + 19] = BASE + 2 * dim + IDX(i - 4, j    );
	ja[offset6 + 20] = BASE + 2 * dim + IDX(i - 3, j    );
	ja[offset6 + 21] = BASE + 2 * dim + IDX(i - 2, j    );
	ja[offset6 + 22] = BASE + 2 * dim + IDX(i - 1, j    );
	ja[offset6 + 23] = BASE + 2 * dim + IDX(i    , j - 3);
	ja[offset6 + 24] = BASE + 2 * dim + IDX(i    , j - 2);
	ja[offset6 + 25] = BASE + 2 * dim + IDX(i    , j - 1);
	ja[offset6 + 26] = BASE + 2 * dim + IDX(i    , j    );
	ja[offset6 + 27] = BASE + 2 * dim + IDX(i    , j + 1);
	ja[offset6 + 28] = BASE + 2 * dim + IDX(i + 1, j    );
	ja[offset6 + 29] = BASE + 4 * dim + IDX(i - 3, j    );
	ja[offset6 + 30] = BASE + 4 * dim + IDX(i - 2, j    );
	ja[offset6 + 31] = BASE + 4 * dim + IDX(i - 1, j    );
	ja[offset6 + 32] = BASE + 4 * dim + IDX(i    , j    );
	ja[offset6 + 33] = BASE + 4 * dim + IDX(i + 1, j    );
	ja[offset6 + 34] = BASE + 5 * dim + IDX(i - 4, j    );
	ja[offset6 + 35] = BASE + 5 * dim + IDX(i - 3, j    );
	ja[offset6 + 36] = BASE + 5 * dim + IDX(i - 2, j    );
	ja[offset6 + 37] = BASE + 5 * dim + IDX(i - 1, j    );
	ja[offset6 + 38] = BASE + 5 * dim + IDX(i    , j - 4);
	ja[offset6 + 39] = BASE + 5 * dim + IDX(i    , j - 3);
	ja[offset6 + 40] = BASE + 5 * dim + IDX(i    , j - 2);
	ja[offset6 + 41] = BASE + 5 * dim + IDX(i    , j - 1);
	ja[offset6 + 42] = BASE + 5 * dim + IDX(i    , j    );
	ja[offset6 + 43] = BASE + 5 * dim + IDX(i    , j + 1);
	ja[offset6 + 44] = BASE + 5 * dim + IDX(i + 1, j    );


	// All done.
	return;
}