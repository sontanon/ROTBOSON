#include "tools.h"

// Interpolate into axis value.
double axis_i(const double u, const double Dr_u, const double Drr_u, const double dr)
{
	return u - 0.3125 * dr * Dr_u + 0.03125 * dr * dr * Drr_u;
}

// Differentiate with respect to r on axis.
double axis_Drr_u(const double axis_u, double *u, const double dr, const MKL_INT j, const MKL_INT ghost, const MKL_INT NrTotal, const MKL_INT NzTotal)
{
	return (-u[IDX(ghost + 1, j)] + 81.0 * u[IDX(ghost, j)] - 80.0 * axis_u) / (9.0 * dr * dr);
}

double axis_Drrrr_u(const double axis_u, double *u, const double dr, const MKL_INT j, const MKL_INT ghost, const MKL_INT NrTotal, const MKL_INT NzTotal)
{
	return (-6.0 * u[IDX(ghost + 2, j)] + 130.0 * u[IDX(ghost + 1, j)] - 1020.0 * u[IDX(ghost, j)] + 896.0 * axis_u) / (15.0 * dr * dr * dr * dr);
}

// Regularization lambda at origin for all l > 1.
double lambda_A(const double H00, const double H01, const double H20, const double alpha00, const double alpha01, const double alpha20)
{
	return pow(H01,2)/(4.*H00) + H20 + (H01*alpha01)/alpha00 + (2*H00*alpha20)/alpha00;
}

// Add this term to above for l = 1.
double lambda_B(const double H00, const double psi00)
{
	return 8.0 * M_PI * H00 * psi00 * psi00;
}

// Regularization Drr_lambda. This is the first term which is only a function of H.
double Drr_lambda_A(const double H00, const double H01, const double H02, const double H03, const double H20, const double H21, const double H22, const double H40)
{
	return (-pow(H01,4)/(6.*pow(H00,3)) + (3*pow(H01,2)*H02)/(8.*pow(H00,2)) - pow(H02,2)/(12.*H00) 
	- (H01*H03)/(12.*H00) + (7*pow(H01,2)*H20)/(24.*pow(H00,2)) + (17*pow(H20,2))/(12.*H00) + (5*H01*H21)/(12.*H00) 
	- H22/6. + H40/18.);
}

// Next, add this term which contains alpha.
double Drr_lambda_B(const double H00, const double H01, const double H02, const double H03, const double H20, const double H21, const double H22, const double H40,
	const double alpha00, const double alpha01, const double alpha02, const double alpha03, const double alpha20, const double alpha21, const double alpha22, const double alpha40,
	const double beta01)
{
	return ((5*pow(H01,3)*alpha01)/(24.*pow(H00,2)*alpha00) + (H01*H02*alpha01)/(3.*H00*alpha00) - (H03*alpha01)/(6.*alpha00) + 
   (13*H01*H20*alpha01)/(6.*H00*alpha00) + (H21*alpha01)/(6.*alpha00) + (pow(H01,2)*pow(alpha01,2))/(2.*H00*pow(alpha00,2)) + 
   (H02*pow(alpha01,2))/(3.*pow(alpha00,2)) - (H01*pow(alpha01,3))/(3.*pow(alpha00,3)) + (pow(H01,2)*alpha02)/(3.*H00*alpha00) - 
   (H02*alpha02)/(3.*alpha00) + (H01*alpha01*alpha02)/(2.*pow(alpha00,2)) - (H01*alpha03)/(6.*alpha00) + (13*pow(H01,2)*alpha20)/(12.*H00*alpha00) - 
   (H02*alpha20)/(3.*alpha00) + (5*H20*alpha20)/alpha00 + (19*H01*alpha01*alpha20)/(6.*pow(alpha00,2)) - 
   (2*H00*pow(alpha01,2)*alpha20)/(3.*pow(alpha00,3)) + (H00*alpha02*alpha20)/(3.*pow(alpha00,2)) + (3*H00*pow(alpha20,2))/pow(alpha00,2) + 
   (H01*alpha21)/(6.*alpha00) + (2*H00*alpha01*alpha21)/(3.*pow(alpha00,2)) - (H00*alpha22)/(3.*alpha00) + (H00*alpha40)/(9.*alpha00) + 
   (pow(H00,2)*pow(beta01,2))/(4.*pow(alpha00,2)));
}

// For l = 2, add this term.
double Drr_lambda_C(const double H00, const double psi00)
{
	return 32.0 * M_PI * H00 * psi00 * psi00 / 3.0;
}

// For l = 1, add this term instead.
double Drr_lambda_D(const double H00, const double H01, const double H02, const double H03, const double H20, const double H21, const double H22, const double H40,
	const double alpha00, const double alpha01, const double alpha02, const double alpha03, const double alpha20, const double alpha21, const double alpha22, const double alpha40,
	const double beta00, const double psi00, const double psi01, const double psi02, const double w, const double m)
{
	return ((5*pow(H01,2)*M_PI*pow(psi00,2))/H00 - (4*H02*M_PI*pow(psi00,2))/3. + (64*H20*M_PI*pow(psi00,2))/3. + 
   (4*pow(H00,2)*pow(m,2)*M_PI*pow(psi00,2))/3. + (16*H01*M_PI*alpha01*pow(psi00,2))/alpha00 + 
   (32*H00*M_PI*alpha20*pow(psi00,2))/alpha00 + (224*H00*pow(M_PI,2)*pow(psi00,4))/3. - 4*H00*M_PI*pow(psi01,2) - 
   (8*H00*M_PI*psi00*psi02)/3. - (4*pow(H00,2)*M_PI*pow(psi00,2)*pow(w,2))/(3.*pow(alpha00,2)) - 
   (8*pow(H00,2)*M_PI*pow(psi00,2)*w*beta00)/(3.*pow(alpha00,2)) - 
   (4*pow(H00,2)*M_PI*pow(psi00,2)*pow(beta00,2))/(3.*pow(alpha00,2)));
}
