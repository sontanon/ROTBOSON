double axis_i(const double u, const double Dr_u, const double Drr_u, const double dr);
double axis_Drr_u(const double axis_u, double *u, const double dr, const MKL_INT j, const MKL_INT ghost, const MKL_INT NrTotal, const MKL_INT NzTotal);
double axis_Drrrr_u(const double axis_u, double *u, const double dr, const MKL_INT j, const MKL_INT ghost, const MKL_INT NrTotal, const MKL_INT NzTotal);
double lambda_A(const double H00, const double H01, const double H20, const double alpha00, const double alpha01, const double alpha20);
double lambda_B(const double H00, const double psi00);
double Drr_lambda_A(const double H00, const double H01, const double H02, const double H03, const double H20, const double H21, const double H22, const double H40);
double Drr_lambda_B(const double H00, const double H01, const double H02, const double H03, const double H20, const double H21, const double H22, const double H40,
	const double alpha00, const double alpha01, const double alpha02, const double alpha03, const double alpha20, const double alpha21, const double alpha22, const double alpha40,
	const double beta01);
double Drr_lambda_C(const double H00, const double psi00);
double Drr_lambda_D(const double H00, const double H01, const double H02, const double H03, const double H20, const double H21, const double H22, const double H40,
	const double alpha00, const double alpha01, const double alpha02, const double alpha03, const double alpha20, const double alpha21, const double alpha22, const double alpha40,
	const double beta00, const double psi00, const double psi01, const double psi02, const double w, const double m);