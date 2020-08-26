#include "tools.h"
#include "param.h"

// Macros for parameter ranges.
#define MAX_DR 1.0
#define MIN_DR 0.001

#define MAX_NRINTERIOR 9999999LL
#define MIN_NRINTERIOR 16LL

#define MAX_L 6LL
#define MIN_L 1LL

#define MAX_M 1.0E+3
#define MIN_M 1.0E-3

#define MAX_PSI0 1.0E+5
#define MIN_PSI0 1.0E-25

#define MAX_SIGMA 1.0E+3
#define MIN_SIGMA 1.0E-3

#define MAX_R_EXT 1.0E+3
#define MIN_R_EXT 1.0E-3

#define MAX_W0 1.0
#define MIN_W0 0.0

#define MAX_MAXITER 100000LL
#define MIN_MAXITER 0LL

#define MIN_WEIGHT 1.0E-26

#define MAX_EPS 1.0E-1
#define MIN_EPS 1.0E-16

void parser(const char *fname)
{
	// Initialize cfg.
	config_init(&cfg);

	// Read the file. If there is an error, report and exit.
	if (!config_read_file(&cfg, fname))
	{
		fprintf(stderr, "PARSER: CRITICAL ERROR IN FILE!\n");
		fprintf(stderr, "%s:%d - %s\n", config_error_file(&cfg),
		config_error_line(&cfg), config_error_text(&cfg));
		config_destroy(&cfg);
		exit(-1);
	}

	// Parse arguments doing sanity checks.

	// GRID.
	// dr.
	if (config_lookup_float(&cfg, "dr", &dr) == CONFIG_TRUE)
	{
		if (MAX_DR < dr || dr < MIN_DR)
		{
			fprintf(stderr, "PARSER: ERROR! dr = %3.5E is not in range [%3.5E, %3.5E]\n", dr, MIN_DR, MAX_DR);
			fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"dr\" from parameter file. Setting to default value, dr = %3.5E\n", dr);
	}
	// dz.
	if (config_lookup_float(&cfg, "dz", &dz) == CONFIG_TRUE)
	{
		if (MAX_DR < dz || dz < MIN_DR)
		{
			fprintf(stderr, "PARSER: ERROR! dz = %3.5E is not in range [%3.5E, %3.5E]\n", dz, MIN_DR, MAX_DR);
			fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"dz\" from parameter file. Setting to default value, dz = %3.5E\n", dz);
	}
	// NrInterior.
	if (config_lookup_int64(&cfg, "NrInterior", &NrInterior) == CONFIG_TRUE)
	{
		if (MAX_NRINTERIOR < NrInterior || NrInterior < MIN_NRINTERIOR)
		{
			fprintf(stderr, "PARSER: ERROR! NrInterior = %lld is not in range [%lld, %lld]\n", NrInterior, MIN_NRINTERIOR, MAX_NRINTERIOR);
			fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"NrInterior\" from parameter file. Setting to default value, NrInterior = %lld\n", NrInterior);
	}
	// NzInterior.
	if (config_lookup_int64(&cfg, "NzInterior", &NzInterior) == CONFIG_TRUE)
	{
		if (MAX_NRINTERIOR < NzInterior || NzInterior < MIN_NRINTERIOR)
		{
			fprintf(stderr, "PARSER: ERROR! NzInterior = %lld is not in range [%lld, %lld]\n", NzInterior, MIN_NRINTERIOR, MAX_NRINTERIOR);
			fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"NzInterior\" from parameter file. Setting to default value, NzInterior = %lld\n", NzInterior);
	}
	// order.
	if (config_lookup_int64(&cfg, "order", &order) == CONFIG_TRUE)
	{
		if (order != 2 && order != 4)
		{
			fprintf(stderr, "PARSER: ERROR! order = %lld is not supported. Only 2 or 4 are supported finite difference orders.\n", order);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"order\" from parameter file. Setting to default value, order = %lld\n", order);
	}
	// Determine parity ghost zones.
	if (order == 2)
	{
		ghost = 1;
	}
	else if (order == 4)
	{
		ghost = 2;
	}

	// DO NOT FORGET TO CALCULATE NRTOTAL, NZTOTAL, DIM, AND W_IDX!
	NrTotal = NrInterior + 2 * ghost;
	NzTotal = NzInterior + 2 * ghost;
	dim = NrTotal * NzTotal;
	w_idx = GNUM * dim;

	// SCALAR FIELD PARAMETERS.
	// l.
	if (config_lookup_int64(&cfg, "l", &l) == CONFIG_TRUE)
	{
		if (MAX_L < l || l < MIN_L)
		{
			fprintf(stderr, "PARSER: ERROR! l = %lld is not in range [%lld, %lld]\n", l, MIN_L, MAX_L);
			fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"l\" from parameter file. Setting to default value, l = %lld\n", l);
	}
	// m.
	if (config_lookup_float(&cfg, "m", &m) == CONFIG_TRUE)
	{
		if (MAX_M < m || m < MIN_M)
		{
			fprintf(stderr, "PARSER: ERROR! m = %3.5E is not in range [%3.5E, %3.5E]\n", m, MIN_M, MAX_M);
			fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"m\" value from parameter file. Setting to default value, m = %3.5E\n", m);
	}
	// fixedPhi.
	if (config_lookup_int64(&cfg, "fixedPhi",&fixedPhi) == CONFIG_TRUE)
	{
		if (fixedPhi != 0 && fixedPhi != 1)
		{
			fprintf(stderr, "PARSER: ERROR! fixedPhi = %lld must be a boolean value 0 or 1.\n", fixedPhi);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	// fixedOmega.
	if (config_lookup_int64(&cfg, "fixedOmega",&fixedOmega) == CONFIG_TRUE)
	{
		if (fixedOmega != 0 && fixedOmega != 1)
		{
			fprintf(stderr, "PARSER: ERROR! fixedOmega = %lld must be a boolean value 0 or 1.\n", fixedOmega);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	// Assert that one of the two last variables is fixed.
	if (!fixedOmega && !fixedPhi)
	{
		fprintf(stderr, "PARSER: ERROR! fixedPhi = %lld and fixedOmega = %lld. One quantity must be held fixed.\n", fixedPhi, fixedOmega);
		fprintf(stderr, "        Please specify whether Phi or Omega is held fixed in parameter file.\n");
		exit(-1);
	}
	if (fixedOmega && fixedPhi)
	{
		fprintf(stderr, "PARSER: ERROR! fixedPhi = %lld and fixedOmega = %lld. Only one variable can be fixed.\n", fixedPhi, fixedOmega);
		fprintf(stderr, "        Please specify which variable (and only one variable) is to be fixed.\n");
		exit(-1);
	}

	// Read fixed coordinates.
	if (fixedPhi)
	{
		// fixedPhiR.
		if (config_lookup_int64(&cfg, "fixedPhiR", &fixedPhiR) == CONFIG_TRUE)
		{
			if (NrInterior < fixedPhiR || fixedPhiR < 1)
			{
				fprintf(stderr, "PARSER: ERROR! fixedPhiR = %lld is not in range [%lld, %lld]\n", fixedPhiR, 1LL, NrInterior);
				fprintf(stderr, "        Please input proper value in parameter file.\n");
				exit(-1);
			}
		}
		else
		{
			fprintf(stderr, "PARSER: WARNING! Could not properly read \"fixedPhiR\" from parameter file. Setting to default value, fixedPhiR = %lld\n", fixedPhiR);
		}
		// fixedPhiZ.
		if (config_lookup_int64(&cfg, "fixedPhiZ", &fixedPhiZ) == CONFIG_TRUE)
		{
			if (NzInterior < fixedPhiZ || fixedPhiZ < 1)
			{
				fprintf(stderr, "PARSER: ERROR! fixedPhiZ = %lld is not in range [%lld, %lld]\n", fixedPhiZ, 1LL, NzInterior);
				fprintf(stderr, "        Please input proper value in parameter file.\n");
				exit(-1);
			}
		}
		else
		{
			fprintf(stderr, "PARSER: WARNING! Could not properly read \"fixedPhiZ\" from parameter file. Setting to default value, fixedPhiZ = %lld\n", fixedPhiZ);
		}
	}

	// INITIAL DATA.
	// readInitialData.
	if (config_lookup_int64(&cfg, "readInitialData", &readInitialData) == CONFIG_TRUE)
	{
		if (readInitialData != 0 && readInitialData != 1 && readInitialData != 2 && readInitialData != 3)
		{
			fprintf(stderr, "PARSER: ERROR! readInitialData = %lld is not supported. Only 0, 1, 2, or 3 as boolean values for indication of whether to read initial data specified by user.\n", readInitialData);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"readInitialData\" from parameter file. Setting to default value, readInitialData = %lld\n", readInitialData);
	}

	// Read initial data parameters.
	switch (readInitialData)
	{
		// Interpolation from different size and/or resolution grid.
		case 3:
			// Read filenames.
			if (config_lookup_string(&cfg, "log_alpha_i", &log_alpha_i) == CONFIG_FALSE)
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires values for all initial files. Did not find \"log_alpha_i\".\n");
				exit(-1);
			}
			if (config_lookup_string(&cfg, "beta_i", &beta_i) == CONFIG_FALSE)
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires values for all initial files. Did not find \"beta_i\".\n");
				exit(-1);
			}
			if (config_lookup_string(&cfg, "log_h_i", &log_h_i) == CONFIG_FALSE)
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires values for all initial files. Did not find \"log_h_i\".\n");
				exit(-1);
			}
			if (config_lookup_string(&cfg, "log_a_i", &log_a_i) == CONFIG_FALSE)
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires values for all initial files. Did not find \"log_a_i\".\n");
				exit(-1);
			}
			if (config_lookup_string(&cfg, "psi_i", &psi_i) == CONFIG_FALSE)
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires values for all initial files. Did not find \"psi_i\".\n");
				exit(-1);
			}
			if (config_lookup_string(&cfg, "lambda_i", &lambda_i) == CONFIG_FALSE)
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires values for all initial files. Did not find \"psi_i\".\n");
				exit(-1);
			}


			// Grid parameters.
			if (config_lookup_int64(&cfg, "NrTotalInitial", &NrTotalInitial) == CONFIG_TRUE)
			{
				if (MAX_NRINTERIOR < NrTotalInitial || NrTotalInitial < MIN_NRINTERIOR)
				{
					fprintf(stderr, "PARSER: ERROR! NrTotalInitial = %lld is not in range [%lld, %lld]\n", NrTotalInitial, MIN_NRINTERIOR, MAX_NRINTERIOR);
					fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
					exit(-1);
				}
			}
			else
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires value for NrTotalInitial.\n");
				exit(-1);
			}
			if (config_lookup_int64(&cfg, "NzTotalInitial", &NzTotalInitial) == CONFIG_TRUE)
			{
				if (MAX_NRINTERIOR < NzTotalInitial || NzTotalInitial < MIN_NRINTERIOR)
				{
					fprintf(stderr, "PARSER: ERROR! NzTotalInitial = %lld is not in range [%lld, %lld]\n", NzTotalInitial, MIN_NRINTERIOR, MAX_NRINTERIOR);
					fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
					exit(-1);
				}
			}
			else
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires value for NzTotalInitial.\n");
				exit(-1);
			}
			if (config_lookup_int64(&cfg, "order_i", &order_i) == CONFIG_TRUE)
			{
				if (order_i != 2 && order_i != 4)
				{
					fprintf(stderr, "PARSER: ERROR! order_i = %lld is not supported. Only 2 or 4 are supported finite difference orders.\n", order);
					fprintf(stderr, "        Please input proper value in parameter file.\n");
					exit(-1);
				}
			}
			else
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires value for order_i.\n");
				exit(-1);
			}
			if (config_lookup_int64(&cfg, "ghost_i", &ghost_i) == CONFIG_TRUE)
			{
				if (ghost_i != 1 && ghost_i != 2)
				{
					fprintf(stderr, "PARSER: ERROR! ghost_i = %lld is not supported. Only 1 or 2 are supported.\n", order);
					fprintf(stderr, "        Please input proper value in parameter file.\n");
					exit(-1);
				}
			}
			else
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires value for ghost_i.\n");
				exit(-1);
			}			
			if (config_lookup_float(&cfg, "dr_i", &dr_i) == CONFIG_TRUE)
			{
				if (MAX_DR < dr_i || dr_i < MIN_DR)
				{
					fprintf(stderr, "PARSER: ERROR! dr_i = %3.5E is not in range [%3.5E, %3.5E]\n", dr_i, MIN_DR, MAX_DR);
					fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
					exit(-1);
				}
			}
			else
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires value for dr_i.\n");
				exit(-1);
			}
			if (config_lookup_float(&cfg, "dz_i", &dz_i) == CONFIG_TRUE)
			{
				if (MAX_DR < dz_i || dz_i < MIN_DR)
				{
					fprintf(stderr, "PARSER: ERROR! dz_i = %3.5E is not in range [%3.5E, %3.5E]\n", dz_i, MIN_DR, MAX_DR);
					fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
					exit(-1);
				}
			}
			else
			{
				fprintf(stderr, "PARSER: ERROR! readInitialData = 3 requires value for dz_i.\n");
				exit(-1);
			}
			config_lookup_string(&cfg, "w_i", &w_i);

			break;
	
		// Default case for 1 or 2.
		default:
			config_lookup_string(&cfg, "log_alpha_i", &log_alpha_i);
			config_lookup_string(&cfg, "beta_i", &beta_i);
			config_lookup_string(&cfg, "log_h_i", &log_h_i);
			config_lookup_string(&cfg, "log_a_i", &log_a_i);
			config_lookup_string(&cfg, "psi_i", &psi_i);
			config_lookup_string(&cfg, "lambda_i", &psi_i);
			config_lookup_string(&cfg, "w_i", &w_i);
			
			// Initial Data extensions.
			if (readInitialData == 2)
			{
				// NrTotalInitial.
				config_lookup_int64(&cfg, "NrTotalInitial", &NrTotalInitial);
				// NzTotalInitial.
				config_lookup_int64(&cfg, "NzTotalInitial", &NzTotalInitial);
			}
			else
			{
				NrTotalInitial = NrTotal;
				NzTotalInitial = NzTotal;
			}

			// psi0.
			if (config_lookup_float(&cfg, "psi0", &psi0) == CONFIG_TRUE)
			{
				if (MAX_PSI0 < psi0 || psi0 < MIN_PSI0)
				{
					fprintf(stderr, "PARSER: ERROR! psi0 = %3.5E is not in range [%3.5E, %3.5E]\n", psi0, MIN_PSI0, MAX_PSI0);
					fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
					exit(-1);
				}
			}
			break;
	}
	// Generate via analytic guess.
	if (!readInitialData)
	{
		// psi0.
		if (config_lookup_float(&cfg, "psi0", &psi0) == CONFIG_TRUE)
		{
			if (MAX_PSI0 < psi0 || psi0 < MIN_PSI0)
			{
				fprintf(stderr, "PARSER: ERROR! psi0 = %3.5E is not in range [%3.5E, %3.5E]\n", psi0, MIN_PSI0, MAX_PSI0);
				fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
				exit(-1);
			}
		}
		else
		{
			fprintf(stderr, "PARSER: WARNING! Could not properly read \"psi0\" value from parameter file. Setting to default value, psi0 = %3.5E\n", psi0);
		}
		// sigmaR.
		if (config_lookup_float(&cfg, "sigmaR", &sigmaR) == CONFIG_TRUE)
		{
			if (MAX_SIGMA < sigmaR || sigmaR < MIN_SIGMA)
			{
				fprintf(stderr, "PARSER: ERROR! sigmaR = %3.5E is not in range [%3.5E, %3.5E]\n", sigmaR, MIN_SIGMA, MAX_SIGMA);
				fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
				exit(-1);
			}
		}
		else
		{
			fprintf(stderr, "PARSER: WARNING! Could not properly read \"sigmaR\" value from parameter file. Setting to default value, sigmaR = %3.5E\n", sigmaR);
		}
		// sigmaZ.
		if (config_lookup_float(&cfg, "sigmaZ", &sigmaZ) == CONFIG_TRUE)
		{
			if (MAX_SIGMA < sigmaZ || sigmaZ < MIN_SIGMA)
			{
				fprintf(stderr, "PARSER: ERROR! sigmaZ = %3.5E is not in range [%3.5E, %3.5E]\n", sigmaZ, MIN_SIGMA, MAX_SIGMA);
				fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
				exit(-1);
			}
		}
		else
		{
			fprintf(stderr, "PARSER: WARNING! Could not properly read \"sigmaZ\" value from parameter file. Setting to default value, sigmaZ = %3.5E\n", sigmaZ);
		}
		// rExt.
		if (config_lookup_float(&cfg, "rExt", &rExt) == CONFIG_TRUE)
		{
			if (MAX_R_EXT < rExt || rExt < MIN_R_EXT)
			{
				fprintf(stderr, "PARSER: ERROR! rExt = %3.5E is not in range [%3.5E, %3.5E]\n", rExt, MIN_R_EXT, MAX_R_EXT);
				fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
				exit(-1);
			}
		}
		else
		{
			fprintf(stderr, "PARSER: WARNING! Could not properly read \"rExt\" value from parameter file. Setting to default value, rExt = %3.5E\n", rExt);
		}
	}
	// Initial frequency.
	if (!w_i)
	{
		// w0.
		if (config_lookup_float(&cfg, "w0", &w0) == CONFIG_TRUE)
		{
			if (MAX_W0 < w0 / m || w0 / m < MIN_W0)
			{
				fprintf(stderr, "PARSER: ERROR! (w0 / m) = (%3.5E / m) is not in range (%3.5E, %3.5E)\n", w0, MIN_W0, MAX_W0);
				fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
				exit(-1);
			}
		}
		else
		{
			fprintf(stderr, "PARSER: WARNING! Could not properly read \"w0\" value from parameter file. Setting to default value, w0 = %3.5E\n", w0);
		}	
	}

	// SOLVER PARAMETERS.
	// solverType.
	if (config_lookup_int64(&cfg, "solverType", &solverType) == CONFIG_TRUE)
	{
		if (solverType != 1 && solverType != 2)
		{
			fprintf(stderr, "PARSER: ERROR! solverType = %lld is not supported. Only 1 or 2 are supported for indication of whether to use error or residual based solver.\n", solverType);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"solverType\" from parameter file. Setting to default value, solverType = %lld\n", solverType);
	}
	// localSolver.
	if (config_lookup_int64(&cfg, "localSolver", &localSolver) == CONFIG_TRUE)
	{
		if (localSolver != 0 && localSolver != 1)
		{
			fprintf(stderr, "PARSER: ERROR! localSolver = %lld is not supported. Only 0 or 1 boolean are supported for indication of whether to use local solver inside global solver.\n", localSolver);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"localSolver\" from parameter file. Setting to default value, localSolver = %lld\n", localSolver);
	}
	// epsilon.
	if (config_lookup_float(&cfg, "epsilon", &epsilon) == CONFIG_TRUE)
	{
		if (MAX_EPS < epsilon || epsilon < MIN_EPS)
		{
			fprintf(stderr, "PARSER: ERROR! exit tolerance epsilon = %3.5E is not in range [%3.5E, %.35E]\n", epsilon, MIN_EPS, MAX_EPS);
			fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"epsilon\" from parameter file. Setting to default value, epsilon = %3.5E\n", epsilon);
	}
	// maxNewtonIter.
	if (config_lookup_int64(&cfg, "maxNewtonIter", &maxNewtonIter) == CONFIG_TRUE)
	{
		if (MAX_MAXITER < maxNewtonIter || maxNewtonIter < MIN_MAXITER)
		{
			fprintf(stderr, "PARSER: ERROR! maxNewtonIter = %lld is not in range [%lld, %lld]\n", maxNewtonIter, MIN_MAXITER, MAX_MAXITER);
			fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"maxNewtonIter\" from parameter file. Setting to default value, maxIter = %lld\n", maxNewtonIter);
	}
	// lambda0.
	if (config_lookup_float(&cfg, "lambda0", &lambda0) == CONFIG_TRUE)
	{
		if (1.0 < lambda0 || lambda0 <= MIN_WEIGHT)
		{
			fprintf(stderr, "PARSER: ERROR! initial damping factor lambda0 = %3.5E is not in range (%3.5E, 1.0]\n", lambda0, MIN_WEIGHT);
			fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"lambda0\" from parameter file. Setting to default value, lambda0 = %3.5E\n", lambda0);
	}
	// lambdaMin.
	if (config_lookup_float(&cfg, "lambdaMin", &lambdaMin) == CONFIG_TRUE)
	{
		if (lambda0 <= lambdaMin || lambdaMin < MIN_WEIGHT)
		{
			fprintf(stderr, "PARSER: ERROR! minimum damping factor lambdaMin = %3.5E is not in range [%3.5E, lambda0)\n", lambdaMin, MIN_WEIGHT);
			fprintf(stderr, "        Please edit range in \"parser.c\" source file or input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"lambdaMin\" from parameter file. Setting to default value, lambdaMin = %3.5E\n", lambdaMin);
	}
	// useLowRank.
	if (config_lookup_int64(&cfg, "useLowRank", &useLowRank) == CONFIG_TRUE)
	{
		if (useLowRank != 0 && useLowRank != 1)
		{
			fprintf(stderr, "PARSER: ERROR! useLowRank = %lld is not supported. Only 0 or 1 boolean are supported for indication of whether to use Low Rank Update.\n", useLowRank);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"useLowRank\" from parameter file. Setting to default value, useLowRank = %lld\n", useLowRank);
	}

	// BOUNDARY TYPES.
	// alphaBoundOrder.
	if (config_lookup_int64(&cfg, "alphaBoundOrder", &alphaBoundOrder) == CONFIG_TRUE)
	{
		if (alphaBoundOrder != 0 && alphaBoundOrder != 1 && alphaBoundOrder != 2)
		{
			fprintf(stderr, "PARSER: ERROR! alphaBoundOrder = %lld is not supported. Only 0, 1, or 2 supported orders.\n", order);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"alphaBoundOrder\" from parameter file. Setting to default value, alphaBoundOrder = %lld\n", alphaBoundOrder);
	}
	// betaBoundOrder.
	if (config_lookup_int64(&cfg, "betaBoundOrder", &betaBoundOrder) == CONFIG_TRUE)
	{
		if (betaBoundOrder != 0 && betaBoundOrder != 1 && betaBoundOrder != 2)
		{
			fprintf(stderr, "PARSER: ERROR! betaBoundOrder = %lld is not supported. Only 0, 1, or 2 supported orders.\n", order);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"betaBoundOrder\" from parameter file. Setting to default value, betaBoundOrder = %lld\n", betaBoundOrder);
	}
	// hBoundOrder.
	if (config_lookup_int64(&cfg, "hBoundOrder", &hBoundOrder) == CONFIG_TRUE)
	{
		if (hBoundOrder != 0 && hBoundOrder != 1 && hBoundOrder != 2)
		{
			fprintf(stderr, "PARSER: ERROR! hBoundOrder = %lld is not supported. Only 0, 1, or 2 supported orders.\n", order);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"hBoundOrder\" from parameter file. Setting to default value, hBoundOrder = %lld\n", hBoundOrder);
	}
	// aBoundOrder.
	if (config_lookup_int64(&cfg, "aBoundOrder", &aBoundOrder) == CONFIG_TRUE)
	{
		if (aBoundOrder != 0 && aBoundOrder != 1 && aBoundOrder != 2)
		{
			fprintf(stderr, "PARSER: ERROR! aBoundOrder = %lld is not supported. Only 0, 1, or 2 supported orders.\n", order);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"aBoundOrder\" from parameter file. Setting to default value, aBoundOrder = %lld\n", aBoundOrder);
	}
	// phiBoundOrder.
	if (config_lookup_int64(&cfg, "phiBoundOrder", &phiBoundOrder) == CONFIG_TRUE)
	{
		if (phiBoundOrder != 0 && phiBoundOrder != 1 && phiBoundOrder != 2)
		{
			fprintf(stderr, "PARSER: ERROR! phiBoundOrder = %lld is not supported. Only 0, 1, or 2 supported orders.\n", order);
			fprintf(stderr, "        Please input proper value in parameter file.\n");
			exit(-1);
		}
	}
	else
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"phiBoundOrder\" from parameter file. Setting to default value, phiBoundOrder = %lld\n", phiBoundOrder);
	}
	// OUTPUT
	// dirname.
	if (config_lookup_string(&cfg, "dirname", &dirname) != CONFIG_TRUE)
	{
		fprintf(stderr, "PARSER: WARNING! Could not properly read \"dirname\" from parameter file. Setting to default value, dirname = %s\n", dirname);
	}

	// All done.
	return;
}
