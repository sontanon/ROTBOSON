#include "tools.h"

void io(const char *dirname, const char *parfile)
{
	char opt;
	struct stat st = { 0 };
	if (stat(dirname, &st) == -1)
	{
#ifdef WIN
		_mkdir(dirname);
#else
		mkdir(dirname, 0755);
#endif
	}
	else
	{
		printf("***                                                \n");
		printf("***           WARNING: Output directory            \n");
		printf("***                    %-18s          \n", dirname);
		printf("***                    already exists!             \n");
		printf("***                                                \n");
		printf("***            Press (y/n) to proceed: ");
		opt = getchar();
		if ((opt == 'y') || (opt == 'Y'))
		{
			printf("***                                                \n");
			printf("***             User chose to continue.            \n");
			printf("***                                                \n");
		}
		else
		{
			printf("***                                                \n");
			printf("***               User chose to abort.             \n");
			printf("***                                                \n");
			printf("******************************************************\n");
			printf("******************************************************\n");
			exit(1);
		}
	}

	// Copy parameter file to directory.
	/*
	char cmd[256];
	memset(cmd, 0, 256);
#ifdef WIN
	sprintf(cmd, "COPY %s %s", parfile, dirname);
#else
	sprintf(cmd, "cp %s %s", parfile, dirname);
#endif
	// Check for errors.
	if (system(cmd) == -1)
	{
		printf("***                                                \n");
		printf("***           WARNING: Could not copy par          \n");
		printf("***                    file to output_dir!         \n");
		printf("***                                                \n");
		printf("***            Press (y/n) to ignore: ");

		// Ask user whether or not to proceed.
		opt = getchar();
		if ((opt == 'y') || (opt == 'Y'))
		{
			printf("***                                                \n");
			printf("***             User chose to continue.            \n");
			printf("***                                                \n");
		}
		else
		{
			printf("***                                                \n");
			printf("***               User chose to abort.             \n");
			printf("***                                                \n");
			printf("******************************************************\n");
			printf("******************************************************\n");
			exit(1);
		}
	}
	*/
	// Cd to output directory.
	if (chdir(dirname) == -1)
	{
		printf("***                                                \n");
		printf("***           WARNING: Could not cd to             \n");
		printf("***                    output directory!           \n");
		printf("***                                                \n");
		printf("***            Press (y/n) to write in             \n");
		printf("***               current directory: ");

		// Ask user whether or not to proceed.
		opt = getchar();
		if ((opt == 'y') || (opt == 'Y'))
		{
			printf("***                                                \n");
			printf("***             User chose to continue.            \n");
			printf("***                                                \n");
		}
		else
		{
			printf("***                                                \n");
			printf("***               User chose to abort.             \n");
			printf("***                                                \n");
			printf("******************************************************\n");
			printf("******************************************************\n");
			exit(1);
		}
	}

	// All done.
	return;
}
