# --------------------------------------------------------- 
# DEFAULTS AND INITIALIZATION.
# --------------------------------------------------------- 
# C compiler.
CC = gcc
# C flags for 64-bit architecture, OpenMP and optimization.
CFLAGS = -DMKL_ILP64 -Wall -m64 -O3 # -ggdb
# Linker flags.
LDFLAGS = 
# MKL.
MKL_INCLUDE = -I${MKLROOT}/include
MKL_LD_PATH = -L${MKLROOT}/lib/intel64
MKL_LIBS = -lmkl_intel_ilp64 -lmkl_core 
# Libconfig.
LIBCONFIG_INCLUDE = -I${LIBCONFIGROOT}/include
LIBCONFIG_LD_PATH = -L${LIBCONFIGROOT}/lib -L${LIBCONFIGROOT}/lib/x86_64-linux-gnu
LIBCONFIG_LIB = -lconfig
# OpenMP.
OMP_LIB = 
# Other Libraries.
OTHER_LIBS = -lpthread -lm -ldl

# --------------------------------------------------------- 
# Help.
# --------------------------------------------------------- 
help:
	@echo "ROTBOSON help."
	@echo "" 
	@echo "Usage: make Target [Options...]"
	@echo "" 
	@echo "   Target:"
	@echo "      v_omega	- Compile for variable omega."
	@echo "      f_omega	- Compile for fixed omega."
	@echo "      clean	- Remove binaries and executable."
	@echo "      check_env	- Check if environment variables are set."
	@echo "      help	- Print this help."
	@echo "" 
	@echo "   Options:"
	@echo "      compiler={gnu|intel}"
	@echo "         Specifies whether to use GNU's gcc or Intel's icc C compiler."
	@echo "         Default = gnu."
	@echo ""

# --------------------------------------------------------- 
# Check compiler options and then for errors.
# --------------------------------------------------------- 
MSG = 
ifneq ($(compiler),gnu)
 ifneq ($(compiler),intel)
  MSG += compiler = $(compiler)
 endif
endif

ifneq ("$(MSG)","")
 WRONG_OPTION = \n\n*** COMMAND LINE ERROR: Wrong value of option(s): $(MSG)\n\n
 TARGET = help
endif

# -----------------------------------------------------------------------------
# SETUP VARIABLES.
# -----------------------------------------------------------------------------
ifeq ($(compiler),intel)
 override CC = icc
 # Intel OpenMP.
 CFLAGS += -qopenmp
 override OMP_LIB = -liomp5
 # MKL intrinsic library.
 MKL_LIBS += -lmkl_intel_thread
else
 override CC = gcc
 # Modify flags for OpenMP.
 CFLAGS += -fopenmp
 override OMP_LIB = -lgomp
 # MKL intrinsic library.
 MKL_LIBS += -lmkl_gnu_thread
 # Modify linker flags.
 LDFLAGS += -Wl,--no-as-needed
endif

# -----------------------------------------------------------------------------
# SETUP VARIABLES.
# -----------------------------------------------------------------------------
SRC_DIR := ./src
OBJ_DIR := ./obj
OUT_DIR := ./out

# Create build directory if necessary.
$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(OUT_DIR))

#SRCS = $(wildcard $(SRC_DIR)/*.c)
SRCS = src/bicubic_interpolation.c src/cart_to_pol.c src/low_rank.c src/csr_exp_decay.c src/csr_omega_constraint.c src/csr_robin.c src/csr_symmetry.c src/csr_vars.c src/derivatives.c src/initial.c src/io.c src/main.c src/nleq_err.c src/nleq_res.c src/omega_calc.c src/pardiso_solve.c src/pardiso_start.c src/pardiso_stop.c src/parser.c src/qnerr.c src/rhs_vars.c src/rhs.c src/tools.c src/vector_algebra.c src/csr.c src/csr_grid_fill.c
OBJS = $(subst src/,obj/,$(subst .c,.o,$(SRCS)))

MAIN = ROTBOSON

# --------------------------------------------------------- 
# Check environment.
# --------------------------------------------------------- 
# check_env:
# ifdef MKLROOT
# 	$(shell echo MKLROOT set to ${MKLROOT})
# endif
# ifndef MKLROOT
# 	$(shell echo Critical ERROR!)
# 	$(shell echo MKLROOT must be a defined environment variable.)
# 	$(shell echo Please set MKLROOT to MKL\'s installation directory, usually /opt/intel/mkl)
# endif
# 
# ifdef LIBCONFIGROOT
# 	$(shell echo LIBCONFIGROOT set to ${LIBCONFIGROOT})
# endif
# ifndef LIBCONFIGROOT
# 	$(shell echo Critical ERROR!)
# 	$(shell echo LIBCONFIGROOT must be a defined environment variable.)
# 	$(shell echo Please set LIBCONFIGROOT to libconfig\'s installation directory, usually /usr)
# endif

all: $(MAIN)
	@echo Executable has been compiled.

obj/%.o: src/%.c
	$(CC) $(CFLAGS) $(MKL_INCLUDE) $(LIBCONFIG_INCLUDE) -c $< -o $@

$(MAIN): $(OBJS)
	@echo "Compiling object files..."
	$(CC) $(CFLAGS) $(MKL_INCLUDE) $(LIBCONFIG_INCLUDE) -o $(MAIN) $(OBJS) $(LDFLAGS) $(MKL_LD_PATH) $(LIBCONFIG_LD_PATH) $(MKL_LIBS) $(LIBCONFIG_LIB) $(OMP_LIB) $(OTHER_LIBS)
#	$(shell cp $(MAIN) $(OUT_DIR))

clean:
	rm -rf obj/*.o $(MAIN) $(OUT_DIR)/$(MAIN)
