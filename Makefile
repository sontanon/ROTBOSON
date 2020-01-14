CC = icc
#CC = gcc

CFLAGS = -DMKL_ILP64 -Wall -O3 -qopenmp
#CFLAGS = -DMKL_ILP64 -Wall -fopenmp -m64 -O3 #-ggdb

INCLUDES = -I${MKLROOT}/include 

LFLAGS = -L${MKLROOT}/lib/intel64 -L/usr/lib/x86_64-linux-gnu
#LFLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed

LIBS = -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -lconfig
#LIBS = -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -lconfig

SRC_DIR := ./src
OBJ_DIR := ./obj

#SRCS = $(wildcard $(SRC_DIR)/*.c)
SRCS = src/csr_exp_decay.c src/csr_omega_constraint.c src/csr_robin.c src/csr_symmetry.c src/csr_vars.c src/derivatives.c src/initial.c src/io.c src/main.c src/nleq_err.c src/nleq_res.c src/omega_calc.c src/pardiso_solve.c src/pardiso_start.c src/pardiso_stop.c src/parser.c src/qnerr.c src/rhs_vars.c src/rhs.c src/tools.c src/vector_algebra.c src/csr.c src/csr_grid_fill.c
OBJS = $(subst src/,obj/,$(subst .c,.o,$(SRCS)))

MAIN = ROTBOSON

all: $(MAIN)
	@echo Executable has been compiled.

obj/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(MAIN): $(OBJS)
	@echo "Compiling object files..."
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

clean:
	rm -rf obj/*.o $(MAIN)
