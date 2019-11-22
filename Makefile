#CC = icc
CC = gcc

#CFLAGS = -DMKL_ILP64 -Wall -O3 -qopenmp
CFLAGS = -DMKL_ILP64 -Wall -O3 -fopenmp -m64

INCLUDES = -I${MKLROOT}/include 

#LFLAGS = -L${MKLROOT}/lib/intel64 
LFLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed

#LIBS = -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -lconfig
LIBS = -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -lconfig

SRC_DIR := ./src
OBJ_DIR := ./obj

SRCS = $(wildcard $(SRC_DIR)/*.c)
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
