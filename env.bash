# Setup MKL by assuming that it was installed in the following location.
source /opt/intel/oneapi/setvars.sh

# Setup Libconfig by assuming it was installed in the following location.
export LIBCONFIGROOT=/usr/local

# Add MKL and Libconfig to the LD library.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MKLROOT/lib:$LIBCONFIGROOT/lib