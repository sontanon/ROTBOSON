# Installation Instructions

## Prerequisites

This assumes you are running this on Linux with GCC. Before running the makefile, you must setup two libraries.

### MKL 

Download Intel's MKL and follow the installation instructions [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&linux-install=online). As of June 2024, this is:

```bash
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/2f3a5785-1c41-4f65-a2f9-ddf9e0db3ea0/l_onemkl_p_2024.1.0.695.sh

sudo sh ./l_onemkl_p_2024.1.0.695.sh
```

By following the default installation, this will install MKL at  `/opt/intel/oneapi`. In particular, there should be a script named `setvars.sh`. By executing the following command, you will setup all the necessary MKL dependencies. 

```bash
source /opt/intel/oneapi/setvars.sh
```

You can then verify that `MKLROOT` exists in your environment by running

```bash 
echo $MKLROOT 
``` 

In my case, this will output `/opt/intel/oneapi/mkl/2024.1`.

### Libconfig

Install libconfig by going [here](https://hyperrealm.github.io/libconfig/), download the latest tarball and follow the install instructions. In my case this was 

```bash
tar -xf libconfig-1.7.3.tar.gz
cd libconfig-1.7.3
./configure 
make 
sudo make install
```

Notice the last `sudo` call: this will install the header and libraries to `/usr/local`. You should keep track of where you installed libconfig.

## Compilation

After setting up these two dependencies, you are almost ready to compile. The provided `env.bash` will make sure that MKL and libconfig are available to the compiler and linker. By executing 

```bash
source env.bash
``` 

you should have `MKLROOT` and `LIBCONFIGROOT` properly setup. If you installed MKL or libconfig elsewhere, change the script as needed.

Then, compile using 

```bash
make all 
``` 

This will generate an executable in the current directory: `ROTBOSON`. 

# Generating l=1 data

I have provided two parameter files to generate $l=1$ data in `out`. 

Execute `l1_from_scratch.par` first. For example:

```bash
cd out 
../ROTBOSON l1_from_scratch.par
``` 

This should generate initial data for $l=1$, $m=1$, $\omega=0.95$. The output should be a directory named `l=1,w=9.50000E-01,dr=6.25000E-02,N=0256` if you keep the parameters unchanged.

Then, you can use the other parameter file to generate a lot more solutions (by using the previous "seed"):

```bash
../ROTBOSON l1_from_initial_data.par
``` 

This will execute for a while (for this configuration it will generate solutions up to $\omega = 0.675222$, at which point it will exit because the scalar field is too "spiky" and the grid resolution should be increased).

# TODO

* Explain $l \geq 2$.
* Explain interpolation as initial data.
* Explain the nonlinear solver.
* Redo everything in a friendlier language... 😂