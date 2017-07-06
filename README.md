This repository contains scripts to reproduce the results from the paper "The Tensor Algebra Compiler".

# Build
Build tacoBench using CMake 2.8.3 or greater:
```
cd <taco-bench>
mkdir build
cd build
cmake ..
make 
```

# EIGEN
Download and Extract Eigen's source code from https://eigen.tuxfamily.org/dox/GettingStarted.html
Specify the variable EIGEN_INCLUDE_DIR before using cmake

# GMM++
Use the installation guide http://getfem.org/gmm/install.html
Specify the variable GMM_INCLUDE_DIR before using cmake

# UBLAS
UBLAS is usually already installed in /usr/include
Documentation can be found at http://www.boost.org/doc/libs/1_45_0/libs/numeric/ublas/doc/index.htm
Specify the variable UBLAS_INCLUDE_DIR before using cmake

# OSKI
Follow the user's guide to install and tune OSKI
http://bebop.cs.berkeley.edu/oski/downloads.html
Specify OSKI_INCLUDE_DIR and OSKI_LIBRARY_DIR before using cmake

# POSKI
Follow the user's guide to install and tune pOSKI
http://bebop.cs.berkeley.edu/poski/
Specify POSKI_INCLUDE_DIR and POSKI_LIBRARY_DIR before using cmake
Nota: Install first OSKI and then use this installation to install POSKI

# MKL
Commercial licensed product from INTEL
https://software.intel.com/en-us/mkl
Specify MKL_ROOT before using cmake
Nota: use the mklvars.sh script of INTEL to set properly your environment

