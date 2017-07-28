This repository contains benchmarks for taco and scripts to reproduce the results from the paper *The Tensor Algebra Compiler*.

# Build taco-bench 
Tell taco-bench where taco is located:

    export TACO_INCLUDE_DIR=<taco-directory>/include
    export TACO_LIBRARY_DIR=<taco-build-directory>/lib

Build taco-bench with CMake 2.8.3 or greater:

    cd <taco-bench-directory>
    mkdir build
    cd build
    cmake ..
    make 

# Installing and building with other products

Do the following steps before you build taco-bench with cmake to benchmark against several libraries.

## EIGEN

1. Download and extract Eigen's source code from https://eigen.tuxfamily.org/dox/GettingStarted.html.
2. Specify the variable `EIGEN_DIR`.

## GMM++

1. Use the installation guide http://getfem.org/gmm/install.html.
2. Specify the variable `GMM_INCLUDE_DIR`.

## UBLAS

1. Follow the documentation to install ublas: http://www.boost.org/doc/libs/1_45_0/libs/numeric/ublas/doc/index.htm.
2. Specify the variable `UBLAS_INCLUDE_DIR`.

## OSKI

1. Follow the user guide to install and tune OSKI: http://bebop.cs.berkeley.edu/oski/downloads.html.
3. Specify `OSKI_INCLUDE_DIR` and `OSKI_LIBRARY_DIR`.

## POSKI

1. Follow the user guide to install and tune pOSKI: http://bebop.cs.berkeley.edu/poski/.
2. Specify `POSKI_INCLUDE_DIR` and `POSKI_LIBRARY_DIR`.

Note: Install first OSKI and then use this installation to install POSKI.

## MKL
1. Commercial licensed product from Intel: https://software.intel.com/en-us/mkl.
2. Specify `MKL_ROOT`.

Note: use the `mklvars.sh` script of Intel to set properly your environment.

# Benchmarking taco against your product

1. Modify CMakeList.txt: add an environment variable `YOUR`, compilation options, `YOUR_INCLUDE`, and `YOUR_LIBRARY` directories to your project.

2. Implement the expression using *your* project. Modify `your4taco.h` file.

3. Use the `TACO_BENCH` macro to benchmark and `validate` method to compare against expected results.

