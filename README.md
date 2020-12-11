# LEAP Accelerate

![License](https://img.shields.io/badge/license-LGPL_2.1-blue)
[![Build Status](https://travis-ci.com/ICRAR/leap-accelerate.svg?token=1YzqBsytWggkjwq3sjZP&branch=master)](https://travis-ci.com/ICRAR/leap-accelerate)

Low-frequency Excision of the Atmosphere in Parallel (LEAP) Calibration using GPU acceleration.

LEAP Accelerate includes:

* [leap-accelerate-lib](src/icrar/leap-accelerate/ReadMe.md): a shared library for gpu accelerated direction centering and phase calibration
* [leap-accelerate-cli](src/icrar/leap-accelerate-cli/ReadMe.md): a CLI interface for I/O datastream or plasma data access 

## Usage

See [leap-accelerate-cli](src/icrar/leap-accelerate-cli/ReadMe.md)

Example Usage:

`LeapAccelerateCLI --help`

`LeapAccelerateCLI --config "./askap.json"`

`LeapAccelerateCLI --config "./askap.json" --output "askap_calibration.json"`

## System Dependencies

### Recommended Versions Compatibility

* g++ 9.3.0
* cuda 10.1
* boost 1.71.0
* casacore 3.1.2

### Minimum Versions Compatibility

* g++ 6.3.0
* cuda 9.0
* boost 1.63.0
* cmake 3.15.1
* casacore 3.1.2

### Ubuntu/Debian Dependencies

Ubuntu Focal 20.04 LTS

* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10 libboost1.71-all-dev libgsl-dev
* https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal

Ubuntu Bionic 18.04 LTS

* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10 libboost1.65-all-dev libgsl-dev
* https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal

Ubuntu Xenial 16.04 LTS

* https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
* sudo apt-get install gcc-6 g++-6 gdb doxygen casacore-dev libboost1.58-all-dev libgsl-dev
* https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal

## Compiling from Source

### Linux

`export CUDA_HOME=/usr/local/cuda`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64`

`export PATH=$PATH:$CUDA_HOME/bin`

#### Debug

`mkdir -p build/Debug && cd build/Debug`

`cmake ../../ -DCMAKE_CXX_FLAGS_DEBUG="-g -O1" -DCMAKE_BUILD_TYPE=Debug`

with tracing:

`mkdir -p build/Debug && cd build/Debug`

`cmake ../../ -DCMAKE_CXX_FLAGS_DEBUG="-g -O1" -DCMAKE_BUILD_TYPE=Debug -DTRACE=ON`

#### Release

`mkdir -p build/Release && cd build/Release`

`cmake ../../ -DCMAKE_BUILD_TYPE=Release`

### Linux Cluster

`module load cmake/3.15.1 gcc/6.3.0 boost/1.66.0 casacore/3.1.2`

`module unload gfortran/default`

`module load isl/default`

`export CUDA_HOME=/usr/local/cuda`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64`

`export PATH=$PATH:$CUDA_HOME/bin`

`mkdir -p build/Release && cd build/Release`

`cmake ../../ -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCUDA_HOST_COMPILER=g++ -DCASACORE_ROOT_DIR=$BLDR_CASACORE_BASE_PATH -DCMAKE_BUILD_TYPE=Release`

## Test

Testing provided via googletest. To test using CTest use the following command in build/linux:

`make test` or `ctest`

for verbose output use:

`ctest --verbose` or `ctest --output-on-failure`

To test using the google test runner, the test binaries can be executed directly using the following commands:

`./src/icrar/leap-accelerate/tests/LeapAccelerateLib.Tests`
`./src/icrar/leap-accelerate-cli/tests/LeapAccelerateCLI.Tests`

## Documentation

Doxygen documentation can generated with the following target:

`make doxygen`

Generated doxygen is available at the following file location:

`src/out/html/index.html`
