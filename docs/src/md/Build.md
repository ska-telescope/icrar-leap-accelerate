
# Installation

## Compiling from Source

leap-accelerate compilation is compatible with g++ and clang++ on debian or ubuntu. Support for compiling on other operating systems is currently experimental.

### Dependencies

#### Recommended Versions Compatibility

* g++ 9.3.0
* cuda 10.1
* boost 1.71.0
* casacore 3.1.2

#### Minimum Versions Compatibility

* g++ 6.3.0
* cuda 9.0
* boost 1.63.0
* cmake 3.15.1
* casacore 3.1.2

#### Ubuntu/Debian Dependencies

##### 20.04 LTS

* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10 libboost1.71-all-dev libgsl-dev
* https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal

##### 18.04 LTS

* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10 libboost1.65-all-dev libgsl-dev
* https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal

##### 16.04 LTS

* https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
* sudo apt-get install gcc-6 g++-6 gdb doxygen casacore-dev libboost1.58-all-dev libgsl-dev
* https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal

#### CMake Options

Use `cmake .. -D<OPTION>=<VALUE> ...` or `ccmake ..` to set cmake options.

Setting an environment variable of the same name will also override these cmake options:

`CUDA_ENABLED` - Enables building with cuda support

`HIGH_GPU_MEMORY` - Optimizes device performance at the cost of extra device memory

`WERROR` - Enables warnings as Errors

`WCONVERSION` - Enables warnings on implicit numeric conversions

`TRACE` - Traces data to the local directory

`CMAKE_RUN_CLANG_TIDY` - Enables running clang-tidy with the compiler

#### Compile Commands

From the repository root folder run:

`git submodule update --init --recursive`

NOTE: pulling exernal submodules is now automated by CMake. When downloading the source files via tools other than git the folder `external/` will need to be copied manually.

##### Linux

`export CUDA_HOME=/usr/local/cuda`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64`

`export PATH=$PATH:$CUDA_HOME/bin`

###### Debug

`mkdir -p build/Debug && cd build/Debug`

`cmake ../../ -DCMAKE_CXX_FLAGS_DEBUG="-g -O1" -DCUDA_ENABLED=TRUE -DCMAKE_BUILD_TYPE=Debug`

With tracing to file:

`cmake ../../ -DCMAKE_CXX_FLAGS_DEBUG="-g -O1" -DCUDA_ENABLED=TRUE -DTRACE=ON -DCMAKE_BUILD_TYPE=Debug`

###### Release

`mkdir -p build/Release && cd build/Release`

`cmake ../../ -DCUDA_ENABLED=TRUE -DCMAKE_BUILD_TYPE=Release`

## Testing

Testing provided via googletest. To test using the google test runner, test binaries can be executed directly using the following commands:

`./src/icrar/leap-accelerate/tests/LeapAccelerate.Tests`
`./src/icrar/leap-accelerate-cli/tests/LeapAccelerateCLI.Tests`

To test using CTest use the following command in build/linux:

`make test` or `ctest`

for verbose output use:

`ctest --verbose` or `ctest --output-on-failure`

## Documentation

Generated documentation is available locally at the following file location:

`docs/sphinx/index.html`

Once deployed to a branch the docs will be available here:

https://icrar-leap-accelerate.readthedocs.io/

### Building from Source

Doxygen documentation is generated for all C++ and cuda files with the following target:

`make doxygen`

Sphinx/Breath/Exhale docs is a dependent target generated with the following command:

`make sphinx`
