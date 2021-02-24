
module load cmake/3.15.1 gcc/6.3.0 boost/1.66.0 casacore/3.1.2
module unload gfortran/default
module load isl/default
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin

mkdir build && cd build
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCUDA_ENABLED=TRUE -DHIGH_GPU_MEMORY=TRUE -DCUDA_HOST_COMPILER=g++ -DCASACORE_ROOT_DIR=$BLDR_CASACORE_BASE_PATH -DCMAKE_BUILD_TYPE=Release
make LeapAccelerate LeapAccelerateCLI -j 2