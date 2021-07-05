# This is Dockerfile installs everything from scratch into a Ubuntu 20.04 based container
FROM ubuntu:20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    gnupg2 git wget gcc g++ gdb doxygen cmake casacore-dev libboost1.71-all-dev \
    software-properties-common

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin &&\
     mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt update &&\
    DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install cuda-minimal-build-11-2

RUN git clone https://github.com/illuhad/hipSYCL.git &&\
    cd hipSYCL/install/scripts &&\
    bash add-hipsycl-repo/ubuntu-20.04.sh &&\
    bash packaging/make-ubuntu-cuda-pkg.sh &&\
    apt install -y hipsycl-cuda-nightly

# Get the LEAP sources and install them in the system
COPY / /leap-accelerate
RUN cd /leap-accelerate && git submodule update --init --recursive &&\
    export CUDA_HOME=/usr/local/cuda &&\
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64 &&\
    export PATH=$PATH:$CUDA_HOME/bin &&\
    mkdir -p build/linux/Debug && cd build/linux/Debug &&\
    cmake ../../.. -DCMAKE_CXX_FLAGS_DEBUG=-O1 -DCMAKE_BUILD_TYPE=Debug &&\
    make && make install

# Second stage to cleanup the mess
FROM ubuntu:20.04
COPY --from=0 /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY --from=0 /usr/local /usr/local
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends liblapack3

# add a user to run this container under rather than root.
RUN useradd ray
USER ray
WORKDIR /home/ray
CMD ["/usr/local/bin/LeapAccelerateCLI", "--help"]
