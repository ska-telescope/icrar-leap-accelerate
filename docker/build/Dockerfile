# This is Dockerfile installs the leap-accelerate build dependencies from scratch
FROM ubuntu:22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    gnupg2 software-properties-common wget git gcc g++ gdb cmake \
    casacore-dev libboost1.74-all-dev doxygen python3-pip clang clang-tidy clang-tools iwyu cppcheck

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin &&\
     mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub &&\
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"


RUN DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
    cuda-minimal-build-11-2 cuda-libraries-dev-11-7 &&\
    ln -s /usr/local/cuda-11.7 /usr/local/cuda &&\
    ln -s /usr/local/cuda/targets/x86_64-linux/lib /usr/local/cuda/lib &&\
    ln -s /usr/local/cuda/targets/x86_64-linux/include /usr/local/cuda/include

RUN wget https://gitlab.com/ska-telescope/ska-cicd-cpp-build-base/raw/master/images/ska-cicd-cpp-build-base/ctest2junit &&\
    wget https://gitlab.com/ska-telescope/ska-cicd-cpp-build-base/raw/master/images/ska-cicd-cpp-build-base/ctest2junit.xsl &&\
    mkdir -p /usr/local/share/ctest2junit &&\
    mv ./ctest2junit.xsl /usr/local/share/ctest2junit &&\
    mv ./ctest2junit /usr/local/bin &&\
    chmod +x /usr/local/bin/ctest2junit
