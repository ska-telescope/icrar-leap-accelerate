# This is Dockerfile installs the leap-accelerate build dependencies from scratch into a debian bullseye based container
FROM debian:bullseye-slim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    gnupg2 software-properties-common wget git gcc g++ gdb cmake \
    casacore-dev libboost1.74-all-dev doxygen python3-pip clang clang-tidy clang-tools iwyu cppcheck

RUN wget -nv https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-debian10-11-2-local_11.2.2-460.32.03-1_amd64.deb &&\
    dpkg -i cuda-repo-debian10-11-2-local_11.2.2-460.32.03-1_amd64.deb &&\
    apt-key add /var/cuda-repo-debian10-11-2-local/7fa2af80.pub &&\
    add-apt-repository contrib &&\
    apt-get update &&\
    DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install cuda-minimal-build-11-2

RUN DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
    cuda-minimal-build-11-2 cuda-libraries-dev-11-2 &&\
    ln -s /usr/local/cuda-11.2 /usr/local/cuda &&\
    ln -s /usr/local/cuda/targets/x86_64-linux/lib /usr/local/cuda/lib &&\
    ln -s /usr/local/cuda/targets/x86_64-linux/include /usr/local/cuda/include

RUN wget https://gitlab.com/ska-telescope/ska-cicd-cpp-build-base/raw/master/images/ska-cicd-cpp-build-base/ctest2junit &&\
    wget https://gitlab.com/ska-telescope/ska-cicd-cpp-build-base/raw/master/images/ska-cicd-cpp-build-base/ctest2junit.xsl &&\
    mkdir -p /usr/local/share/ctest2junit &&\
    mv ./ctest2junit.xsl /usr/local/share/ctest2junit &&\
    mv ./ctest2junit /usr/local/bin &&\
    chmod +x /usr/local/bin/ctest2junit
