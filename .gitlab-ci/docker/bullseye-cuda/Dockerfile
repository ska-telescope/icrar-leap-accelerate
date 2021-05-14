FROM debian:bullseye-slim

RUN apt update &&\
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin &&\
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub &&\
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" &&\
    apt update &&\
    mkdir -p /usr/share/man/man1

RUN DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
    cuda-minimal-build-11-2 cuda-libraries-11-2 cuda-libraries-dev-11-2 &&\
    ln -s /usr/local/cuda-11.2 /usr/local/cuda &&\
    ln -s /usr/local/cuda/targets/x86_64-linux/lib /usr/local/cuda/lib &&\
    ln -s /usr/local/cuda/targets/x86_64-linux/include /usr/local/cuda/include

# Leap build dependencies
# RUN apt install doxygen graphviz casacore-dev libboost-all-dev rapidjson-dev
