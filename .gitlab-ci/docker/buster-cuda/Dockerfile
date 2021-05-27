FROM debian:buster-slim

RUN apt update &&\
    apt install -y wget gnupg2 software-properties-common &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/7fa2af80.pubsudo &&\
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/ /" &&\
    add-apt-repository contrib &&\
    apt update &&\
    mkdir -p /usr/share/man/man1

RUN DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
    cuda-minimal-build-10-1 cuda-libraries-10-1 cuda-libraries-dev-10-1 &&\
    ln -s /usr/local/cuda-10.1 /usr/local/cuda &&\
    ln -s /usr/local/cuda/targets/x86_64-linux/lib /usr/local/cuda/lib &&\
    ln -s /usr/local/cuda/targets/x86_64-linux/include /usr/local/cuda/include

# Leap build dependencies
# RUN apt install doxygen graphviz casacore-dev libboost-all-dev rapidjson-dev
