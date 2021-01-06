apt update
apt install -y cmake jq clang clang-tidy clang-tools iwyu cppcheck
apt install -y git moreutils wget
apt install -y doxygen graphviz casacore-dev libboost-all-dev

#CUDA
apt install -y gnupg2 software-properties-common
wget -nv https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-debian10-11-1-local_11.1.1-455.32.00-1_amd64.deb
dpkg -i cuda-repo-debian10-11-1-local_11.1.1-455.32.00-1_amd64.deb
apt-key add /var/cuda-repo-debian10-11-1-local/7fa2af80.pub
add-apt-repository contrib
