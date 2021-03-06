#!/bin/bash
#
# Travis CI install script
#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
# All rights reserved
#
# Contributed by Callan Gray
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# MA 02111-1307  USA
#

# CUDA 11.1
#wget -nv https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-debian10-11-1-local_11.1.1-455.32.00-1_amd64.deb
#dpkg -i cuda-repo-debian10-11-1-local_11.1.1-455.32.00-1_amd64.deb
#apt-key add /var/cuda-repo-debian10-11-1-local/7fa2af80.pub
#add-apt-repository contrib

#apt-get update
#DEBIAN_FRONTEND=noninteractive apt-get -y install cuda

# Set Environment Config
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
