
# Getting Started

Make sure you have installed the NVIDIA driver and Docker engine for your Linux distribution Note that you do not need to install the CUDA Toolkit on the host system, but the NVIDIA driver needs to be installed

* https://sarus.readthedocs.io/en/stable/user/custom-cuda-images.html
* https://github.com/NVIDIA/nvidia-docker
* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

## Install docker

```
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

## Install nvidia-docker-toolkit

```(bash)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
```

### pre docker 19.03

`sudo apt install -y nvidia-docker2`

### post docker 19.03

```(bash)
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
reboot
```

### check cuda

sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Create Docker Image

`sudo docker build -t bullseye-cuda . [container_name]`
`docker commit [container_name]`

## check image id

`docker images`

## tag and push

`docker tag [image_id] [hubname]/[image_name]:version`
`docker push [user]/[image_name]`


# Gitlab CI Runner

Setup gitlab-runner in docker mode:

`gitlab-runner start`

## Gitlab CI Runner Container

Execute runner from within a docker container

`docker run -d --name gitlab-runner-buster --restart always \
     -v /srv/gitlab-runner/config:/etc/gitlab-runner \
     -v /var/run/docker.sock:/var/run/docker.sock \
     [image_name]`

## Example

`sudo docker run -d --name gitlab-runner --restart always \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v gitlab-runner-config:/etc/gitlab-runner \
    --gpus all \
    gitlab/gitlab-runner:latest`
