# LEAP Accelerate

![License](https://img.shields.io/badge/license-LGPL_2.1-blue)

LEAP-Accelerate is a calibration tool implementing Low-frequency Excision of the Atmosphere in Parallel ([LEAP](https://arxiv.org/abs/1807.04685)) for low-frequency radio antenna arrays. Leap utilizes GPGPU acceleration for parallel computation across baselines, channels and polarizations and is freely available on [GitLab](https://gitlab.com/ska-telescope/icrar-leap-accelerate) under the LGPLv2 [License](LICENSE).

LEAP-Accelerate includes:

* [leap-accelerate-lib](src/icrar/leap-accelerate/ReadMe.md): a shared library for gpu accelerated direction centering and phase calibration.
* [leap-accelerate-cli](src/icrar/leap-accelerate-cli/ReadMe.md): a CLI interface for I/O datastream or plasma data access.
<!---* leap-accelerate-client: a socket client interface for processing data from a LEAP-Cal server--->
<!---* leap-accelerate-server: a socket server interface for dispatching data processing to LEAP-Cal clients--->

See the [online documentation](https://developer.skatelescope.org/projects/icrar-leap-accelerate/en/latest/) for more information.

## Installation

The latest leap release is published as a debian a docker image available at the following location:

`nexus.engageska-portugal.pt/ska-docker/icrar-leap-accelerate:latest`

This image can be run locally using the following command:

`docker run -it --rm nexus.engageska-portugal.pt/ska-docker/icrar-leap-accelerate:latest LeapAccelerateCLI --help`

See the [docker](docs/src/md/Docker.md) documentation for instructions about how to create a docker image.

See the [build](docs/src/md/Build.md) documentation for instructions on platform specific compilation.

## Usage

See [leap-accelerate-cli](docs/src/md/LeapAccelerateCLI.md) for instructions on command line arguments and configuration files.

Examples:

`LeapAccelerateCLI --help`

`LeapAccelerateCLI --config "./askap.json"`

## Contributions

Refer to the following style guides for making repository contributions

* [CMake Style Guide](docs/src/md/specs/CMakeStyleGuide.md)
* [C++ Style Guide](docs/src/md/specs/CPlusPlusStyleGuide.md)
