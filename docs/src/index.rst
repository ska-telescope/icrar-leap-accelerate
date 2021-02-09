.. Leap Accelerate documentation master file, created by
.. sphinx-quickstart on Mon Feb  8 17:21:19 2021.
.. You can adapt this file completely to your liking, but it should at least
.. contain the root `toctree` directive.

.. toctree::
    :maxdepth: 3
    :caption: Contents:

Leap Accelerate
===============

![License](https://img.shields.io/badge/license-LGPL_2.1-blue)

LEAP-Accelerate is a calibration tool implementing Low-frequency Excision of the Atmosphere in Parallel ([LEAP](https://arxiv.org/abs/1807.04685)) for low-frequency radio antenna arrays. Leap utilizes GPGPU acceleration for parallel computation across baselines, channels and polarizations and is freely available on [GitLab](https://gitlab.com/ska-telescope/icrar-leap-accelerate) under the LGPLv2 [License](LICENSE).

LEAP-Accelerate includes:

* [leap-accelerate-lib](src/icrar/leap-accelerate/ReadMe.md): a shared library for gpu accelerated direction centering and phase calibration.
* [leap-accelerate-cli](src/icrar/leap-accelerate-cli/ReadMe.md): a CLI interface for I/O datastream or plasma data access.


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
* :ref:`DocMap`


Installation
============

.. See the [build](docs/src/Build.md) documentation for instructions on platform specific compilation.

Usage
=====

See [leap-accelerate-cli](src/icrar/leap-accelerate-cli/ReadMe.md) for instructions on command line arguments and configuration files.

Examples:

`LeapAccelerateCLI --help`

`LeapAccelerateCLI --config "./askap.json"`

Contributions
=============

Refer to the following style guides for making repository contributions

.. * [CMake Style Guide](docs/src/CMakeStyleGuide.md)
.. * [C++ Style Guide](docs/src/CPlusPlusStyleGuide.md)


Docs
====

.. autodoxysummary::
    :toctree: generated/
    :template: doxyclass.rst

    icrar::ILeapCalibrator
    icrar::cpu::CpuLeapCalibrator
    icrar::cuda::CudaLeapCalibrator
