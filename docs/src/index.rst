.. Leap Accelerate documentation master file, created by
.. sphinx-quickstart on Mon Feb  8 17:21:19 2021.
.. You can adapt this file completely to your liking, but it should at least
.. contain the root `toctree` directive.

###############
Leap Accelerate
###############

.. image:: https://img.shields.io/badge/license-LGPL_2.1-blue
	:alt: Project License: LGPL 2.1
	:target: https://img.shields.io/badge/license-LGPL_2.1-blue

Leap Accelerate is a calibration tool implementing Low-frequency Excision of the Atmosphere in Parallel (`LEAP <https://arxiv.org/abs/1807.04685>`_) for low-frequency radio antenna arrays.
Leap utilizes GPGPU acceleration for parallel computation across baselines,
channels and polarizations and is freely available on `GitLab <https://gitlab.com/ska-telescope/icrar-leap-accelerate>`_ under the LGPLv2 License.

Leap Accelerate consists of:

* :ref:`api`: a shared library for accelerated direction centering and phase calibration.
* :ref:`cli`: a CLI interface for I/O datastream or plasma data access.

.. toctree::
    :caption: Leap Accelerate

    index

.. toctree::
    :caption: Library Documentation
    :maxdepth: 2

    api.rst
    cli.rst
    api/library_root.rst

.. toctree::
    :caption: Install
    :maxdepth: 2
    :glob:

    md/Build

.. toctree::
    :caption: Usage
    :maxdepth: 2
    :glob:

    md/LeapAccelerateCLI

.. about.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
.. * :ref:`DocMap`

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


