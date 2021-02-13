.. _api:

Leap Accelerate API
===================

Leap Accelerate API consists of the following components:

* :ref:`core`
* :ref:`log`
* :ref:`common`
* :ref:`model`
* :ref:`algorithm`
* :ref:`math`
* :ref:`cuda`
* :ref:`ms`

.. _core:

core
====

.. doxygenenum:: icrar::ComputeImplementation
    :project: LeapAccelerate

.. doxygenenum:: icrar::StreamOutType
    :project: LeapAccelerate

.. .. autodoxygenfile:: stream_out_type.h
..     :project: auto

.. _common:

common
======

.. _model:

.. _log:

log
====

.. doxygenenum:: icrar::log::Verbosity
    :project: LeapAccelerate

model
=====

.. doxygenclass:: icrar::cpu::MetaData
    :project: LeapAccelerate
    :members:

.. doxygenclass:: icrar::cpu::Integration
    :project: LeapAccelerate
    :members:

.. doxygenclass:: icrar::cpu::
    :project: LeapAccelerate
    :members:

.. _algorithm:

algorithm
=========

index
-----

TODO

classes
-------

.. doxygenclass:: icrar::ILeapCalibrator
    :project: LeapAccelerate
    :members:

.. doxygenclass:: icrar::cpu::CpuLeapCalibrator
    :project: LeapAccelerate
    :members:

.. doxygenclass:: icrar::cuda::CudaLeapCalibrator
    :project: LeapAccelerate
    :members:

.. _cuda:

cuda
====


.. toctree::
    :maxdepth: 4
    :caption: Contents:

