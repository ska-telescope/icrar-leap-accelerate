.. _api:

Leap Accelerate API
===================

Leap Accelerate API is a library of components available for use by calibration applications such
as Leap Accelerate CLI.


Components
==========

* :ref:`core`
* :ref:`log`
* :ref:`common`
* :ref:`model`
* :ref:`algorithm`
* :ref:`math`
* :ref:`cuda`
* :ref:`ms`

Getting Started
===============

import the leap-accelerate cmake target and add the following include:

.. code-block:: cpp

    #include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>

create a calibrator object using the factory method and an output callback:

.. code-block:: cpp

    std::vector<cpu::Calibration> calibrations;
    auto outputCallback = [&](const cpu::Calibration& calibration)
    {
        calibrations.push_back(calibration);
    };
    
    LeapCalibratorFactory::Create(args.GetComputeImplementation())->Calibrate(
        outputCallback,
        args.GetMeasurementSet(),
        args.GetDirections(),
        args.GetSolutionInterval(),
        args.GetMinimumBaselineThreshold(),
        args.GetReferenceAntenna(),
        args.IsFileSystemCacheEnabled());


.. .. doxygenindex::
..    :project: LeapAccelerate

.. toctree::
    :maxdepth: 4
    :caption: Contents:

