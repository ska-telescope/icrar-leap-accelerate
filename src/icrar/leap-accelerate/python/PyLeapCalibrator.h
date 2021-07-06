/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

#pragma once

#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>
#include <icrar/leap-accelerate/model/cpu/calibration/CalibrationCollection.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <Eigen/Core>

#include <boost/python.hpp>
#include <boost/python/call.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/object.hpp>

#include <iostream>

namespace icrar
{
namespace python
{
    /**
     * @brief An adapter to a boost::python compatible class
     * @note the Calibrate signature may change as needed
     */
    class PyLeapCalibrator
    {
        std::shared_ptr<MeasurementSet> m_measurementSet;
        std::shared_ptr<ILeapCalibrator> m_calibrator;

    public:
        PyLeapCalibrator(ComputeImplementation impl);
        PyLeapCalibrator(std::string impl);
        PyLeapCalibrator(const PyLeapCalibrator& other);

        void Calibrate(
            std::string msPath,
            const boost::python::numpy::ndarray& directions,
            std::optional<std::string> outputPath);

        void Calibrate(
            std::string msPath,
            const boost::python::numpy::ndarray& directions,
            PyObject* callback);

        /**
         * @brief A boost python interop compatible signature
         */
        void PythonCalibrate(
            boost::python::object& msPath,
            const boost::python::numpy::ndarray& directions,
            boost::python::object& outputPath);

        void PythonPlasmaCalibrate(
            boost::python::object& plasmaTM,
            const boost::python::numpy::ndarray& directions,
            boost::python::object& outputPath);

        /**
         * @brief Performs calibration that 
         */
        boost::python::object PythonCalibrateAsync(
            boost::python::object& msPath,
            const boost::python::numpy::ndarray& directions,
            PyObject* callback);
    };
} // namespace python
} // namespace icrar


