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
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <Eigen/Core>

#include <boost/python.hpp>
#include <boost/python/call.hpp>
#include <boost/python/numpy.hpp>

#include <iostream>

namespace np = boost::python::numpy;

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
        std::unique_ptr<MeasurementSet> m_measurementSet;
        std::unique_ptr<ILeapCalibrator> m_calibrator;

    public:
        PyLeapCalibrator(ComputeImplementation impl)
        {
            m_calibrator = LeapCalibratorFactory::Create(impl);
        }

        PyLeapCalibrator(std::string impl)
        {
            m_calibrator = LeapCalibratorFactory::Create(ParseComputeImplementation(impl));
        }

        PyLeapCalibrator(const PyLeapCalibrator& other) {}

        // void Calibrate(
        //     std::function<void(const cpu::Calibration&)> outputCallback,
        //     const icrar::MeasurementSet& ms,
        //     const std::vector<SphericalDirection>& directions,
        //     const Slice& solutionInterval,
        //     double minimumBaselineThreshold,
        //     boost::optional<unsigned int> referenceAntenna,
        //     const ComputeOptionsDTO& computeOptions) override
        // {
        //     m_calibrator->Calibrate(
        //         outputCallback,
        //         ms,
        //         directions,
        //         solutionInterval,
        //         minimumBaselineThreshold,
        //         referenceAntenna,
        //         computeOptions);
        // }

        std::vector<SphericalDirection> ToSphericalDirectionVector(const np::ndarray& array)
        {
            assert(array.get_nd() == 2);
            assert(array.shape(0) == 2);
            assert(array.get_dtype().get_itemsize() == 8); // double
            auto directionMatrix = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
                reinterpret_cast<double*>(array.get_data()), array.shape(0), array.shape(1)
            );

            auto output = std::vector<SphericalDirection>();
            output.reserve(directionMatrix.rows());
            for(int row = 0; row < directionMatrix.rows(); ++row)
            {
                output.push_back(directionMatrix(row, Eigen::all));
            }
            return output;
        }

        void Calibrate(
            PyObject* callback,
            std::string msPath,
            bool useAutoCorrelations,
            const np::ndarray& directions)
        {
            m_measurementSet = std::make_unique<MeasurementSet>(
                    msPath,
                    boost::none,
                    useAutoCorrelations);
            auto validatedDirections = ToSphericalDirectionVector(directions);
            auto solutionInterval = Slice(0,1,1);
            double minimumBaselineThreshold = 0.0;
            int referenceAntenna = 0;
            ComputeOptionsDTO computeOptions = {false, false, false};

            m_calibrator->Calibrate(
                [&](const cpu::Calibration& cal)
                {
                    boost::python::call<void>(callback);
                },
                *m_measurementSet,
                validatedDirections,
                solutionInterval,
                minimumBaselineThreshold,
                referenceAntenna,
                computeOptions);
        }
    };
} // namespace python
} // namespace icrar


BOOST_PYTHON_MODULE(LeapAccelerate)
{
    boost::python::numpy::initialize();

    boost::python::class_<icrar::python::PyLeapCalibrator>("LeapCalibrator", boost::python::init<icrar::ComputeImplementation>())
        .def(boost::python::init<std::string>())
        .def("Calibrate", &icrar::python::PyLeapCalibrator::Calibrate);

    boost::python::enum_<icrar::ComputeImplementation>("ComputeImplementation")
        .value("cpu", icrar::ComputeImplementation::cpu)
        .value("cuda", icrar::ComputeImplementation::cuda);
}

