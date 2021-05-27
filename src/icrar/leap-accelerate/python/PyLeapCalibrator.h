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

#include <boost/python.hpp>
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
    class PyLeapCalibrator : public ILeapCalibrator
    {
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

        void Hello()
        {
            std::cout << "hello world" << std::endl;
        }

        void Calibrate(
            std::function<void(const cpu::Calibration&)> outputCallback,
            const icrar::MeasurementSet& ms,
            const std::vector<SphericalDirection>& directions,
            const Slice& solutionInterval,
            double minimumBaselineThreshold,
            boost::optional<unsigned int> referenceAntenna,
            const ComputeOptionsDTO& computeOptions) override
        {
            m_calibrator->Calibrate(
                outputCallback,
                ms,
                directions,
                solutionInterval,
                minimumBaselineThreshold,
                referenceAntenna,
                computeOptions);
        }

        void Calibrate(
            std::function<void()> outputCallback,
            std::string msPath,
            np::ndarray directions)
        {

        }
    };
} // namespace python
} // namespace icrar


BOOST_PYTHON_MODULE(LeapAccelerate)
{
    boost::python::class_<icrar::python::PyLeapCalibrator>("LeapCalibrator", boost::python::init<icrar::ComputeImplementation>())
        .def("hello",     &icrar::python::PyLeapCalibrator::Hello);
        //.def("calibrate", &icrar::python::PyLeapCalibrator::Calibrate);

    boost::python::enum_<icrar::ComputeImplementation>("ComputeImplementation")
        .value("cpu", icrar::ComputeImplementation::cpu)
        .value("cuda", icrar::ComputeImplementation::cuda);
}

