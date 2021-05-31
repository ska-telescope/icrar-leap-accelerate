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

#include "PyLeapCalibrator.h"

namespace icrar
{
namespace python
{
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

    PyLeapCalibrator::PyLeapCalibrator(ComputeImplementation impl)
    {
        m_calibrator = LeapCalibratorFactory::Create(impl);
    }

    PyLeapCalibrator::PyLeapCalibrator(std::string impl)
    {
        m_calibrator = LeapCalibratorFactory::Create(ParseComputeImplementation(impl));
    }

    PyLeapCalibrator::PyLeapCalibrator(const PyLeapCalibrator& other)
    {
        m_measurementSet = other.m_measurementSet;
        m_calibrator = other.m_calibrator;
    }

    void PyLeapCalibrator::Calibrate(
        PyObject* callback,
        std::string msPath,
        bool useAutoCorrelations,
        const np::ndarray& directions,
        std::optional<std::string> outputPath)
    {
        m_measurementSet = std::make_unique<MeasurementSet>(msPath, boost::none, useAutoCorrelations);
        auto validatedDirections = ToSphericalDirectionVector(directions);
        auto solutionInterval = Slice(0,1,1);
        double minimumBaselineThreshold = 0.0;
        int referenceAntenna = 0;
        ComputeOptionsDTO computeOptions = {false, false, false};

        std::vector<cpu::Calibration> calibrations;
        
        auto outputCallback = [&](const cpu::Calibration& cal)
        {
            calibrations.push_back(cal);
            if(callback != nullptr)
            {
                boost::python::call<void>(callback);
            }
        };

        m_calibrator->Calibrate(
            outputCallback,
            *m_measurementSet,
            validatedDirections,
            solutionInterval,
            minimumBaselineThreshold,
            referenceAntenna,
            computeOptions);

        auto calibrationCollection = cpu::CalibrationCollection(std::move(calibrations));

        if(outputPath.has_value())
        {
            std::ofstream file(outputPath.value());
            calibrationCollection.Serialize(file);
        }
    }
    
    void PyLeapCalibrator::PythonCalibrate(
        PyObject* callback,
        boost::python::object& msPath,
        boost::python::object& useAutoCorrelations,
        const np::ndarray& directions,
        boost::python::object& outputPath)
    {
        Calibrate(
            callback,
            bp::extract<std::string>(msPath),
            bp::extract<bool>(useAutoCorrelations),
            directions,
            PythonToOptional<std::string>(outputPath));
    }
} // namespace python
} // namespace icrar


BOOST_PYTHON_MODULE(LeapAccelerate)
{
    boost::python::numpy::initialize();

    boost::python::class_<icrar::python::PyLeapCalibrator>("LeapCalibrator", boost::python::init<icrar::ComputeImplementation>())
        .def(boost::python::init<std::string>())
        .def("Calibrate", &icrar::python::PyLeapCalibrator::PythonCalibrate, (
            boost::python::arg("callback"),
            boost::python::arg("ms_path"),
            boost::python::arg("autocorrelations"),
            boost::python::arg("directions"),
            boost::python::arg("output_path")=boost::python::object()
        ));

    boost::python::enum_<icrar::ComputeImplementation>("ComputeImplementation")
        .value("cpu", icrar::ComputeImplementation::cpu)
        .value("cuda", icrar::ComputeImplementation::cuda);
}
