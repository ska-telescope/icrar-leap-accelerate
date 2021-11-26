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

#if PYTHON_ENABLED

#include "PyLeapCalibrator.h"
#include "PyMeasurementSet.h"

#include <future>

namespace py = pybind11;

template<typename T>
inline T ternary(bool condition, T trueValue, T falseValue)
{
    return condition ? trueValue : falseValue;
}

namespace icrar
{
namespace python
{
    template<typename T>
    inline boost::optional<T> ToOptional(const py::object& obj)
    {
        boost::optional<T> output;
        if(!obj.is_none())
        {
            output = obj.cast<T>();
        }
        return output;
    }

    Slice ToSlice(const py::slice& obj)
    {
        return Slice(
            ToOptional<int64_t>(obj.attr("start")),
            ToOptional<int64_t>(obj.attr("stop")),
            ToOptional<int64_t>(obj.attr("step"))
        );
    }

    std::vector<SphericalDirection> ToSphericalDirectionVector(const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions)
    {
        auto output = std::vector<SphericalDirection>();
        for(int64_t row = 0; row < directions.rows(); ++row)
        {
            output.push_back(directions(row, Eigen::all));
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
        m_calibrator = other.m_calibrator;
    }

    void PyLeapCalibrator::Calibrate(
        const MeasurementSet& measurementSet,
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
        const Slice& solutionInterval,
        const std::function<void(const cpu::Calibration&)>& callback)
    {
        auto validatedDirections = ToSphericalDirectionVector(directions);
        double minimumBaselineThreshold = 0.0;
        int referenceAntenna = 0;
        ComputeOptionsDTO computeOptions = {boost::none, boost::none, boost::none};

        m_calibrator->Calibrate(
            callback,
            measurementSet,
            validatedDirections,
            solutionInterval,
            minimumBaselineThreshold,
            referenceAntenna,
            computeOptions);
    }

    void PyLeapCalibrator::PythonCalibrate(
        const PyMeasurementSet& measurementSet,
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
        const pybind11::slice& solutionInterval,
        const std::function<void(const cpu::Calibration&)>& callback)
    {
        Calibrate(
            measurementSet.Get(),
            directions,
            ToSlice(solutionInterval),
            callback);
    }

    void PyLeapCalibrator::CalibrateToFile(
        const MeasurementSet& measurementSet,
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
        const Slice& solutionInterval,
        boost::optional<std::string> outputPath)
    {
        auto validatedDirections = ToSphericalDirectionVector(directions);
        double minimumBaselineThreshold = 0.0;
        int referenceAntenna = 0;
        ComputeOptionsDTO computeOptions = {boost::none, boost::none, boost::none};

        std::vector<cpu::Calibration> calibrations;
        m_calibrator->Calibrate(
            [&](const cpu::Calibration& cal) { calibrations.push_back(cal); },
            measurementSet,
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
        else
        {
            calibrationCollection.Serialize(std::cout);
        }
    }

    void PyLeapCalibrator::PythonCalibrateToFile(
        const std::string& msPath,
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
        const py::slice& solutionInterval,
        const py::object& outputPath)
    {
        icrar::log::Initialize(icrar::log::Verbosity::warn);
        auto measurementSet = std::make_unique<MeasurementSet>(msPath);
        CalibrateToFile(
            *measurementSet,
            directions,
            ToSlice(solutionInterval),
            ToOptional<std::string>(outputPath));
    }
} // namespace python
} // namespace icrar

#endif // PYTHON_ENABLED