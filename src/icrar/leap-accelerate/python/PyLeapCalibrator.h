
#pragma once

#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>
#include <boost/python.hpp>

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
        PythonLeapCalibrator(ComputeImplementation impl)
        {
            m_calibrator = LeapCalibratorFactory::Create(impl);
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
            m_calibrator(
                outputCallback,
                ms,
                directions,
                solutionInterval,
                minimumBaselineThreshold,
                referenceAntenna,
                computeOptions);
        }
    }
}
}

using namespace boost::python;

BOOST_PYTHON_MODULE("leap_accelerate")
{
    boost::python::class_<icrar::python::PythonLeapCalibrator>("LeapCalibrator")
        .def("calibrate", icrar::python::PythonLeapCalibrator::Calibrate);
}

