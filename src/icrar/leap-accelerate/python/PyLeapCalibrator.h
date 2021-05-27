
#pragma once

#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>
#include <boost/python.hpp>
#include <iostream>

namespace icrar
{
namespace python
{
    /**
     * @brief An adapter to a boost::python compatible class
     * @note the Calibrate signature may change as needed
     */
    class PyLeapCalibrator // : public ILeapCalibrator
    {
        std::unique_ptr<ILeapCalibrator> m_calibrator;
    public:
        PyLeapCalibrator(ComputeImplementation impl)
        {
            m_calibrator = LeapCalibratorFactory::Create(impl);
        }

        PyLeapCalibrator(const PyLeapCalibrator& other) {}

        void Hello()
        {
            std::cout << "hello world" << std::endl;
        }

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
    };
} // namespace python
} // namespace icrar


BOOST_PYTHON_MODULE(leap_accelerate)
{
    boost::python::class_<icrar::python::PyLeapCalibrator>("LeapCalibrator", boost::python::init<icrar::ComputeImplementation>())
        .def("hello",     &icrar::python::PyLeapCalibrator::Hello);
        //.def("calibrate", &icrar::python::PyLeapCalibrator::Calibrate);

    boost::python::enum_<icrar::ComputeImplementation>("ComputeImplementation")
        .value("cpu", icrar::ComputeImplementation::cpu)
        .value("cuda", icrar::ComputeImplementation::cuda);
}

